# rag_engine.py
import io
import logging
import os
import uuid

from functools import lru_cache
from typing import Iterable, Optional

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ==========================================
# LOGGING
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Persistent directory for RAG document vectors.
# Kept SEPARATE from ./chroma_memory (long-term chat memory) so the
# two stores can be nuked independently without affecting each other.
#
# On Railway this MUST point inside the mounted volume (e.g.
# /app/data/chroma_rag), otherwise every redeploy wipes the index.
# Locally it falls back to the module directory.
RAG_CHROMA_PATH = os.getenv(
    "CHROMA_RAG_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_rag"),       
)

# Stable collection name — never a random UUID, so uploads survive
# restarts and remain queryable.
RAG_COLLECTION_NAME = "rag_documents"


# ==========================================
# PERSISTENT CHROMA CLIENT (shared across instances)
# ==========================================

@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Single persistent Chroma client for RAG documents.

    Cached so repeated RAGService() instantiations share one
    underlying client (Chroma's PersistentClient is not safe to
    open multiple times against the same path).
    """
    # Ensure the directory exists before Chroma tries to open it.
    # On Railway the volume mount creates /app/data, but not the
    # chroma_rag subdirectory inside it.
    os.makedirs(RAG_CHROMA_PATH, exist_ok=True)

    logger.info("Opening persistent Chroma store at: %s", RAG_CHROMA_PATH)
    return chromadb.PersistentClient(path=RAG_CHROMA_PATH)


# ==========================================
# EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ==========================================
# LOAD UPLOADED DOCUMENTS
# ==========================================

def load_uploaded_documents(
    uploaded_files: Iterable
) -> list[Document]:

    documents: list[Document] = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        extension = os.path.splitext(file_name)[1].lower()

        # --------------------------------------
        # PDF
        # --------------------------------------

        if extension == ".pdf":

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_name,
                                "page": page_number,
                            },
                        )
                    )

        # --------------------------------------
        # TXT
        # --------------------------------------

        elif extension == ".txt":

            text = file_bytes.decode("utf-8")

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": 1,
                            "file_type": "text",
                        },
                    )
                )

        # --------------------------------------
        # UNSUPPORTED FILE
        # --------------------------------------

        else:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

    # ------------------------------------------
    # CHECK DOCUMENTS
    # ------------------------------------------

    if not documents:

        raise ValueError(
            "No documents found in the uploaded files"
        )

    return documents


# ==========================================
# FORMAT RETRIEVED CONTEXT
# ==========================================

def format_context(
    documents: list[Document]
) -> str:

    blocks = []

    for index, doc in enumerate(
        documents,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        blocks.append(
            f"""
[Source {index}: {source}, Page {page}]
{doc.page_content}
"""
        )

    return "\n\n".join(blocks)


# ==========================================
# RAG SERVICE
# ==========================================

class RAGService:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ) -> None:

        # --------------------------------------
        # CHECK GROQ API KEY
        # --------------------------------------

        if not os.getenv("GROQ_API_KEY"):

            raise ValueError(
                "GROQ_API_KEY is not set"
            )

        # --------------------------------------
        # CONFIGURATION
        # --------------------------------------

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # --------------------------------------
        # LOAD EMBEDDING MODEL
        # --------------------------------------

        self.embedding_model = get_embedding_model()

        # --------------------------------------
        # LOAD GROQ LLM
        # --------------------------------------

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_retries=2,
        )

        # --------------------------------------
        # PROMPT
        # --------------------------------------

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a document question-answering assistant.

Answer the user's question using ONLY the context below.

Rules:

1. Do not use outside knowledge.

2. If the answer is not available in the context,
say exactly:

"I couldn't find that information in the uploaded documents."

3. Keep the answer clear and concise.

4. When useful, mention the source filename
and page number.

Context:

{context}
""",
                ),
                (
                    "human",
                    "Question: {question}",
                ),
            ]
        )

        # --------------------------------------
        # ANSWER CHAIN
        # --------------------------------------

        self.answer_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        # --------------------------------------
        # PERSISTENT CHROMA CLIENT
        # --------------------------------------

        self.chroma_client = get_chroma_client()

        # --------------------------------------
        # ATTACH TO EXISTING COLLECTION IF PRESENT
        # --------------------------------------
        # On startup, if a previous run already populated
        # "rag_documents", reconnect to it so uploaded files
        # survive restarts. If nothing exists yet, vector_store
        # stays None until the first upload.

        self.vector_store: Optional[Chroma] = None
        self.retriever = None

        self.documents: list[Document] = []
        self.chunks: list[Document] = []

        self.document_names: list[str] = []

        self._reconnect_existing_collection()

    # ==========================================
    # RECONNECT TO EXISTING COLLECTION (on startup)
    # ==========================================

    def _reconnect_existing_collection(self) -> None:
        """
        If a persistent 'rag_documents' collection exists on disk,
        attach to it and rebuild the in-memory tracking lists from
        the stored metadata (source filenames).
        """
        try:
            existing = self.chroma_client.list_collections()
        except Exception as e:
            logger.warning("Could not list collections: %s", e)
            return

        names = [c.name if hasattr(c, "name") else c for c in existing]
        if RAG_COLLECTION_NAME not in names:
            return

        try:
            self.vector_store = Chroma(
                collection_name=RAG_COLLECTION_NAME,
                embedding_function=self.embedding_model,
                client=self.chroma_client,
            )

            # Rebuild document_names from stored metadata
            store = self.vector_store.get(include=["metadatas"])
            sources = set()
            for meta in (store.get("metadatas") or []):
                if meta and meta.get("source"):
                    sources.add(meta["source"])

            self.document_names = sorted(sources)

            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k},
            )

            logger.info(
                "Reconnected to existing collection with %d document(s).",
                len(self.document_names),
            )

        except Exception as e:
            logger.warning("Could not reconnect to collection: %s", e)
            self.vector_store = None
            self.document_names = []
            self.retriever = None

    # ==========================================
    # BUILD VECTOR INDEX (replace everything)
    # ==========================================

    def build_index(
        self,
        uploaded_files: Iterable
    ) -> dict:

        # --------------------------------------
        # LOAD DOCUMENTS
        # --------------------------------------

        self.documents = load_uploaded_documents(
            uploaded_files
        )

        # --------------------------------------
        # SPLIT DOCUMENTS
        # --------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        self.chunks = splitter.split_documents(
            self.documents
        )

        # --------------------------------------
        # RESET COLLECTION (wipe and rebuild)
        # --------------------------------------
        # For replace semantics: delete any existing collection
        # with this name, then create a fresh one.

        try:
            self.chroma_client.delete_collection(RAG_COLLECTION_NAME)
            logger.info("Deleted old collection '%s'.", RAG_COLLECTION_NAME)
        except Exception:
            # Collection didn't exist — fine.
            pass

        self.vector_store = Chroma(
            collection_name=RAG_COLLECTION_NAME,
            embedding_function=self.embedding_model,
            client=self.chroma_client,
        )

        # --------------------------------------
        # ADD DOCUMENT CHUNKS
        # --------------------------------------

        self.vector_store.add_documents(
            documents=self.chunks
        )

        # --------------------------------------
        # CREATE RETRIEVER (unfiltered, default k)
        # --------------------------------------

        self.retriever = (
            self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.top_k
                },
            )
        )

        # --------------------------------------
        # TRACK DOCUMENT NAMES FOR SELECTION UI
        # --------------------------------------

        self.document_names = sorted(
            {
                doc.metadata.get("source", "unknown")
                for doc in self.documents
            }
        )

        # --------------------------------------
        # EMBEDDING DIMENSION
        # --------------------------------------

        embedding_dimension = len(
            self.embedding_model.embed_query(
                "dimension check"
            )
        )

        # --------------------------------------
        # RETURN INDEX INFORMATION
        # --------------------------------------

        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "document_names": self.document_names,
            "embedding_dimension": embedding_dimension,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GROQ_MODEL,
        }

    # ==========================================
    # LIST AVAILABLE DOCUMENT NAMES
    # ==========================================

    def get_document_names(self) -> list[str]:
        """Filenames currently indexed, for a 'select document' UI control."""
        return self.document_names

    # ==========================================
    # RETRIEVE (optionally scoped to one document)
    # ==========================================

    def retrieve(
        self,
        question: str,
        source: Optional[str] = None,
    ) -> list[Document]:

        if self.vector_store is None:

            raise RuntimeError(
                "Please process documents before asking a question."
            )

        search_kwargs = {"k": self.top_k}

        if source:

            if source not in self.document_names:

                raise ValueError(
                    f"Unknown document: {source!r}. "
                    f"Available documents: {self.document_names}"
                )

            search_kwargs["filter"] = {"source": source}

        scoped_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        return scoped_retriever.invoke(question)

    # ==========================================
    # ASK QUESTION
    # ==========================================

    def ask(
        self,
        question: str,
        source: Optional[str] = None,
    ) -> tuple[str, list[Document]]:

        retrieved_docs = self.retrieve(
            question,
            source=source,
        )

        context = format_context(
            retrieved_docs
        )

        answer = self.answer_chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return answer, retrieved_docs

    # ==========================================
    # APPEND DOCUMENTS (additive ingestion)
    # ==========================================

    def append_documents(
        self,
        uploaded_files: Iterable,
    ) -> dict:
        """
        Chunk + embed files and ADD to the existing Chroma collection.
        Does NOT recreate the collection (unlike build_index()).

        If no collection exists yet, falls back to build_index().
        """

        # --------------------------------------
        # FIRST UPLOAD EVER → full build
        # --------------------------------------

        if self.vector_store is None:

            return self.build_index(uploaded_files)

        # --------------------------------------
        # LOAD NEW DOCUMENTS
        # --------------------------------------

        new_documents = load_uploaded_documents(uploaded_files)

        # --------------------------------------
        # SPLIT NEW DOCUMENTS
        # --------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        new_chunks = splitter.split_documents(new_documents)

        # --------------------------------------
        # ADD TO EXISTING CHROMA COLLECTION
        # --------------------------------------

        self.vector_store.add_documents(
            documents=new_chunks
        )

        # --------------------------------------
        # UPDATE IN-MEMORY TRACKING
        # --------------------------------------

        self.documents.extend(new_documents)
        self.chunks.extend(new_chunks)

        for doc in new_documents:
            src = doc.metadata.get("source", "unknown")
            if src not in self.document_names:
                self.document_names.append(src)

        self.document_names = sorted(self.document_names)

        # --------------------------------------
        # REFRESH RETRIEVER
        # --------------------------------------

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # --------------------------------------
        # EMBEDDING DIMENSION
        # --------------------------------------

        embedding_dimension = len(
            self.embedding_model.embed_query(
                "dimension check"
            )
        )

        # --------------------------------------
        # RETURN INDEX INFORMATION
        # --------------------------------------

        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "document_names": self.document_names,
            "embedding_dimension": embedding_dimension,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GROQ_MODEL,
        }

    # ==========================================
    # DELETE DOCUMENT (by source filename)
    # ==========================================

    def delete_document(self, name: str) -> int:
        """
        Remove all chunks whose metadata['source'] == name.

        Returns -1 (Chroma's delete() doesn't return a count).
        Raises ValueError if the document is not in the index.
        """

        if self.vector_store is None:

            raise ValueError("No documents indexed yet.")

        if name not in self.document_names:

            raise ValueError(
                f"Document '{name}' not found in index."
            )

        # --------------------------------------
        # DELETE FROM CHROMA
        # --------------------------------------

        try:

            self.vector_store.delete(
                where={"source": name}
            )

        except Exception as e:

            raise ValueError(
                f"Failed to delete '{name}': {e}"
            )

        # --------------------------------------
        # UPDATE IN-MEMORY TRACKING
        # --------------------------------------

        self.document_names = [
            n for n in self.document_names if n != name
        ]

        self.documents = [
            d for d in self.documents
            if d.metadata.get("source") != name
        ]

        self.chunks = [
            c for c in self.chunks
            if c.metadata.get("source") != name
        ]

        # --------------------------------------
        # REFRESH RETRIEVER
        # --------------------------------------

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        return -1