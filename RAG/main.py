from contextlib import asynccontextmanager
from typing import List, Optional
import json
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGService
from agent import Agent

import chat_store
import memory_store
import memory_manager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# CONFIG
# =====================================================
# FIX: env-driven CORS. "*" + credentials is invalid per spec, and
# Streamlit talks to FastAPI server-to-server so CORS is mostly a no-op.
# If you ever add a browser-based client, list its exact origin here.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
] or ["*"]

# FIX: gate expensive startup backfill behind an env flag so Railway
# health checks aren't blocked by LLM calls during cold start.
RUN_STARTUP_BACKFILL = os.getenv("RUN_STARTUP_BACKFILL", "false").lower() == "true"


# =====================================================
# GLOBAL STATE
# =====================================================
state = {}


# =====================================================
# FILE SHIM
# =====================================================
class UploadedFileShim:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# =====================================================
# USER SCOPING
# =====================================================
def get_user_id(x_user_id: Optional[str]) -> str:
    """Read X-User-Id header, fall back to 'default'."""
    return x_user_id or "default"


# =====================================================
# LIFESPAN
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG service and agent...")
    rag_service = RAGService()
    state["agent"] = Agent(rag_service)

    # FIX: removed dead `state["sessions"] = {}` — never read anywhere.

    # ---- Persistence + memory bootstrap ----
    chat_store.init_db()
    logger.info("chat_store initialized")

    # FIX: startup backfill is now opt-in and non-blocking by default.
    # Railway cold-starts containers frequently; running N LLM calls here
    # delays /health and can trigger restart loops. Prefer the per-turn
    # background summarization path instead, or run backfill as a separate
    # cron service.
    if RUN_STARTUP_BACKFILL:
        try:
            llm = getattr(state["agent"], "llm", None)
            if llm is not None:
                drained = memory_manager.drain_summarization_queue(
                    user_id="default", llm=llm, max_sessions=5
                )
                logger.info("Startup memory backfill: %d sessions summarized", drained)
            else:
                logger.warning("Agent has no .llm attribute — skipping memory backfill")
        except Exception:
            logger.exception("Startup memory backfill failed (non-fatal)")

    logger.info("RAG API ready.")
    yield

    # ---- Shutdown: flush remaining sessions into memory ----
    try:
        llm = getattr(state.get("agent"), "llm", None)
        if llm is not None:
            drained = memory_manager.drain_summarization_queue(
                user_id="default", llm=llm, max_sessions=10
            )
            logger.info("Shutdown memory flush: %d sessions summarized", drained)
    except Exception:
        logger.exception("Shutdown flush failed (non-fatal)")

    state.clear()
    logger.info("RAG API shut down.")


app = FastAPI(title="RAG API", version="1.2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # FIX: credentials must be False when origin is "*", otherwise browsers
    # silently drop the response. Streamlit doesn't send cookies anyway.
    allow_credentials=False if ALLOWED_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# SCHEMAS
# =====================================================
class Source(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str = ""


class ChatRequest(BaseModel):
    question: str
    selected_document: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    session_id: Optional[str] = None
    memories_used: int = 0


class UploadResponse(BaseModel):
    documents: int
    chunks: int
    document_names: List[str]
    embedding_dimension: int
    embedding_model: str
    llm_model: str
    replaced: bool = False


class DocumentInfo(BaseModel):
    name: str


class DocumentsResponse(BaseModel):
    document_names: List[str]
    documents: List[DocumentInfo] = []


class SessionResponse(BaseModel):
    session_id: str
    messages: List[dict] = []


class ConversationMeta(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


# =====================================================
# HELPERS
# =====================================================
def get_agent() -> Agent:
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


def extract_sources(docs) -> List[Source]:
    sources: List[Source] = []
    seen = set()
    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source")
        page = meta.get("page")
        if not src:
            continue
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        content = getattr(doc, "page_content", "") or ""
        sources.append(Source(
            source=src,
            page=page if isinstance(page, int) else None,
            snippet=content[:300],
        ))
    return sources


# =====================================================
# 1. HEALTH (liveness — use this for Railway health check)
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================
# 2. READY (readiness — vector store + agent)
# =====================================================
@app.get("/ready")
def ready():
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        names = agent.rag.get_document_names()
    except Exception:
        # FIX: don't leak the internal exception text to the caller.
        logger.exception("Vector store readiness check failed")
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    return {"status": "ready", "documents": len(names)}


# =====================================================
# 3. UPLOAD (additive by default)
# =====================================================
@app.post("/upload", response_model=UploadResponse)
async def upload(
    files: List[UploadFile] = File(...),
    replace: bool = Query(False, description="If true, wipe index and rebuild"),
):
    agent = get_agent()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    shims: List[UploadedFileShim] = []
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        if not data:
            continue
        shims.append(UploadedFileShim(f.filename, data))

    if not shims:
        raise HTTPException(status_code=400, detail="All uploaded files were empty.")

    try:
        if replace:
            info = agent.rag.build_index(shims)
        else:
            info = agent.rag.append_documents(shims)
    except ValueError as e:
        # ValueError is a safe, user-facing validation error.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # FIX: log full traceback server-side, return generic message.
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail="Indexing failed")

    return UploadResponse(**info, replaced=replace)


# =====================================================
# 4. DOCUMENTS (list)
# =====================================================
@app.get("/documents", response_model=DocumentsResponse)
def documents():
    agent = get_agent()
    names = agent.rag.get_document_names()
    return DocumentsResponse(
        document_names=names,
        documents=[DocumentInfo(name=n) for n in names],
    )


# =====================================================
# 5. DELETE DOCUMENT
# =====================================================
@app.delete("/documents/{name}")
def delete_document(name: str):
    agent = get_agent()
    try:
        removed = agent.rag.delete_document(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Document delete failed")
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"deleted": name, "chunks_removed": removed}


# =====================================================
# 6. CHAT (with memory + persistence)
# =====================================================
@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    background: BackgroundTasks,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    agent = get_agent()
    user_id = get_user_id(x_user_id)

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- 1. Resolve / create conversation in SQLite ---
    conversation_id = req.session_id
    if not conversation_id:
        conversation_id = chat_store.create_conversation(
            title=req.question, user_id=user_id
        )
    elif not chat_store.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    # --- 2. Persist user turn ---
    chat_store.add_message(conversation_id, "user", req.question)

    # --- 3. Retrieve cross-session memories relevant to this query ---
    memory_block = memory_manager.build_memory_context(
        user_query=req.question,
        user_id=user_id,
        n_results=5,
        min_similarity=0.25,
    )
    memories_used = memory_block.count("- ") if memory_block else 0

    # --- 4. Call agent (with memory injected if agent supports it) ---
    try:
        if memory_block and hasattr(agent, "agent_with_memory"):
            answer, docs = agent.agent_with_memory(
                req.question, req.selected_document, memory_block
            )
        else:
            answer, docs = agent.agent(req.question, req.selected_document)
    except (RuntimeError, ValueError) as e:
        # These are typically validation / no-context errors.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail="Agent error")

    # --- 5. Persist assistant turn ---
    sources = extract_sources(docs)
    chat_store.add_message(
        conversation_id,
        "assistant",
        answer,
        sources=[s.model_dump() for s in sources] or None,
    )

    # --- 6. Every N turns, summarize in background ---
    history = chat_store.get_messages(conversation_id)
    if len(history) >= 6 and len(history) % 6 == 0:
        llm = getattr(agent, "llm", None)
        if llm is not None:
            background.add_task(
                memory_manager.commit_session_to_memory,
                conversation_id, user_id, llm,
            )

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=conversation_id,
        memories_used=memories_used,
    )


# =====================================================
# 7. CHAT STREAM (SSE, with memory + persistence)
# =====================================================
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    background: BackgroundTasks,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    agent = get_agent()
    user_id = get_user_id(x_user_id)

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Resolve / create conversation before streaming begins
    conversation_id = req.session_id
    if not conversation_id:
        conversation_id = chat_store.create_conversation(
            title=req.question, user_id=user_id
        )
    elif not chat_store.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    chat_store.add_message(conversation_id, "user", req.question)

    memory_block = memory_manager.build_memory_context(
        user_query=req.question,
        user_id=user_id,
        n_results=5,
        min_similarity=0.25,
    )
    memories_used = memory_block.count("- ") if memory_block else 0

    def event_gen():
        full_answer = ""
        sources_payload = []
        persisted = False

        try:
            # FIX: support both token-only streams and structured event streams.
            # If your agent already yields {"type": "sources", "docs": [...]},
            # we capture them here so the done event carries real sources.
            stream = None
            if memory_block and hasattr(agent, "agent_stream_with_memory"):
                stream = agent.agent_stream_with_memory(
                    req.question, req.selected_document, memory_block
                )
            elif hasattr(agent, "agent_stream"):
                stream = agent.agent_stream(req.question, req.selected_document)

            if stream is not None:
                for chunk in stream:
                    if isinstance(chunk, dict):
                        ctype = chunk.get("type")
                        if ctype == "token":
                            token = chunk.get("content", "")
                            full_answer += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        elif ctype == "sources":
                            sources_payload = [
                                s.model_dump()
                                for s in extract_sources(chunk.get("docs"))
                            ]
                        # Unknown chunk types are ignored for forward-compat.
                    else:
                        full_answer += chunk
                        yield f"data: {json.dumps({'token': chunk})}\n\n"

                # FIX: fallback — if the agent stashes retrieved docs on
                # itself (common pattern), pick them up here so the client
                # actually receives sources.
                if not sources_payload:
                    last_docs = getattr(agent, "last_docs", None)
                    if last_docs:
                        sources_payload = [
                            s.model_dump() for s in extract_sources(last_docs)
                        ]
            else:
                # Non-streaming fallback path
                answer, docs = agent.agent(req.question, req.selected_document)
                full_answer = answer
                yield f"data: {json.dumps({'token': answer})}\n\n"
                sources_payload = [s.model_dump() for s in extract_sources(docs)]

            # --- Persist assistant turn ---
            chat_store.add_message(
                conversation_id,
                "assistant",
                full_answer,
                sources=sources_payload or None,
            )
            persisted = True

            # --- Trigger background summarization every N turns ---
            history = chat_store.get_messages(conversation_id)
            if len(history) >= 6 and len(history) % 6 == 0:
                llm = getattr(agent, "llm", None)
                if llm is not None:
                    background.add_task(
                        memory_manager.commit_session_to_memory,
                        conversation_id, user_id, llm,
                    )

            yield (
                "data: "
                + json.dumps({
                    "sources": sources_payload,
                    "session_id": conversation_id,
                    "memories_used": memories_used,
                    "done": True,
                })
                + "\n\n"
            )

        except Exception:
            logger.exception("Stream failed")
            # FIX: don't leak exception text to the client.
            yield f"data: {json.dumps({'error': 'Stream failed'})}\n\n"

        finally:
            # FIX: if the client disconnected mid-stream, the generator is
            # closed and the code above may not have persisted the partial
            # answer. Persist what we have so history isn't silently lost.
            if not persisted and full_answer:
                try:
                    chat_store.add_message(
                        conversation_id,
                        "assistant",
                        full_answer,
                        sources=sources_payload or None,
                    )
                    logger.info(
                        "Persisted partial assistant turn for session %s "
                        "(likely client disconnect)",
                        conversation_id,
                    )
                except Exception:
                    logger.exception("Failed to persist partial assistant turn")

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# =====================================================
# 8. SESSIONS (backed by SQLite)
# =====================================================
@app.post("/sessions", response_model=SessionResponse)
def create_session(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    """Create an empty conversation. Kept for API compatibility."""
    user_id = get_user_id(x_user_id)
    sid = chat_store.create_conversation(title="New Conversation", user_id=user_id)
    return SessionResponse(session_id=sid, messages=[])


@app.get("/sessions", response_model=List[ConversationMeta])
def list_sessions(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    """List all past sessions for this user (survives restarts)."""
    user_id = get_user_id(x_user_id)
    return [ConversationMeta(**c) for c in chat_store.list_conversations(user_id=user_id)]


@app.get("/sessions/{sid}", response_model=SessionResponse)
def get_session(
    sid: str,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    user_id = get_user_id(x_user_id)
    convo = chat_store.get_conversation(sid)
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = chat_store.get_messages(sid)
    return SessionResponse(session_id=sid, messages=messages)


@app.delete("/sessions/{sid}")
def delete_session(
    sid: str,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    user_id = get_user_id(x_user_id)
    convo = chat_store.get_conversation(sid)
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Clean up both SQLite and Chroma memories for this session
    memory_store.delete_for_session(sid, user_id=user_id)
    chat_store.delete_conversation(sid)
    return {"deleted": sid}


# =====================================================
# 9. END SESSION (force-summarize into long-term memory)
# =====================================================
@app.post("/sessions/{sid}/end")
def end_session(
    sid: str,
    background: BackgroundTasks,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    user_id = get_user_id(x_user_id)
    convo = chat_store.get_conversation(sid)
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found.")

    agent = get_agent()
    llm = getattr(agent, "llm", None)
    if llm is None:
        raise HTTPException(status_code=500, detail="Agent has no LLM for summarization")

    background.add_task(
        memory_manager.commit_session_to_memory, sid, user_id, llm
    )
    return {"status": "summarizing", "session_id": sid}


# =====================================================
# 10. MEMORY INTROSPECTION
# =====================================================
@app.get("/memories")
def list_memories(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    user_id = get_user_id(x_user_id)
    return {"user_id": user_id, "count": memory_store.count(user_id=user_id)}