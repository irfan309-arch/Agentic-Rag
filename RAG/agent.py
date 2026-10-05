# agent.py
from typing import Optional

from rag_engine import RAGService


# Fixed message returned when the documents genuinely don't contain
# the answer. Using a hard-coded string here (instead of trusting the
# LLM to say this on its own) guarantees the system never falls back
# to the model's outside/general knowledge for a real question.
NOT_FOUND_MESSAGE = "I couldn't find that information in the uploaded documents."

# Fixed message when a personal question is asked but no memories exist.
NO_MEMORY_MESSAGE = (
    "I don't have any notes from our past conversations yet. "
    "Feel free to tell me about yourself and I'll remember it."
)


class Agent:

    def __init__(self, rag_service: RAGService):
        self.rag = rag_service
        self.llm = rag_service.llm

    # =====================================================
    # 1b. MEMORY-AWARE SYSTEM INSTRUCTIONS
    # =====================================================

    def _memory_preamble(self, memory_block: str) -> str:
        """
        Build the preamble injected ahead of every LLM prompt when
        cross-session memories exist.

        This is used by the document-only generation path (llm_generate)
        for personalization (tone, format, constraints). The stricter
        memory-only and doc+memory paths use their own prompts.
        """

        if not memory_block or not memory_block.strip():
            return ""

        return (
            "The following are notes remembered from past conversations "
            "with this user. Use them to personalise tone, honour user "
            "preferences and constraints, and avoid asking the user to "
            "repeat themselves.\n\n"
            f"{memory_block.strip()}\n\n"
            "---\n\n"
        )

    # =====================================================
    # 1. DECIDE WHETHER RETRIEVAL IS REQUIRED
    # =====================================================

    def llm_decide(self, question: str) -> str:

        prompt = f"""
You are the decision-making agent of a RAG system
that must ONLY answer questions using the uploaded
documents. It never answers factual questions from
its own general knowledge.

Decide whether the user's message is a greeting or
casual pleasantry, or an actual question/request for
information.

Return ONLY one of:

RETRIEVE
DIRECT

Rules:

- DIRECT ONLY for greetings, small talk, or pleasantries
  with no informational content (e.g. "Hello", "Hi",
  "How are you", "Thanks", "Bye").
- RETRIEVE for EVERY other message, including general
  knowledge questions that don't obviously mention the
  documents. Do not try to guess whether the documents
  contain the answer here — that is checked in a later
  step. If in doubt, choose RETRIEVE.

Examples:

Question: What does the Government AI Cloud document
say about multi-LLM architecture?
Answer: RETRIEVE

Question: What is the capital of France?
Answer: RETRIEVE

Question: Hello
Answer: DIRECT

Question: How are you?
Answer: DIRECT

Question: Thanks!
Answer: DIRECT

User question:
{question}
"""

        response = self.llm.invoke(prompt)

        decision = response.content.strip().upper()

        if "DIRECT" in decision:
            return "DIRECT"

        return "RETRIEVE"

    # =====================================================
    # 1c. ROUTE QUESTION (memory-aware)
    # =====================================================

    def llm_route(self, question: str, has_memories: bool) -> str:
        """
        Classify the question for the memory-aware agent:

          DOCUMENT  — needs info from uploaded documents
          MEMORY    — about the user personally (name, prefs, history)
          BOTH      — needs documents + personal context
          DIRECT    — greeting / small talk

        Only called from agent_with_memory(). The classic agent()
        path uses llm_decide() instead.
        """

        memory_hint = (
            "The user HAS past-conversation memories available."
            if has_memories
            else "The user has NO past-conversation memories."
        )

        prompt = f"""
You are routing a user message in a RAG system that has BOTH:
  - Access to uploaded documents
  - Access to notes from past conversations with this user

{memory_hint}

Classify the message into ONE of:

DOCUMENT  — the user is asking about the content of uploaded
            documents (facts, summaries, quotes, explanations).
MEMORY    — the user is asking about themselves, their preferences,
            their name, their history, or things they told you
            before.
BOTH      — the user wants document content, but personalised
            using known preferences (tone, format, constraints).
DIRECT    — greeting, thanks, small talk, no informational content.

Rules:
- If the question mentions "I", "my", "me", "we", "our" and asks
  about personal facts → MEMORY (or BOTH if it also asks about
  documents).
- If the question could be answered from uploaded documents → DOCUMENT.
- If both personal context AND document content are needed → BOTH.
- If it's just a greeting → DIRECT.
- If unsure and memories exist → prefer BOTH.

Examples:

"What does the paper say about neural networks?"
→ DOCUMENT

"What is my name?"
→ MEMORY

"What dietary restrictions should this recipe respect?"
→ BOTH

"Give me a bullet-point summary of chapter 3"
→ BOTH

"Hello!"
→ DIRECT

Message:
{question}

Return ONLY the single word: DOCUMENT, MEMORY, BOTH, or DIRECT.
"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().upper()

        # Order matters: check BOTH before MEMORY/DOCUMENT so "BOTH"
        # doesn't get mis-matched as containing "DOCUMENT".
        for label in ("BOTH", "MEMORY", "DOCUMENT", "DIRECT"):
            if label in result:
                return label

        # Default: assume document question if LLM gives unclear output.
        return "DOCUMENT"

    # =====================================================
    # 2. RETRIEVE DOCUMENTS (optionally scoped to one file)
    # =====================================================

    def retrieve(
        self,
        question: str,
        selected_document: Optional[str] = None,
    ):

        if self.rag.vector_store is None:
            raise RuntimeError(
                "Please process documents before asking a question."
            )

        return self.rag.retrieve(
            question,
            source=selected_document,
        )

    # =====================================================
    # 3. CHECK DOCUMENT RELEVANCE
    # =====================================================

    def llm_check_relevance(
        self,
        question: str,
        documents
    ) -> str:

        if not documents:
            return "NOT_RELEVANT"

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are evaluating documents retrieved from
a document knowledge base.

Question:
{question}

Retrieved documents:
{context}

Return ONLY:

RELEVANT
or
NOT_RELEVANT

Return RELEVANT if the documents contain useful
information for answering the question.

Return NOT_RELEVANT if they do not.
"""

        response = self.llm.invoke(prompt)

        result = response.content.strip().upper()

        if "NOT_RELEVANT" in result:
            return "NOT_RELEVANT"

        return "RELEVANT"

    # =====================================================
    # 4. REWRITE QUESTION
    # =====================================================

    def llm_rewrite(self, question: str) -> str:

        prompt = f"""
Rewrite the following question into a better
search query for a document knowledge base.

Keep the original meaning.

Make the query more specific and descriptive.

Return ONLY the rewritten query.

Original question:
{question}
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    # =====================================================
    # 5a. GENERATE — document-only (with optional memory personalization)
    # =====================================================

    def llm_generate(
        self,
        question: str,
        documents=None,
        memory_block: str = "",
    ) -> str:

        preamble = self._memory_preamble(memory_block)

        if not documents:

            # This branch only runs for the DIRECT (greeting/small-talk)
            # path. It must never be used to answer a real question
            # from outside knowledge.
            prompt = f"""
{preamble}You are the assistant for a document question-answering
system. This system can ONLY answer questions using the
documents the user has uploaded — it must never use its
own outside/general knowledge to answer a factual question.

The message below has been classified as a greeting or
casual pleasantry, not a request for information.

Reply warmly and briefly. If the message actually contains
a real question or request for information, do NOT answer
it — instead say you can only help with questions about
their uploaded documents, and invite them to ask one.

If a memory above mentions how the user prefers to be
addressed or their tone preferences, honour that.

Message:
{question}
"""

            response = self.llm.invoke(prompt)

            return response.content

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
{preamble}You are a document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Do not use outside knowledge for factual answers.
2. Do not invent information.
3. If the answer is not available in the documents,
say:

"I couldn't find that information in the uploaded documents."

4. Keep the answer clear and concise.
5. If memories above reveal user preferences or constraints
   (e.g. dietary, formatting, language), apply them naturally
   to how you present the answer.

Question:
{question}

Document context:
{context}
"""

        response = self.llm.invoke(prompt)

        return response.content

    # =====================================================
    # 5b. GENERATE — from memory only
    # =====================================================

    def llm_generate_from_memory(
        self,
        question: str,
        memory_block: str,
    ) -> str:
        """
        Answer a question using ONLY cross-session memories.
        Used when the question is about the user personally and
        the uploaded documents aren't the right source.
        """

        prompt = f"""
You are a helpful assistant with long-term memory of past
conversations with this user.

Answer the user's question using ONLY the remembered notes
below. These notes are things the user told you in previous
sessions.

Rules:
1. If the notes contain the answer, answer directly and naturally.
2. If the notes don't contain the answer, reply exactly:
   "I don't have that in my memory from our past conversations."
3. Do NOT invent or guess.
4. Do NOT use the uploaded documents for this answer — this is
   a personal-memory question.

Remembered notes:
{memory_block.strip()}

Question:
{question}
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    # =====================================================
    # 5c. GENERATE — documents + memory personalization
    # =====================================================

    def llm_generate_with_both(
        self,
        question: str,
        documents,
        memory_block: str,
    ) -> str:
        """
        Answer using document content, personalized using memory.
        Used when the question needs both document facts AND
        user-specific context (format, tone, constraints).
        """

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        memory_section = (
            memory_block.strip() if memory_block and memory_block.strip()
            else "(no notes about the user yet)"
        )

        prompt = f"""
You are a document question-answering assistant with memory
of past conversations with this user.

Answer the user's question using:
  - The document context for FACTS
  - The remembered notes for PERSONALIZATION (tone, format,
    constraints, preferences)

Rules:
1. Facts must come from the document context — do not invent.
2. Apply the user's remembered preferences naturally
   (e.g., if they prefer bullet points, use bullet points).
3. Honour any remembered constraints (dietary, medical, etc.).
4. If the documents don't contain the answer, say:
   "I couldn't find that information in the uploaded documents."
5. Keep the answer clear and concise.

Remembered notes about the user:
{memory_section}

Document context:
{context}

Question:
{question}
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    # =====================================================
    # 6. CLASSIC AGENT (no memory)
    # =====================================================

    def agent(
        self,
        question: str,
        selected_document: Optional[str] = None,
    ):

        print("\n" + "=" * 60)
        print("AGENT STARTED")
        print("=" * 60)

        print(f"Question: {question}")
        print(f"Selected document: {selected_document or 'ALL'}")

        # -------------------------------------------------
        # STEP 1: UNDERSTAND QUESTION
        # -------------------------------------------------

        print("\n[STEP 1] Understanding question...")

        decision = self.llm_decide(question)

        print(f"[STEP 1] Decision: {decision}")

        # -------------------------------------------------
        # STEP 2: DIRECT ANSWER
        # -------------------------------------------------

        if decision == "DIRECT":

            print("\n[STEP 2] Retrieval not required.")

            answer = self.llm_generate(question)

            print("\n[STEP 3] Answer generated.")

            return answer, []

        # -------------------------------------------------
        # STEP 2: RETRIEVE (scoped to selected_document, if any)
        # -------------------------------------------------

        print("\n[STEP 2] Retrieving documents...")

        documents = self.retrieve(question, selected_document)

        print(
            f"[STEP 2] Retrieved {len(documents)} documents."
        )

        # -------------------------------------------------
        # STEP 3: CHECK RELEVANCE
        # -------------------------------------------------

        print("\n[STEP 3] Checking document relevance...")

        relevance = self.llm_check_relevance(
            question,
            documents
        )

        print(f"[STEP 3] Relevance: {relevance}")

        # -------------------------------------------------
        # STEP 4: REWRITE AND RETRIEVE AGAIN (same scope)
        # -------------------------------------------------

        if relevance == "NOT_RELEVANT":

            print("\n[STEP 4] Documents are not relevant.")

            print("[STEP 4] Rewriting question...")

            new_question = self.llm_rewrite(question)

            print(
                f"[STEP 4] New query: {new_question}"
            )

            print("[STEP 4] Retrieving again...")

            documents = self.retrieve(new_question, selected_document)

            print(
                f"[STEP 4] Retrieved {len(documents)} documents."
            )

            relevance = self.llm_check_relevance(
                new_question,
                documents
            )

            print(f"[STEP 4] Relevance after retry: {relevance}")

            if relevance == "NOT_RELEVANT":

                print("\n[STEP 5] Still not relevant — returning not-found message.")
                print("=" * 60)

                return NOT_FOUND_MESSAGE, []

        # -------------------------------------------------
        # STEP 5: GENERATE FINAL ANSWER
        # -------------------------------------------------

        print("\n[STEP 5] Generating final answer...")

        answer = self.llm_generate(
            question,
            documents
        )

        print("\n[STEP 6] Agent finished.")
        print("=" * 60)

        return answer, documents

    # =====================================================
    # 7. MEMORY-AWARE AGENT (non-streaming)
    # =====================================================

    def agent_with_memory(
        self,
        question: str,
        selected_document: Optional[str] = None,
        memory_block: str = "",
    ):
        """
        Full agent with memory. Routes the question to:

          DIRECT   — greeting
          MEMORY   — answers from past-session memories only
          DOCUMENT — answers from uploaded documents only
          BOTH     — combines documents + memory personalization

        Falls back to memory when the document path fails but
        memories exist.
        """

        print("\n" + "=" * 60)
        print("AGENT STARTED (with memory)")
        print("=" * 60)

        has_memories = bool(memory_block and memory_block.strip())

        print(f"Question: {question}")
        print(f"Selected document: {selected_document or 'ALL'}")
        print(f"Has memories: {has_memories}")

        # -------------------------------------------------
        # STEP 1: ROUTE
        # -------------------------------------------------

        print("\n[STEP 1] Routing question...")

        route = self.llm_route(question, has_memories)

        print(f"[STEP 1] Route: {route}")

        # -------------------------------------------------
        # DIRECT — greeting
        # -------------------------------------------------

        if route == "DIRECT":

            print("\n[STEP 2] Direct greeting response.")

            answer = self.llm_generate(question, memory_block=memory_block)

            return answer, []

        # -------------------------------------------------
        # MEMORY — personal question
        # -------------------------------------------------

        if route == "MEMORY":

            if not has_memories:

                print("\n[STEP 2] MEMORY route but no memories available.")

                return NO_MEMORY_MESSAGE, []

            print("\n[STEP 2] Answering from memory only.")

            answer = self.llm_generate_from_memory(question, memory_block)

            return answer, []

        # -------------------------------------------------
        # DOCUMENT or BOTH — retrieve documents
        # -------------------------------------------------

        print(f"\n[STEP 2] Retrieving documents (route={route})...")

        documents = self.retrieve(question, selected_document)

        print(f"[STEP 2] Retrieved {len(documents)} documents.")

        print("\n[STEP 3] Checking document relevance...")

        relevance = self.llm_check_relevance(question, documents)

        print(f"[STEP 3] Relevance: {relevance}")

        # -------------------------------------------------
        # STEP 4: retry with rewritten query
        # -------------------------------------------------

        if relevance == "NOT_RELEVANT":

            print("\n[STEP 4] Documents not relevant. Rewriting query...")

            new_question = self.llm_rewrite(question)

            print(f"[STEP 4] New query: {new_question}")

            documents = self.retrieve(new_question, selected_document)

            print(f"[STEP 4] Retrieved {len(documents)} documents.")

            relevance = self.llm_check_relevance(new_question, documents)

            print(f"[STEP 4] Relevance after retry: {relevance}")

            if relevance == "NOT_RELEVANT":

                # Documents can't answer — try memory fallback
                if has_memories:

                    print("\n[STEP 5] Docs irrelevant — trying memory fallback.")

                    mem_answer = self.llm_generate_from_memory(
                        question, memory_block
                    )

                    if (
                        mem_answer
                        and "I don't have that in my memory" not in mem_answer
                        and NOT_FOUND_MESSAGE not in mem_answer
                    ):
                        print("[STEP 5] Answered from memory.")
                        return mem_answer, []

                print("\n[STEP 5] Returning not-found message.")
                print("=" * 60)
                return NOT_FOUND_MESSAGE, []

        # -------------------------------------------------
        # STEP 5: GENERATE FINAL ANSWER
        # -------------------------------------------------

        print(f"\n[STEP 5] Generating answer (route={route}).")

        if has_memories and route in ("BOTH", "DOCUMENT"):
            answer = self.llm_generate_with_both(
                question, documents, memory_block
            )
        else:
            answer = self.llm_generate(question, documents)

        print("\n[STEP 6] Agent finished.")
        print("=" * 60)

        return answer, documents

    # =====================================================
    # 8. STREAMING AGENT (memory-aware shim)
    # =====================================================

    def agent_stream_with_memory(
        self,
        question: str,
        selected_document: Optional[str] = None,
        memory_block: str = "",
    ):
        """
        Yield the final answer as chunks.

        This is a shim over agent_with_memory() — the full answer
        is produced, then yielded in small pieces. This is NOT
        true token-by-token streaming; the client receives the
        response chunked, which keeps the routing + retrieval
        pipeline fully intact.
        """

        answer, _docs = self.agent_with_memory(
            question,
            selected_document,
            memory_block,
        )

        chunk_size = 40
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]

    def agent_stream(
        self,
        question: str,
        selected_document: Optional[str] = None,
    ):
        """Non-memory streaming shim — delegates to the memory-aware
        version with an empty memory block."""
        yield from self.agent_stream_with_memory(
            question,
            selected_document,
            memory_block="",
        )