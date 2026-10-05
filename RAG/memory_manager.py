"""
memory_manager.py

Bridges chat_store (raw history) and memory_store (semantic recall).
Called from FastAPI at request time (retrieval) and in the
background (summarization).
"""

import logging

import chat_store
import memory_store

logger = logging.getLogger(__name__)


# -------- Retrieval side (called at chat request time) --------

def build_memory_context(
    user_query: str,
    user_id: str = "default",
    n_results: int = 5,
    min_similarity: float = 0.25,
) -> str:
    """Return a formatted block to inject into the system prompt.
    Empty string if no memories are relevant."""

    try:
        memories = memory_store.recall(
            query=user_query,
            user_id=user_id,
            n_results=n_results,
            min_similarity=min_similarity,
        )
    except Exception:
        logger.exception("Memory recall failed")
        return ""

    if not memories:
        return ""

    bullets = "\n".join(f"- {m['text']}" for m in memories)
    return (
        "Relevant memories from past conversations with this user:\n"
        f"{bullets}\n"
        "(Use these naturally. Don't claim to remember things not listed.)"
    )


# -------- Write side (called in background) --------

def summarize_session(conversation_id: str, llm) -> str:
    """Ask the LLM to distill a session into concise memory bullets."""

    messages = chat_store.get_recent_messages(conversation_id, limit=50)
    if not messages:
        return ""

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = (
    "You are extracting long-term memory about a user from a "
    "conversation between them and an AI assistant.\n\n"
    "Extract EVERY concrete fact the user reveals about themselves:\n"
    "  • their name, profession, role, or identity\n"
    "  • personal preferences (formatting, tone, style, length)\n"
    "  • constraints (dietary, medical, accessibility, language)\n"
    "  • ongoing projects, goals, or tasks\n"
    "  • tools, technologies, or stacks they use\n"
    "  • important relationships or context they mention\n\n"
    "Write each fact as its OWN standalone bullet point that makes "
    "sense without the conversation.\n"
    "Aim for 3-7 bullets. Do NOT summarize the conversation itself — "
    "extract facts about the user.\n\n"
    "If the user revealed NOTHING concrete about themselves, "
    "reply exactly: NONE\n\n"
    "Conversation:\n"
    f"{transcript}"
    )

    try:
        response = llm.invoke(prompt)
        summary = getattr(response, "content", str(response)).strip()
    except Exception:
        logger.exception("Summarization LLM call failed")
        return ""

    if summary.upper() == "NONE" or len(summary) < 10:
        return ""

    return summary


def commit_session_to_memory(
    conversation_id: str,
    user_id: str,
    llm,
) -> bool:
    """Summarize a session and store it in Chroma. Idempotent —
    won't re-summarize the same conversation."""

    if chat_store.is_summarized(conversation_id):
        return False

    summary = summarize_session(conversation_id, llm)
    if not summary:
        # Mark as summarized anyway so we don't retry endlessly.
        chat_store.save_session_summary(
            conversation_id=conversation_id,
            summary="",
            user_id=user_id,
        )
        return False

    memory_store.save_memory(
        text=summary,
        session_id=conversation_id,
        user_id=user_id,
        memory_type="session_summary",
    )
    chat_store.save_session_summary(
        conversation_id=conversation_id,
        summary=summary,
        user_id=user_id,
    )
    logger.info("Committed memory for session %s", conversation_id)
    return True


def drain_summarization_queue(user_id: str, llm, max_sessions: int = 5) -> int:
    """Summarize any pending sessions. Call on startup and on shutdown."""

    pending = chat_store.get_unsummarized_conversations(
        user_id=user_id, min_messages=4, limit=max_sessions
    )

    count = 0
    for convo in pending:
        try:
            if commit_session_to_memory(convo["id"], user_id, llm):
                count += 1
        except Exception:
            logger.exception("Failed to summarize %s", convo["id"])

    return count