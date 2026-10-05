"""
chat_store.py

Lightweight persistence layer for chat conversations, backed by
SQLite. Each conversation has a title, an owning user, and a list
of messages (role, content, and optionally a list of source-document
dicts).

This is deliberately independent of Streamlit and of RAGService/
Agent so it can be tested or swapped out on its own. A new
connection is opened per call rather than held open, which keeps
it safe across Streamlit's per-session threads.

Cross-session memory support:
  - Conversations are scoped by `user_id` so multiple users can
    share one DB without leaking context.
  - `get_recent_conversations()` and `get_recent_messages()` give
    the memory layer (memory_manager.py) what it needs to build
    session summaries for long-term vector storage.
  - A `session_summaries` table records which conversations have
    already been distilled into memory, so we never summarize the
    same session twice.

Railway deployment:
  - DB_PATH is env-driven so it can point at the mounted volume
    (e.g. /app/data/chat.db). Defaults to the module directory for
    local dev.
  - WAL journaling + a busy timeout are enabled so concurrent writes
    from /chat and background summarization tasks don't throw
    "database is locked" on Railway's network-attached volume.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

# FIX: env-driven DB path. On Railway, mount a volume at /app/data and
# set CHAT_DB_PATH=/app/data/chat.db. Locally this falls back to the
# module directory, preserving existing behavior.
DB_PATH = os.getenv(
    "CHAT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "chat_history.db"),  
)

DEFAULT_USER_ID = "default"

# Sidebar titles are kept short — a few-word summary of the first
# question, not the question in full.
TITLE_MAX_WORDS = 6
TITLE_MAX_LENGTH = 40


def _shorten_title(text: str) -> str:
    """Reduce a question/title down to a short few-word sidebar label."""

    clean_text = (text or "New Conversation").strip()

    # Only look at the first line — multi-line pastes shouldn't
    # spill into the sidebar.
    clean_text = clean_text.splitlines()[0].strip()

    words = clean_text.split()
    truncated_by_words = len(words) > TITLE_MAX_WORDS
    short_text = " ".join(words[:TITLE_MAX_WORDS])

    if len(short_text) > TITLE_MAX_LENGTH:
        short_text = short_text[:TITLE_MAX_LENGTH].rstrip()
        truncated_by_words = True

    if truncated_by_words:
        short_text = short_text.rstrip(".,;:!?") + "…"

    return short_text or "New Conversation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==========================================
# CONNECTION / SCHEMA
# ==========================================

def _get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    # FIX: SQLite over a network-attached volume (Railway volume) needs
    # WAL + a busy timeout to survive concurrent writers. Without these,
    # a /chat write and a background summarization task can collide and
    # raise "database is locked".
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")

    return connection


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every run.

    Also performs a lightweight migration: if `conversations` exists
    from an older version without a `user_id` column, we add it with
    a default value so existing data keeps working.
    """

    # FIX: ensure the DB parent directory exists. On Railway the volume
    # mount point (/app/data) is created by the platform, but if you
    # change CHAT_DB_PATH to a subdirectory this prevents a confusing
    # "unable to open database file" error on first boot.
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with _get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            )
            """
        )

        # Tracks which conversations have already been distilled into
        # long-term vector memory. Prevents duplicate summarization
        # when the user reopens a session.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                summary TEXT NOT NULL,
                summarized_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            )
            """
        )

        # --- Migration: add user_id to older `conversations` tables ---
        existing_cols = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "user_id" not in existing_cols:
            connection.execute(
                "ALTER TABLE conversations "
                "ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'"
            )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user "
            "ON conversations (user_id, updated_at DESC)"
        )

        connection.commit()


# ==========================================
# CONVERSATIONS
# ==========================================

def create_conversation(
    title: str,
    user_id: str = DEFAULT_USER_ID,
) -> str:
    """Create a new conversation and return its id."""

    conversation_id = uuid.uuid4().hex
    timestamp = _now()
    clean_title = _shorten_title(title)

    with _get_connection() as connection:

        connection.execute(
            """
            INSERT INTO conversations
                (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, user_id, clean_title, timestamp, timestamp),
        )

        connection.commit()

    return conversation_id


def rename_conversation(conversation_id: str, title: str) -> None:

    clean_title = _shorten_title(title)

    with _get_connection() as connection:

        connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (clean_title, _now(), conversation_id),
        )

        connection.commit()


def touch_conversation(conversation_id: str) -> None:
    """Bump updated_at so the conversation sorts to the top of the sidebar."""

    with _get_connection() as connection:

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )

        connection.commit()


def list_conversations(user_id: str = DEFAULT_USER_ID) -> list[dict]:
    """Most-recently-updated conversations for a user, first."""

    with _get_connection() as connection:

        rows = connection.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> Optional[dict]:
    """Fetch one conversation's metadata, or None if it doesn't exist."""

    with _get_connection() as connection:

        row = connection.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

    return dict(row) if row else None


def get_recent_conversations(
    limit: int = 20,
    user_id: str = DEFAULT_USER_ID,
) -> list[dict]:
    """Recent conversations — used by the memory layer to decide what
    to summarize and store as long-term memory."""

    with _get_connection() as connection:

        rows = connection.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> None:

    with _get_connection() as connection:

        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )

        connection.execute(
            "DELETE FROM session_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )

        connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )

        connection.commit()


# ==========================================
# MESSAGES
# ==========================================

def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> None:
    """
    Append a message. `sources`, if given, should already be plain
    JSON-serializable dicts (see documents_to_source_dicts() in
    app.py) rather than langchain Document objects.
    """

    with _get_connection() as connection:

        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                json.dumps(sources) if sources else None,
                _now(),
            ),
        )

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )

        connection.commit()


def get_messages(conversation_id: str) -> list[dict]:
    """Return messages in chronological order, sources decoded back to dicts."""

    with _get_connection() as connection:

        rows = connection.execute(
            """
            SELECT role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:

        message = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }

        if row["sources"]:
            message["sources"] = json.loads(row["sources"])

        messages.append(message)

    return messages


def get_recent_messages(
    conversation_id: str,
    limit: int = 20,
) -> list[dict]:
    """Last `limit` messages for a conversation (chronological order).
    Used by the summarizer so we don't feed entire mega-sessions into
    the LLM when building memory."""

    with _get_connection() as connection:

        rows = connection.execute(
            """
            SELECT role, content, sources, created_at
            FROM (
                SELECT role, content, sources, created_at, id
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (conversation_id, limit),
        ).fetchall()

    messages = []
    for row in rows:
        message = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        if row["sources"]:
            message["sources"] = json.loads(row["sources"])
        messages.append(message)

    return messages


# ==========================================
# SESSION SUMMARIES (bridge to long-term memory)
# ==========================================

def save_session_summary(
    conversation_id: str,
    summary: str,
    user_id: str = DEFAULT_USER_ID,
) -> None:
    """Record that a session has been summarized. The vector store
    write happens in memory_store.py — this table just prevents us
    from re-summarizing the same session on every app launch."""

    with _get_connection() as connection:

        connection.execute(
            """
            INSERT INTO session_summaries
                (conversation_id, user_id, summary, summarized_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_at = excluded.summarized_at
            """,
            (conversation_id, user_id, summary, _now()),
        )

        connection.commit()


def get_session_summary(conversation_id: str) -> Optional[str]:
    """Return the stored summary for a conversation, or None."""

    with _get_connection() as connection:

        row = connection.execute(
            "SELECT summary FROM session_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    return row["summary"] if row else None


def is_summarized(conversation_id: str) -> bool:
    """True if this conversation has already been distilled into memory."""

    with _get_connection() as connection:

        row = connection.execute(
            "SELECT 1 FROM session_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    return row is not None


def get_unsummarized_conversations(
    user_id: str = DEFAULT_USER_ID,
    min_messages: int = 4,
    limit: int = 10,
) -> list[dict]:
    """Conversations that have enough substance to summarize but
    haven't been yet. This is the queue the memory manager drains
    at session end / app startup."""

    with _get_connection() as connection:

        rows = connection.execute(
            """
            SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
              AND c.id NOT IN (SELECT conversation_id FROM session_summaries)
            GROUP BY c.id
            HAVING message_count >= ?
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (user_id, min_messages, limit),
        ).fetchall()

    return [dict(row) for row in rows]