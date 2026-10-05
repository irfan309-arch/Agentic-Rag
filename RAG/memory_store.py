"""
memory_store.py

Long-term semantic memory backed by ChromaDB.
Persistent across restarts; scoped per user_id.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import chromadb

CHROMA_PATH = os.getenv(
    "CHROMA_MEMORY_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_memory"),    
)

# PersistentClient survives process restarts.
# This is the whole point — in-memory Client() would lose everything.
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_memory = _client.get_or_create_collection(
    name="long_term_memory",
    metadata={"hnsw:space": "cosine"},
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_memory(
    text: str,
    session_id: str,
    user_id: str = "default",
    memory_type: str = "session_summary",
) -> str:
    """Store a memory. Returns the generated memory id."""

    if not text or not text.strip():
        return ""

    mem_id = f"{user_id}::{session_id}::{datetime.now().timestamp()}"
    _memory.add(
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "session_id": session_id,
            "type": memory_type,
            "timestamp": _now(),
        }],
        ids=[mem_id],
    )
    return mem_id


def recall(
    query: str,
    user_id: str = "default",
    n_results: int = 5,
    min_similarity: Optional[float] = None,
) -> list[dict]:
    """Return relevant memories with their metadata.

    `min_similarity` (0..1) — optional cutoff. Chroma cosine distance
    is 1 - similarity, so we filter on distance <= (1 - min_similarity).
    """

    try:
        if _memory.count() == 0:
            return []
    except Exception:
        return []

    results = _memory.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist
        if min_similarity is not None and similarity < min_similarity:
            continue
        out.append({
            "text": doc,
            "metadata": meta,
            "similarity": similarity,
        })
    return out


def count(user_id: Optional[str] = None) -> int:
    if user_id is None:
        return _memory.count()
    result = _memory.get(where={"user_id": user_id})
    return len(result["ids"])


def delete_for_session(session_id: str, user_id: str = "default") -> None:
    """Remove all memories tied to a conversation (e.g. on delete)."""
    _memory.delete(
        where={"$and": [
            {"user_id": user_id},
            {"session_id": session_id},
        ]}
    )