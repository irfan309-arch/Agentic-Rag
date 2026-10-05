# api_client.py
"""
Thin HTTP client for the FastAPI RAG backend.
Mirrors all endpoints in main.py.

Every request carries an X-User-Id header (defaults to "default").
app.py sets `.user_id` on the client right after construction so
conversations and long-term memories are scoped per browser session.

Railway deployment:
  - The backend URL is read from BACKEND_URL (preferred) or API_BASE
    (legacy). Locally this falls back to http://127.0.0.1:8000.
  - On Railway, set BACKEND_URL=http://backend.railway.internal:8000
    on the frontend service so traffic stays on the private mesh.
"""
import os
from typing import Optional, Iterator, List
import json

import requests


# Prefer BACKEND_URL (Railway convention); fall back to API_BASE for
# any existing local .env files; final fallback is localhost.
API_BASE = (
    os.getenv("BACKEND_URL")
    or os.getenv("API_BASE")
    or "http://127.0.0.1:8000"
)

TIMEOUT = 300


class APIError(Exception):
    """Raised when the API returns a non-2xx response."""
    pass


class _Doc:
    """Minimal LangChain-Document-like object so
    documents_to_source_dicts() works unchanged."""
    def __init__(self, source: str, page, content: str = ""):
        self.metadata = {"source": source, "page": page}
        self.page_content = content


class APIClient:
    def __init__(self, base_url: str = API_BASE, user_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id

    # ---------- helpers ----------
    def _headers(self) -> dict:
        """Every request carries the current user id."""
        return {"X-User-Id": self.user_id}

    def _raise(self, r: requests.Response):
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise APIError(detail)

    # ---------- 1. health ----------
    def health(self) -> dict:
        r = requests.get(
            f"{self.base_url}/health",
            headers=self._headers(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    # ---------- 2. ready ----------
    def ready(self) -> dict:
        r = requests.get(
            f"{self.base_url}/ready",
            headers=self._headers(),
            timeout=10,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 3. upload ----------
    def upload(self, files: list, replace: bool = False) -> dict:
        """
        files: list of Streamlit UploadedFile (has .name and .getvalue()).
        replace=False → additive (default). replace=True → wipe + rebuild.
        """
        files_payload = [
            ("files", (f.name, f.getvalue(), "application/octet-stream"))
            for f in files
        ]
        r = requests.post(
            f"{self.base_url}/upload",
            params={"replace": str(replace).lower()},
            files=files_payload,
            headers=self._headers(),
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 4. documents ----------
    def documents(self) -> list:
        r = requests.get(
            f"{self.base_url}/documents",
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("document_names", [])

    def get_document_names(self) -> list:
        """Facade so existing Streamlit code keeps working."""
        return self.documents()

    # ---------- 5. delete document ----------
    def delete_document(self, name: str) -> dict:
        r = requests.delete(
            f"{self.base_url}/documents/{name}",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 6. chat ----------
    def chat(
        self,
        question: str,
        selected_document: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Returns (answer, documents, session_id, memories_used).

        documents is a list of Document-like objects carrying
        real source + page + snippet. session_id is the (possibly
        newly created) conversation id the server assigned.
        """
        r = requests.post(
            f"{self.base_url}/chat",
            json={
                "question": question,
                "selected_document": selected_document,
                "session_id": session_id,
            },
            headers=self._headers(),
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            self._raise(r)

        data = r.json()
        docs = [
            _Doc(s["source"], s.get("page", "N/A"), s.get("snippet", ""))
            for s in data.get("sources", [])
        ]
        return (
            data["answer"],
            docs,
            data.get("session_id"),
            data.get("memories_used", 0),
        )

    # ---------- 7. chat stream ----------
    def chat_stream(
        self,
        question: str,
        selected_document: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """
        Yields dicts like {"token": "..."} and finally
        {"sources": [...], "session_id": "...", "memories_used": N, "done": True}.
        """
        with requests.post(
            f"{self.base_url}/chat/stream",
            json={
                "question": question,
                "selected_document": selected_document,
                "session_id": session_id,
            },
            headers=self._headers(),
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            if r.status_code >= 400:
                self._raise(r)
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue

    # ---------- 8. create session ----------
    def create_session(self) -> dict:
        r = requests.post(
            f"{self.base_url}/sessions",
            headers=self._headers(),
            timeout=10,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 9. list sessions ----------
    def list_sessions(self) -> List[dict]:
        """
        Returns a list of conversation metadata dicts:
        {id, user_id, title, created_at, updated_at}.
        """
        r = requests.get(
            f"{self.base_url}/sessions",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 10. get session ----------
    def get_session(self, session_id: str) -> dict:
        r = requests.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 11. delete session ----------
    def delete_session(self, session_id: str) -> dict:
        r = requests.delete(
            f"{self.base_url}/sessions/{session_id}",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 12. end session (force-summarize into long-term memory) ----------
    def end_session(self, session_id: str) -> dict:
        """
        Tell the backend to finalize this session — the conversation
        is distilled into long-term memory immediately instead of
        waiting for the next N-turn trigger.
        """
        r = requests.post(
            f"{self.base_url}/sessions/{session_id}/end",
            headers=self._headers(),
            timeout=60,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    # ---------- 13. memory introspection ----------
    def memories_count(self) -> int:
        """How many long-term memories are stored for this user."""
        r = requests.get(
            f"{self.base_url}/memories",
            headers=self._headers(),
            timeout=10,
        )
        if r.status_code >= 400:
            self._raise(r)
        return r.json().get("count", 0)

    # =========================================================
    # ALIASES FOR app.py
    # =========================================================
    # app.py was updated to call list_conversations() and
    # get_messages() — these thin aliases map onto the existing
    # list_sessions() and get_session() methods so the UI code
    # can stay readable without duplicating HTTP logic.

    def list_conversations(self) -> List[dict]:
        """
        Sidebar conversation list for the current user.
        Alias for list_sessions().
        """
        return self.list_sessions()

    def get_messages(self, session_id: str) -> List[dict]:
        """
        Message history for one conversation.
        Alias for get_session()['messages'].
        """
        session = self.get_session(session_id)
        return session.get("messages", [])