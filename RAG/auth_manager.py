# auth_manager.py
"""
Lightweight auth manager for the Streamlit RAG UI.

Responsibilities:
  - Store users in a local SQLite DB (users.db)
  - Hash passwords with bcrypt
  - Authenticate (login) and register (signup)
  - Persist identity into st.session_state
  - Provide role + tenant_id so the RAG backend can filter

This is intended as a simple, self-hosted auth layer.
For production SSO (Google/Okta/Entra), replace with OIDC via
Streamlit's native st.login() — but keep the same session_state keys
(user_id, user_email, user_role, tenant_id) so app.py stays unchanged.

Railway deployment:
  - DB_PATH is env-driven via USERS_DB_PATH so the DB lives on the
    mounted volume (e.g. /app/data/users.db) instead of the container's
    ephemeral layer. Falls back to <module_dir>/users.db for local dev.
  - WAL + busy_timeout pragmas are enabled so concurrent logins don't
    hit "database is locked" on network-attached storage.
  - Connections are explicitly closed (the `with sqlite3.connect(...)`
    idiom commits the transaction but does NOT close the connection).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import streamlit as st


logger = logging.getLogger(__name__)


# ==================================================
# CONFIGURATION
# ==================================================

# FIX: env-driven DB path. On Railway set
#   USERS_DB_PATH=/app/data/users.db
# Locally it falls back to <module_dir>/users.db, which is stable
# regardless of the shell's current working directory.
DB_PATH = Path(
    os.getenv(                                  
        "USERS_DB_PATH",
        os.path.join(os.path.dirname(__file__), "users.db"),   
    )
)


# ==================================================
# VALIDATION HELPERS
# ==================================================

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def _password_ok(password: str) -> tuple[bool, str]:
    """Minimum viable password policy. Relax/strengthen as needed."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain a letter."
    if not re.search(r"\d", password):
        return False, "Password must contain a digit."
    return True, ""


# ==================================================
# CONNECTION FACTORY
# ==================================================

@contextmanager
def _connect(db_path: Path):
    """
    Yield a SQLite connection that is guaranteed to be closed on exit.

    `with sqlite3.connect(...)` only commits/rolls back the transaction —
    it does NOT close the connection. Wrapping explicitly prevents
    connection leakage across repeated login/signup calls.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")

    # FIX: required on Railway's network-attached volumes. Without these,
    # concurrent logins can hit "database is locked".
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    try:
        yield conn
    finally:
        conn.close()


# ==================================================
# AUTH MANAGER
# ==================================================

class AuthManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    # -------- DB bootstrap --------
    def _init_db(self) -> None:
        # FIX: ensure parent directory exists before SQLite tries to
        # open the file. On Railway the volume mount creates /app/data,
        # but not any subdirectory you might point at later.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id       TEXT PRIMARY KEY,
                    email         TEXT UNIQUE NOT NULL,
                    name          TEXT,
                    password_hash TEXT NOT NULL,
                    role          TEXT NOT NULL DEFAULT 'user',
                    tenant_id     TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
            )
            conn.commit()

        logger.info("Auth DB ready at %s", self.db_path)

    # -------- Signup --------
    def signup(
        self,
        email: str,
        password: str,
        name: str = "",
        role: str = "user",
    ) -> tuple[bool, str]:
        email = email.strip().lower()

        if not _is_valid_email(email):
            return False, "Please enter a valid email address."

        ok, msg = _password_ok(password)
        if not ok:
            return False, msg

        # Tenant derived from email domain (swap for real org logic later)
        tenant_id = email.split("@")[-1]

        pw_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # Stable user_id: short deterministic-ish ID based on email
        # (kept simple; swap for uuid4 if you prefer)
        user_id = "u_" + email.replace("@", "_at_").replace(".", "_")

        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO users
                        (user_id, email, name, password_hash,
                         role, tenant_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        email,
                        name or email.split("@")[0],
                        pw_hash,
                        role,
                        tenant_id,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            return True, "Account created. Please log in."
        except sqlite3.IntegrityError:
            return False, "An account with this email already exists."
        except Exception:
            logger.exception("Signup failed for email=%s", email)
            return False, "Could not create account. Please try again."

    # -------- Login --------
    def login(self, email: str, password: str) -> tuple[bool, str]:
        email = email.strip().lower()

        try:
            with _connect(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT user_id, email, name, password_hash,
                           role, tenant_id
                    FROM users WHERE email = ?
                    """,
                    (email,),
                ).fetchone()
        except Exception:
            logger.exception("Login query failed for email=%s", email)
            return False, "Login service unavailable. Please try again."

        # Constant-time-ish: always run bcrypt against *something*.
        if row is None:
            # Dummy hash to avoid user-enumeration timing leak.
            bcrypt.checkpw(
                password.encode("utf-8"),
                b"$2b$12$........................................",
            )
            return False, "Invalid email or password."

        user_id, db_email, name, pw_hash, role, tenant_id = row

        if not bcrypt.checkpw(
            password.encode("utf-8"), pw_hash.encode("utf-8")
        ):
            return False, "Invalid email or password."

        # ---- SUCCESS: persist identity into session_state ----
        st.session_state["authenticated"] = True
        st.session_state["user_id"]      = user_id
        st.session_state["user_email"]   = db_email
        st.session_state["user_name"]    = name
        st.session_state["user_role"]    = role
        st.session_state["tenant_id"]    = tenant_id

        # Stamp last_login_at — non-fatal if it fails.
        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                    (datetime.now(timezone.utc).isoformat(), user_id),
                )
                conn.commit()
        except Exception:
            logger.exception("Failed to stamp last_login_at for %s", user_id)

        return True, "Logged in."

    # -------- Logout --------
    def logout(self) -> None:
        for key in [
            "authenticated",
            "user_id",
            "user_email",
            "user_name",
            "user_role",
            "tenant_id",
            # also wipe per-user conversation view
            "conversation_id",
            "messages",
        ]:
            st.session_state.pop(key, None)


# ==================================================
# CONVENIENCE ACCESSORS
# ==================================================

def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated", False))


def current_user() -> dict:
    """Return identity bundle — safe to pass to authorization logic."""
    return {
        "user_id":   st.session_state.get("user_id"),
        "email":     st.session_state.get("user_email"),
        "name":      st.session_state.get("user_name"),
        "role":      st.session_state.get("user_role"),
        "tenant_id": st.session_state.get("tenant_id"),
    }


def require_role(*allowed: str) -> bool:
    """Helper for UI gating, e.g. `if require_role('admin'):`."""
    return st.session_state.get("user_role") in allowed