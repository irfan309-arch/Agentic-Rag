# app.py
"""
Agentic RAG Studio — Streamlit UI backed by a FastAPI RAG service.

Talks to the API exclusively through api_client.APIClient.
Conversation history is fetched from the backend over HTTP — the UI
no longer touches the conversation SQLite database directly.

Architecture note:
    The FastAPI backend is the SINGLE WRITER to SQLite for
    conversations. This app reads conversation history through the
    backend's /sessions endpoints, not by opening the DB file.
    This is required for deployment (frontend and backend are
    separate services with separate filesystems).

Authentication:
    Users must log in before the RAG UI is rendered. Identity lives
    in st.session_state and is propagated to the API via X-User-Id.
    Authorization is enforced server-side (tenant + role filters in
    the vector store); the UI only gates visibility.
"""

import uuid
from datetime import datetime, timezone

import streamlit as st

from api_client import APIClient, APIError
from auth_manager import AuthManager, is_authenticated, current_user
from login_page import render_login_page


# ============================================================
# PAGE CONFIGURATION  (must be first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AUTH GATE
# ============================================================

# Create the auth manager once, kept for the life of the session.
if "auth_manager" not in st.session_state:
    st.session_state.auth_manager = AuthManager()

if not is_authenticated():
    render_login_page()
    st.stop()   # ← CRITICAL: prevents the RAG UI from rendering


# Identity bundle (user_id, email, name, role, tenant_id)
USER = current_user()
USER_ID = USER["user_id"]
USER_EMAIL = USER["email"]
USER_NAME = USER["name"]
USER_ROLE = USER["role"]
TENANT_ID = USER["tenant_id"]


# ============================================================
# DATABASE BOOTSTRAP
# ============================================================
# NOTE: The frontend no longer opens the conversation DB.
# The FastAPI backend is the SINGLE WRITER to SQLite; this UI
# reads conversation history exclusively through the API.
# (Users DB is still local to the frontend — see auth_manager.py.)


# ============================================================
# BRAND MARK
# ============================================================

LOGO_SVG = (
    '<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Agentic RAG Studio logo">'
    '<defs><linearGradient id="logoGrad" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">'
    '<stop offset="0" stop-color="#3498db"/><stop offset="1" stop-color="#1abc9c"/>'
    '</linearGradient></defs>'
    '<rect width="40" height="40" rx="11" fill="url(#logoGrad)"/>'
    '<rect x="9" y="10" width="18" height="14" rx="5" fill="#ffffff"/>'
    '<path d="M13 24 L13 28.5 L18.5 24 Z" fill="#ffffff"/>'
    '<line x1="12.5" y1="14.5" x2="22.5" y2="14.5" stroke="#3498db" stroke-width="1.6" stroke-linecap="round"/>'
    '<line x1="12.5" y1="18" x2="19" y2="18" stroke="#3498db" stroke-width="1.6" stroke-linecap="round"/>'
    '<line x1="27.4" y1="13" x2="24.6" y2="15.4" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" opacity="0.9"/>'
    '<circle cx="30.5" cy="11" r="3.4" fill="#ffffff"/>'
    '</svg>'
)

st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .brand-row svg {
        flex-shrink: 0;
        display: block;
    }
    .brand-row-sm svg { width: 30px; height: 30px; }
    .brand-row-lg svg { width: 52px; height: 52px; }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.15;
        background: linear-gradient(90deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .brand-subtitle {
        font-size: 0.7rem;
        opacity: 0.5;
        margin-top: 1px;
    }

    .brand-title-lg {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(90deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .brand-subtitle-lg {
        font-size: 0.95rem;
        opacity: 0.65;
        margin-top: 2px;
    }
    .brand-row-lg {
        margin-bottom: 0.5rem;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 12px !important;
        border: 1px solid rgba(250, 250, 250, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(250, 250, 250, 0.08);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .status-idle {
        background-color: rgba(241, 196, 15, 0.15);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    .status-stale {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .status-offline {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }

    .scope-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background-color: rgba(52, 152, 219, 0.15);
        color: #3498db;
        border: 1px solid rgba(52, 152, 219, 0.3);
        margin-bottom: 0.75rem;
    }

    .conv-timestamp {
        font-size: 0.72rem;
        opacity: 0.5;
        margin: -0.35rem 0 0.55rem 0.6rem;
    }

    .conv-group-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.45;
        margin: 1.1rem 0 0.35rem 0.2rem;
    }
    .conv-group-label:first-of-type {
        margin-top: 0.5rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        border-radius: 8px;
        text-align: left;
        justify-content: flex-start;
        transition: background-color 0.12s ease, border-color 0.12s ease;
    }

    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent;
        border: 1px solid transparent;
        color: rgba(250, 250, 250, 0.85);
        font-weight: 400;
        padding: 0.4rem 0.7rem;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: rgba(52, 152, 219, 0.15) !important;
        border: 1px solid rgba(52, 152, 219, 0.4) !important;
        border-left: 3px solid #3498db !important;
        color: #3498db !important;
        font-weight: 600;
        padding: 0.4rem 0.7rem;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: rgba(52, 152, 219, 0.22) !important;
    }

    .conv-delete-btn button {
        opacity: 0.45;
        padding: 0.4rem 0.3rem !important;
    }
    .conv-delete-btn button:hover {
        opacity: 1;
        color: #e74c3c !important;
        background-color: rgba(231, 76, 60, 0.1) !important;
        border-color: rgba(231, 76, 60, 0.3) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        gap: 0.35rem;
        align-items: center;
    }

    .doc-row-name {
        font-size: 0.85rem;
        opacity: 0.85;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding-top: 0.45rem;
    }

    /* ---- User identity chip in sidebar ---- */
    .user-chip {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.6rem 0.7rem;
        border-radius: 10px;
        background: rgba(52, 152, 219, 0.08);
        border: 1px solid rgba(52, 152, 219, 0.18);
        margin-bottom: 0.5rem;
    }
    .user-avatar {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #3498db, #1abc9c);
        color: #fff; font-weight: 700; font-size: 0.85rem;
        flex-shrink: 0;
    }
    .user-meta { overflow: hidden; }
    .user-name {
        font-size: 0.85rem; font-weight: 600;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .user-role {
        font-size: 0.68rem; opacity: 0.6;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    defaults = {
        "api": APIClient(),
        "api_online": None,          # None = unknown, True / False
        "api_ready": None,
        "processed": False,
        "index_info": None,
        "messages": [],
        "selected_document": None,
        "conversation_id": None,
        "document_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Attach the AUTHENTICATED user id to the API client so every
# request carries X-User-Id.
st.session_state.api.user_id = USER_ID


# ============================================================
# HELPERS
# ============================================================

def start_new_conversation():
    """
    Clear local conversation state and tell the backend to
    finalize (summarize into long-term memory) the previous one.
    """
    old_id = st.session_state.get("conversation_id")

    if old_id:
        try:
            st.session_state.api.end_session(old_id)
        except Exception:
            pass

    st.session_state.conversation_id = None
    st.session_state.messages = []


def load_conversation(conversation_id: str):
    """
    Load a conversation's message history from the backend.

    The UI no longer reads SQLite directly — the backend is the
    single source of truth for conversation state.
    """
    st.session_state.conversation_id = conversation_id
    try:
        st.session_state.messages = st.session_state.api.get_messages(conversation_id)
    except APIError as e:
        st.session_state.messages = []
        st.error(f"Could not load conversation: {e}")
    except Exception as e:
        st.session_state.messages = []
        st.error(f"Unexpected error loading conversation: {e}")


def _parse_timestamp(updated_at: str) -> datetime:
    dt = datetime.fromisoformat(updated_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _relative_time(updated_at: str) -> str:
    try:
        dt = _parse_timestamp(updated_at)
    except ValueError:
        return ""
    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours // 24)
    if days < 7:
        return f"{days}d ago"
    weeks = int(days // 7)
    if days < 30:
        return f"{weeks}w ago"
    months = int(days // 30)
    return f"{months}mo ago"


def _group_label(updated_at: str) -> str:
    try:
        dt = _parse_timestamp(updated_at)
    except ValueError:
        return "Older"
    delta_days = (datetime.now(timezone.utc).date() - dt.date()).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 7:
        return "Previous 7 Days"
    return "Older"


def group_conversations_by_recency(conversations: list) -> dict:
    group_order = ["Today", "Yesterday", "Previous 7 Days", "Older"]
    grouped = {label: [] for label in group_order}
    for conversation in conversations:
        grouped[_group_label(conversation["updated_at"])].append(conversation)
    return {label: items for label, items in grouped.items() if items}


def check_api_status():
    """
    Hits /ready so we know the vector store is alive, not just the process.
    Also refreshes the document list on every page load.
    """
    try:
        st.session_state.api.ready()
        st.session_state.api_online = True
        st.session_state.api_ready = True
        docs = st.session_state.api.documents()
        st.session_state.document_names = docs
        st.session_state.processed = len(docs) > 0
        return True
    except APIError:
        st.session_state.api_online = True
        st.session_state.api_ready = False
        return False
    except Exception:
        st.session_state.api_online = False
        st.session_state.api_ready = False
        return False


def refresh_documents():
    """Reload the document list from the API and sync local state."""
    try:
        docs = st.session_state.api.documents()
        st.session_state.document_names = docs
        st.session_state.processed = len(docs) > 0
        if st.session_state.selected_document not in docs:
            st.session_state.selected_document = None
    except Exception:
        pass


api_ok = check_api_status()
api_online = bool(st.session_state.api_online)
api_ready = bool(st.session_state.api_ready)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f'<div class="brand-row brand-row-sm">{LOGO_SVG}'
        f'<div><div class="brand-title">Agentic RAG Studio</div>'
        f'<div class="brand-subtitle">Control Panel</div></div></div>',
        unsafe_allow_html=True,
    )

    # ---------------- Signed-in user chip ----------------
    initials = "".join(
        part[0].upper() for part in (USER_NAME or USER_EMAIL or "?").split()[:2]
    ) or "?"

    st.markdown(
        f'<div class="user-chip">'
        f'  <div class="user-avatar">{initials}</div>'
        f'  <div class="user-meta">'
        f'    <div class="user-name" title="{USER_EMAIL}">{USER_NAME or USER_EMAIL}</div>'
        f'    <div class="user-role">{USER_ROLE} · {TENANT_ID}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        st.session_state.auth_manager.logout()
        st.rerun()

    st.divider()

    # ---------------- Conversations ----------------
    st.markdown("### 💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_conversation()
        st.rerun()

    # Scoped to this AUTHENTICATED user only.
    # Fetched from the backend over HTTP — the frontend no longer
    # touches the conversation SQLite database directly.
    try:
        saved_conversations = st.session_state.api.list_conversations()
    except APIError:
        saved_conversations = []
    except Exception:
        saved_conversations = []

    if not saved_conversations:
        st.caption("No saved conversations yet — ask a question to start one.")
    else:
        for group_label, conversations_in_group in group_conversations_by_recency(saved_conversations).items():
            st.markdown(
                f'<div class="conv-group-label">{group_label}</div>',
                unsafe_allow_html=True,
            )

            for conversation in conversations_in_group:
                is_active = conversation["id"] == st.session_state.conversation_id
                row_select, row_delete = st.columns([6, 1])

                with row_select:
                    if st.button(
                        f"💬 {conversation['title']}",
                        key=f"open_{conversation['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        load_conversation(conversation["id"])
                        st.rerun()

                with row_delete:
                    st.markdown('<div class="conv-delete-btn">', unsafe_allow_html=True)
                    if st.button("✕", key=f"delete_{conversation['id']}", use_container_width=True):
                        try:
                            st.session_state.api.delete_session(conversation["id"])
                        except APIError as e:
                            st.error(f"Could not delete conversation: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error deleting conversation: {e}")

                        if is_active:
                            start_new_conversation()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="conv-timestamp">{_relative_time(conversation["updated_at"])}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ---------------- API status badge ----------------
    if not api_online:
        st.markdown(
            '<div class="status-badge status-offline">● API Offline</div>',
            unsafe_allow_html=True,
        )
    elif not api_ready:
        st.markdown(
            '<div class="status-badge status-stale">⚠️ API Not Ready</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.processed:
        st.markdown(
            '<div class="status-badge status-ready">● Vector Index Active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-idle">○ Awaiting Documents</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------------- Indexed documents ----------------
    st.markdown("### 📄 Indexed Documents")

    if not st.session_state.document_names:
        st.caption("No documents indexed yet.")
    else:
        for name in st.session_state.document_names:
            name_col, del_col = st.columns([5, 1])

            with name_col:
                st.markdown(
                    f'<div class="doc-row-name" title="{name}">📎 {name}</div>',
                    unsafe_allow_html=True,
                )

            with del_col:
                st.markdown('<div class="conv-delete-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_doc_{name}", use_container_width=True):
                    try:
                        st.session_state.api.delete_document(name)
                        refresh_documents()
                        st.rerun()
                    except APIError as e:
                        st.error(str(e))
                st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ---------------- Utilities ----------------
    st.markdown("### 🛠️ Utilities")
    if st.button("Reset Session", type="secondary", use_container_width=True):
        preserved = {
            "auth_manager": st.session_state.auth_manager,
            "authenticated": True,
            "user_id": USER_ID,
            "user_email": USER_EMAIL,
            "user_name": USER_NAME,
            "user_role": USER_ROLE,
            "tenant_id": TENANT_ID,
        }
        st.session_state.clear()
        st.session_state.update(preserved)
        st.rerun()
    st.caption(
        "Resets the in-memory index and UI state for this session. "
        "Saved conversations in the sidebar are not affected."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    f'<div class="brand-row brand-row-lg">{LOGO_SVG}'
    f'<div><div class="brand-title-lg">Agentic RAG Studio</div>'
    f'<div class="brand-subtitle-lg">Knowledge retrieval and reasoning engine powered by agentic search.</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

if not api_online:
    backend_url = getattr(st.session_state.api, "base_url", "the configured backend URL")
    st.error(
        f"⚠️ Cannot reach the FastAPI backend at **{backend_url}**. "
        "Ensure the API service is running and the `BACKEND_URL` environment "
        "variable is set correctly on this service."
    )
elif not api_ready:
    st.warning(
        "⚠️ The API is reachable but not ready. This usually means the vector store "
        "or Groq credentials are unavailable. Check the server logs."
    )


# ============================================================
# UPLOAD & INGESTION
# ============================================================

with st.container(border=True):
    col_upload, col_action = st.columns([3, 1], vertical_alignment="bottom")

    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload Context Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT",
        )

    with col_action:
        process_btn = st.button(
            "🚀 Build Index",
            type="primary",
            use_container_width=True,
            disabled=(not bool(uploaded_files)) or (not api_online),
        )

    replace_index = st.checkbox(
        "Replace existing index (deletes all currently indexed documents)",
        value=False,
        help="Leave unchecked to ADD the uploaded files to the existing index.",
    )

if process_btn and uploaded_files:
    try:
        with st.status("🧠 Processing knowledge base...", expanded=True) as status:
            st.write("Uploading documents to API...")
            info = st.session_state.api.upload(uploaded_files, replace=replace_index)

            st.write("Refreshing document list...")
            st.session_state.document_names = info.get("document_names", [])
            st.session_state.processed = True
            st.session_state.index_info = info
            if replace_index:
                st.session_state.selected_document = None

            status.update(label="✅ Index Ready!", state="complete", expanded=False)
            st.rerun()

    except APIError as e:
        st.error(f"API error while building index: {e}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")


# ============================================================
# METRICS CARD
# ============================================================

if st.session_state.processed and st.session_state.index_info:
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Indexed Documents", st.session_state.index_info.get("documents", 0))
        m2.metric("Generated Chunks", st.session_state.index_info.get("chunks", 0))
        m3.metric("Embedding Dim", st.session_state.index_info.get("embedding_dimension", "-"))


# ============================================================
# DOCUMENT SCOPE SELECTOR
# ============================================================

if st.session_state.processed:
    document_names = st.session_state.document_names

    if len(document_names) > 1:
        with st.container(border=True):
            scope_options = ["All Documents"] + document_names
            current_selection = st.session_state.selected_document
            default_index = (
                scope_options.index(current_selection)
                if current_selection in document_names
                else 0
            )

            chosen = st.selectbox(
                "📄 Ask about",
                options=scope_options,
                index=default_index,
                help="Restrict questions to a single uploaded document, or search across all of them.",
            )

            st.session_state.selected_document = (
                None if chosen == "All Documents" else chosen
            )
    else:
        st.session_state.selected_document = None

if st.session_state.messages and not st.session_state.processed:
    st.info(
        "You're viewing a saved conversation, but no documents are loaded in this "
        "session yet. You can read the history below — upload and build the index "
        "again to keep asking questions in this chat."
    )

st.divider()


# ============================================================
# CHAT INTERFACE
# ============================================================

def render_sources(sources):
    """Render a list of {source, page, content} dicts."""
    if not sources:
        return
    with st.expander("📚 Referenced Sources"):
        for idx, source in enumerate(sources, start=1):
            src_name = source.get("source", "Unknown")
            page = source.get("page", "N/A")
            st.markdown(f"**Source {idx}:** `{src_name}` | *Page {page}*")
            snippet = source.get("snippet") or source.get("content") or ""
            if snippet:
                st.caption(snippet)


def sources_from_stream_event(event_sources) -> list:
    """Map server Source schema {source, page, snippet} into UI format."""
    out = []
    for s in event_sources or []:
        out.append({
            "source": s.get("source", "Unknown Source"),
            "page": s.get("page", "N/A"),
            "content": s.get("snippet", ""),
        })
    return out


# Render message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            render_sources(message["sources"])


# Chat input
scoped_document = st.session_state.selected_document
chat_placeholder = (
    f"Ask a question about {scoped_document}..."
    if scoped_document
    else "Ask a question based on your uploaded documents..."
)

if question := st.chat_input(chat_placeholder):

    if not api_online:
        st.error("⚠️ The FastAPI backend is not reachable. Please start it and try again.")
        st.stop()

    if not st.session_state.processed:
        st.warning("⚠️ Please upload and process documents before starting the query engine.")
        st.stop()

    # ---- Persist + render user turn (server also persists) ----
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ---- Assistant turn: STREAMING ----
    with st.chat_message("assistant"):
        if scoped_document:
            st.markdown(
                f'<div class="scope-badge">📄 Scoped to: {scoped_document}</div>',
                unsafe_allow_html=True,
            )

        placeholder = st.empty()
        buffer = ""
        collected_sources = []
        memory_notice = st.empty()
        error_holder = st.empty()

        try:
            for event in st.session_state.api.chat_stream(
                question,
                selected_document=scoped_document,
                session_id=st.session_state.conversation_id,
            ):
                if "error" in event:
                    error_holder.error(f"API error: {event['error']}")
                    break

                if "token" in event:
                    buffer += event["token"]
                    placeholder.markdown(buffer + "▌")

                if event.get("done"):
                    collected_sources = event.get("sources", [])

                    server_session_id = event.get("session_id")
                    if server_session_id:
                        st.session_state.conversation_id = server_session_id

                    memories_used = event.get("memories_used", 0)
                    if memories_used:
                        memory_notice.caption(
                            f"💭 Recalled {memories_used} "
                            f"{'memory' if memories_used == 1 else 'memories'} "
                            f"from past sessions"
                        )

                    placeholder.markdown(buffer)
                    break

        except APIError as e:
            error_holder.error(f"API error: {e}")
        except Exception as e:
            error_holder.error(f"An error occurred while handling your query: {e}")

        source_dicts = sources_from_stream_event(collected_sources)
        render_sources(source_dicts)

        if buffer:
            st.session_state.messages.append({
                "role": "assistant",
                "content": buffer,
                "sources": source_dicts,
            })