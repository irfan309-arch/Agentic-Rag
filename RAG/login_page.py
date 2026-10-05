# login_page.py
"""
Standalone login / signup UI for Agentic RAG Studio.

Renders BEFORE the main app when the user is not authenticated.
Same visual identity as app.py (brand mark, gradient, dark theme).
"""

from __future__ import annotations

import streamlit as st

from auth_manager import AuthManager


# Reuse the same SVG mark so the login page matches the main app.
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


_LOGIN_CSS = """
<style>
    /* Center the login card visually */
    section.main > div.block-container {
        max-width: 480px;
        padding-top: 5vh;
    }

    .login-brand {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .login-brand svg {
        width: 64px; height: 64px; display: block;
    }
    .login-title {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(90deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .login-subtitle {
        font-size: 0.9rem;
        opacity: 0.65;
    }

    /* Card wrapper */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 14px !important;
        border: 1px solid rgba(250, 250, 250, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
    }

    .login-footer {
        text-align: center;
        font-size: 0.75rem;
        opacity: 0.4;
        margin-top: 2rem;
    }
</style>
"""


def render_login_page() -> None:
    """
    Render login + signup UI.
    Called from app.py when st.session_state.authenticated is False.
    Returns None; login() sets session_state.authenticated = True.
    """
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    # Ensure the manager exists
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthManager()
        


    # -------- Brand header --------
    st.markdown(
        f'<div class="login-brand">{LOGO_SVG}'
        f'<div class="login-title">Agentic RAG Studio</div>'
        f'<div class="login-subtitle">Sign in to continue to your knowledge workspace.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -------- Card with tabs --------
    with st.container(border=True):
        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])

        # ================== LOGIN ==================
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input(
                    "Email",
                    key="login_email",
                    placeholder="you@company.com",
                    autocomplete="email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="••••••••",
                    autocomplete="current-password",
                )
                submitted = st.form_submit_button(
                    "Login", use_container_width=True, type="primary"
                )

                if submitted:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    else:
                        ok, msg = st.session_state.auth_manager.login(email, password)
                        if ok:
                            st.rerun()
                        else:
                            st.error(msg)

        # ================== SIGN UP ==================
        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                new_name = st.text_input(
                    "Full Name",
                    key="signup_name",
                    placeholder="Jane Doe",
                    autocomplete="name",
                )
                new_email = st.text_input(
                    "Email",
                    key="signup_email",
                    placeholder="you@company.com",
                    autocomplete="email",
                )
                new_password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                    placeholder="At least 8 chars, letters + digits",
                    autocomplete="new-password",
                )
                confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="signup_confirm",
                    placeholder="Re-enter password",
                    autocomplete="new-password",
                )
                role = st.selectbox(
                    "Account Type",
                    options=["user", "admin"],
                    index=0,
                    help="Admin can ingest/delete documents. Choose 'user' unless you need admin rights.",
                )

                submitted = st.form_submit_button(
                    "Create Account", use_container_width=True, type="primary"
                )

                if submitted:
                    if not new_email or not new_password:
                        st.error("Email and password are required.")
                    elif new_password != confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = st.session_state.auth_manager.signup(
                            email=new_email,
                            password=new_password,
                            name=new_name,
                            role=role,
                        )
                        if ok:
                            st.success(msg)
                            st.info("Switch to the **Login** tab to sign in.")
                        else:
                            st.error(msg)

    # -------- Footer --------
    st.markdown(
        '<div class="login-footer">'
        'Local account storage — passwords hashed with bcrypt.'
        '</div>',
        unsafe_allow_html=True,
    )