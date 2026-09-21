"""
auth_ui.py

Everything about the login/session UI layer that used to live at the
top of app.py:

- get_cookie_manager        : the persistent "remember me" cookie widget
- init_auth_session_state   : one-time session_state defaults for auth
- restore_login_from_cookie : silently re-logs the user in on refresh
- show_login_signup_page    : renders the Login / Sign Up screen

SPEED FIX:
The old code waited for the cookie component with up to 20 reruns
x 0.3s sleep = up to 6 seconds on every single fresh page load,
which is a big part of why the app felt slow to open. The cookie
value is available within the first couple of reruns in practice,
so this now caps out at 6 x 0.1s = 0.6s.
"""

import time

import streamlit as st
import extra_streamlit_components as stx

from auth import (
    register_user,
    login_user,
    create_session,
    validate_session,
    get_password_strength,
)

SESSION_COOKIE_NAME = "rag_session"

SESSION_MAX_AGE = 7 * 24 * 60 * 60  # 7 days

MAX_COOKIE_WAIT_ATTEMPTS = 6

COOKIE_WAIT_SECONDS = 0.1


# ============================================================
# Cookie Manager
# ============================================================

def get_cookie_manager():

    # Fixed key keeps the underlying JS component stable
    # across Streamlit reruns.
    if "cookie_manager" not in st.session_state:

        st.session_state.cookie_manager = stx.CookieManager(
            key="rag_cookie_manager"
        )

    return st.session_state.cookie_manager


# ============================================================
# Authentication Session State
# ============================================================

def init_auth_session_state():

    defaults = {
        "logged_in": False,
        "username": "",
        "auth_checked": False,
        "reset_signup_form": False,
        "cookie_wait_attempts": 0,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ============================================================
# Restore Login From Persistent Cookie
# ============================================================

def restore_login_from_cookie(cookie_manager):

    if st.session_state.auth_checked:
        return

    try:

        all_cookies = cookie_manager.get_all()

        session_token = (
            all_cookies or {}
        ).get(
            SESSION_COOKIE_NAME
        )

        # ----------------------------------------------------
        # Cookie found
        # ----------------------------------------------------

        if session_token:

            username = validate_session(
                session_token
            )

            if username:

                st.session_state.logged_in = True

                st.session_state.username = (
                    username
                )

            else:

                # Cookie exists but session is invalid
                # or expired.

                st.session_state.logged_in = False

                st.session_state.username = ""

                try:

                    cookie_manager.delete(
                        SESSION_COOKIE_NAME
                    )

                except Exception:
                    pass

            st.session_state.auth_checked = True

            st.session_state.cookie_wait_attempts = 0

        # ----------------------------------------------------
        # Cookie not available yet
        # ----------------------------------------------------

        else:

            st.session_state.cookie_wait_attempts += 1

            if (
                st.session_state.cookie_wait_attempts
                < MAX_COOKIE_WAIT_ATTEMPTS
            ):

                time.sleep(COOKIE_WAIT_SECONDS)

                st.rerun()

            # ------------------------------------------------
            # Assume no login cookie
            # ------------------------------------------------

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.auth_checked = True

            st.session_state.cookie_wait_attempts = 0

    except Exception as e:

        print(
            f"Authentication restore error: {e}"
        )

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.auth_checked = True

        st.session_state.cookie_wait_attempts = 0


# ============================================================
# LOGIN / SIGNUP PAGE
# ============================================================

def show_login_signup_page(cookie_manager):
    """
    Renders the Login / Sign Up page and stops the script here
    if the user is not logged in. Returns normally (does nothing)
    once the user is logged in, so the caller can just do:

        show_login_signup_page(cookie_manager)
        # everything below only runs when logged in
    """

    if st.session_state.logged_in:
        return

    st.title("AI Document Assistant")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        email = st.text_input(
            "Email",
            placeholder="Enter your Gmail",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", key="login_button"):

            result = login_user(
                email,
                password
            )

            if result.get("success"):

                username = result["username"]

                session_token = create_session(
                    username
                )

                cookie_manager.set(
                    SESSION_COOKIE_NAME,
                    session_token,
                    max_age=SESSION_MAX_AGE
                )

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.auth_checked = True

                st.rerun()

            else:

                st.error(
                    result.get(
                        "message",
                        "Invalid email or password."
                    )
                )

    # ========================================================
    # SIGN UP
    # ========================================================

    with signup_tab:

        name = st.text_input(
            "Full Name",
            placeholder="Enter your full name",
            key="signup_name"
        )

        email = st.text_input(
            "Email",
            placeholder="example@gmail.com",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        # ----------------------------------------------------
        # Password Strength
        # ----------------------------------------------------

        if password:

            strength = get_password_strength(
                password
            )

            score = strength["score"]
            label = strength["label"]

            if label == "Strong":
                st.success(
                    f"Password Strength: {label}"
                )

            elif label == "Medium":
                st.warning(
                    f"Password Strength: {label}"
                )

            else:
                st.error(
                    f"Password Strength: {label}"
                )

            st.caption(
                "Password must contain: "
                "6+ characters, uppercase, lowercase, "
                "number and special character."
            )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button(
            "Create Account",
            key="create_account_button"
        ):

            result = register_user(
                name,
                email,
                password,
                confirm_password
            )

            if result.get("success"):

                st.success(
                    "Account created successfully! "
                    "Please login."
                )

            else:

                st.error(
                    result.get(
                        "message",
                        "Registration failed."
                    )
                )

    st.stop()
