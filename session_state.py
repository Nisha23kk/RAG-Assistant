"""
session_state.py

One-time Streamlit session_state defaults for the main app
(everything except the auth/cookie state, which lives in auth_ui.py).
"""

import streamlit as st

from chat_manager import generate_chat_name


def init_app_session_state():

    defaults = {
        "db": None,
        "messages": [],
        "chunk_count": 0,
        "uploaded_count": 0,
        "uploaded_files": [],
        "voice_conversation": [],
        "document_text": "",
        "document_groups": [],
        "analysis_reports": {},
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value

    if "chat_name" not in st.session_state:

        st.session_state.chat_name = generate_chat_name()
