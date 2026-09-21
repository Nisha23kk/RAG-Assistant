import os
import time
import tempfile
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from ingest import load_single_file, get_embeddings
from langchain_core.messages import HumanMessage, AIMessage
from auth import (
    create_db,
    logout_session,
)
from auth_ui import (
    get_cookie_manager,
    init_auth_session_state,
    restore_login_from_cookie,
    show_login_signup_page,
    SESSION_COOKIE_NAME,
)
from session_state import init_app_session_state
from tts import (
    clean_text_for_voice,
    speak_edge_tts,
)
from qa_handler import (
    get_chat_llm,
    get_document_source_label,
    answer_question,
)
from chat_manager import (
    save_chat,
    load_chat,
    list_chats,
    generate_chat_name,
    delete_chat
)
from share_utils import (
    conversation_to_text,
    conversation_to_pdf
)
from image_generator import generate_image
import base64
from analysis_report import (
    generate_analysis_report,
    RateLimitError
)
from rag_engine import answer_question as run_rag_engine

# Reuse the Whisper-based voice assistant + its system commands
# (open website in a NEW TAB, time, date, reminder, web search).
from voice_assistant import listen as whisper_listen
from voice_assistant import try_handle_system_command
from voice_assistant import open_url_in_new_chrome_tab

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Create Database
# ============================================================

create_db()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Persistent Login Session
# ============================================================

cookie_manager = get_cookie_manager()

init_auth_session_state()

restore_login_from_cookie(cookie_manager)


# ============================================================
# LOGIN / SIGNUP PAGE
# ============================================================
# Renders the page and calls st.stop() itself when not logged in.
# ============================================================

show_login_signup_page(cookie_manager)


# ============================================================
# Everything Below This Point Requires Login
# ============================================================


# ============================================================
# Application Session State
# ============================================================

init_app_session_state()


# ============================================================
# Cached LLM Client
# ============================================================

llm = get_chat_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "AI Document Assistant"
    )

    st.write(
        f"Logged in as: "
        f"**{st.session_state.username}**"
    )


    # ========================================================
    # New Chat
    # ========================================================

    if st.button(
        "➕ New Chat"
    ):

        if st.session_state.messages:

            save_chat(
                st.session_state.username,
                st.session_state.chat_name,
                st.session_state.messages
            )

        st.session_state.chat_name = (
            generate_chat_name()
        )

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # Chat History
    # ========================================================

    st.subheader(
        "Chat History"
    )

    chats = list_chats(
        st.session_state.username
    )


    # --------------------------------------------------------
    # Delete confirmation state
    # --------------------------------------------------------

    if "delete_chat_confirmation" not in st.session_state:

        st.session_state.delete_chat_confirmation = None


    for chat in chats:

        chat_col, delete_col = st.columns(
            [5, 1]
        )


        # ----------------------------------------------------
        # Open Chat
        # ----------------------------------------------------

        with chat_col:

            if st.button(
                chat,
                key=f"open_chat_{chat}"
            ):

                st.session_state.chat_name = (
                    chat
                )

                st.session_state.messages = (
                    load_chat(
                        st.session_state.username,
                        chat
                    )
                )

                st.session_state.delete_chat_confirmation = None

                st.rerun()


        # ----------------------------------------------------
        # Delete button
        # ----------------------------------------------------

        with delete_col:

            if st.button(
                "🗑️",
                key=f"delete_chat_{chat}"
            ):

                st.session_state.delete_chat_confirmation = (
                    chat
                )

                st.rerun()


        # ----------------------------------------------------
        # Delete Confirmation
        # ----------------------------------------------------

        if (
            st.session_state.delete_chat_confirmation
            == chat
        ):

            st.warning(
                f"Delete **{chat}**?"
            )

            confirm_col, cancel_col = st.columns(
                2
            )


            with confirm_col:

                if st.button(
                    "Delete",
                    key=f"confirm_delete_{chat}"
                ):

                    deleted = delete_chat(
                        st.session_state.username,
                        chat
                    )

                    if deleted:

                        if (
                            st.session_state.chat_name
                            == chat
                        ):

                            st.session_state.chat_name = (
                                generate_chat_name()
                            )

                            st.session_state.messages = []


                        st.session_state.delete_chat_confirmation = None

                        st.rerun()

                    else:

                        st.error(
                            "Could not delete this chat."
                        )


            with cancel_col:

                if st.button(
                    "Cancel",
                    key=f"cancel_delete_{chat}"
                ):

                    st.session_state.delete_chat_confirmation = None

                    st.rerun()


    st.divider()


    # ========================================================
    # Share Conversation
    # ========================================================

    st.subheader(
        "Share Conversation"
    )


    if st.session_state.messages:

        text_data = conversation_to_text(
            st.session_state.messages
        )

        st.download_button(
            label="Download as TXT",
            data=text_data,
            file_name=(
                f"{st.session_state.chat_name}.txt"
            ),
            mime="text/plain"
        )


        pdf_data = conversation_to_pdf(
            st.session_state.messages
        )

        st.download_button(
            label="Download as PDF",
            data=pdf_data,
            file_name=(
                f"{st.session_state.chat_name}.pdf"
            ),
            mime="application/pdf",
            key="download_chat_pdf"
        )


    st.divider()


    # ========================================================
    # Logout
    # ========================================================

    if st.button(
        "Logout",
        key="logout_button"
    ):

        session_token = (
            cookie_manager.get(
                SESSION_COOKIE_NAME
            )
        )


        if session_token:

            logout_session(
                session_token
            )


        try:

            cookie_manager.delete(
                SESSION_COOKIE_NAME
            )

            time.sleep(0.5)

        except Exception:

            pass


        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.auth_checked = True

        st.session_state.messages = []

        st.session_state.voice_conversation = []

        st.rerun()


    st.divider()


    # ========================================================
    # Dashboard Metrics
    # ========================================================

    st.markdown(
        "### Dashboard"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Documents",
            st.session_state.get(
                "uploaded_count",
                0
            )
        )


    with c2:

        st.metric(
            "Chunks",
            st.session_state.get(
                "chunk_count",
                0
            )
        )


    st.divider()


    # ========================================================
    # Clear Chat
    # ========================================================

    if st.button(
        "🗑 Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # Clear Voice Conversation
    # ========================================================

    if st.session_state.voice_conversation:

        if st.button(
            "Clear Voice Conversation"
        ):

            st.session_state.voice_conversation = []

            st.rerun()


    st.divider()


    # ========================================================
    # Quick Actions  (all websites open in a NEW TAB)
    # ========================================================

    st.markdown(
        "### Quick Actions"
    )


    qa1, qa2 = st.columns(2)


    with qa1:

        if st.button(
            "Open Google"
        ):

            open_url_in_new_chrome_tab(
                "https://www.google.com"
            )


        if st.button(
            "Open YouTube"
        ):

            open_url_in_new_chrome_tab(
                "https://www.youtube.com"
            )

    with qa2:

        if st.button(
            "Open LinkedIn"
        ):

            open_url_in_new_chrome_tab(
                "https://www.linkedin.com"
            )

        if st.button(
            "Open Notepad"
        ):

            subprocess.Popen(
                "notepad.exe"
            )


# ============================================================
# Main Chat Area
# ============================================================

st.title(
    "🤖 AI Document Assistant"
)

st.caption(
    "Ask questions about your uploaded "
    "documents using text or voice."
)


# ============================================================
# Upload Documents
# ============================================================

st.subheader(
    "📄 Upload Documents"
)


uploaded_files = st.file_uploader(
    "Upload Documents",
    type=[
        "pdf",
        "xlsx",
        "xls",
        "csv",
        "docx",
        "doc",
        "txt",
        "pptx"
    ],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


if uploaded_files:

    current_names = [
        f.name
        for f in uploaded_files
    ]


    if (
        st.session_state.get(
            "last_uploaded"
        )
        != current_names
    ):

        st.session_state.last_uploaded = (
            current_names
        )

        st.session_state.uploaded_files = (
            uploaded_files
        )


        documents = []

        document_groups = []


        # ====================================================
        # Process Each File
        # ====================================================

        for uploaded_file in uploaded_files:

            extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getvalue()
                )

                file_path = tmp_file.name


            try:

                file_documents = (
                    load_single_file(
                        file_path
                    )
                )


                for doc in file_documents:

                    doc.metadata["filename"] = (
                        uploaded_file.name
                    )

                    doc.metadata["source"] = (
                        uploaded_file.name
                    )


                document_groups.append(
                    {
                        "filename": uploaded_file.name,
                        "documents": file_documents
                    }
                )


                documents.extend(
                    file_documents
                )


            except Exception as e:

                st.error(
                    f"Error processing "
                    f"{uploaded_file.name}: {e}"
                )


            finally:

                try:

                    os.remove(
                        file_path
                    )

                except Exception:

                    pass


        # ====================================================
        # Create Chunks
        # ====================================================

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150
            )
        )


        chunks = splitter.split_documents(
            documents
        )


        if not chunks:

            st.error(
                "No text chunks could be "
                "created from the uploaded files."
            )

            st.stop()


        # ====================================================
        # Cached Embeddings
        # ====================================================

        embeddings = get_embeddings()


        # ====================================================
        # Build Document Text
        # ====================================================

        document_text = ""


        for doc in documents:

            metadata = (
                doc.metadata or {}
            )


            filename = metadata.get(
                "filename",
                metadata.get(
                    "source",
                    "Unknown file"
                )
            )


            page = metadata.get(
                "page"
            )

            slide = metadata.get(
                "slide"
            )

            sheet = metadata.get(
                "sheet"
            )


            if page is not None:

                try:

                    location = (
                        f"Page {int(page) + 1}"
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    location = (
                        f"Page {page}"
                    )

            elif slide is not None:

                location = (
                    f"Slide {slide}"
                )

            elif sheet is not None:

                location = (
                    f"Sheet: {sheet}"
                )

            else:

                location = "Document"


            document_text += (
                "\n\n"
                "==================================================\n"
                f"FILE: {os.path.basename(str(filename))}\n"
                f"LOCATION: {location}\n"
                "==================================================\n"
                f"{doc.page_content}\n"
            )


        st.session_state.document_text = (
            document_text
        )


        # ====================================================
        # Create FAISS Database
        # ====================================================

        st.session_state.db = (
            FAISS.from_documents(
                chunks,
                embeddings
            )
        )


        st.session_state.document_groups = (
            document_groups
        )


        # Reset reports only when uploaded
        # file set actually changes.

        st.session_state.analysis_reports = {}


        st.session_state.chunk_count = (
            len(chunks)
        )


        st.session_state.uploaded_count = (
            len(uploaded_files)
        )


        st.success(
            f"{len(uploaded_files)} file(s) "
            "processed successfully!"
        )


# ============================================================
# Individual Document Analysis
# ============================================================

if st.session_state.document_groups:

    st.divider()

    st.subheader(
        "📋 Document Analysis"
    )


    st.caption(
        "Each uploaded document is analyzed "
        "separately using its own document "
        "type and content."
    )


    col_gen, col_regen = st.columns(2)


    with col_gen:

        generate_clicked = st.button(
            "📋 Generate Analysis Reports"
        )


    with col_regen:

        regenerate_clicked = st.button(
            "🔄 Regenerate All"
        )


    if regenerate_clicked:

        st.session_state.analysis_reports = {}

        generate_clicked = True


    if generate_clicked:

        pending_groups = [
            g
            for g in st.session_state.document_groups
            if (
                g["filename"]
                not in st.session_state.analysis_reports
            )
        ]


        if not pending_groups:

            st.info(
                "All uploaded documents already "
                "have a report. Use 'Regenerate All' "
                "to redo them."
            )


        with st.spinner(
            "Analyzing documents..."
        ):

            for document_group in pending_groups:

                filename = (
                    document_group["filename"]
                )

                file_documents = (
                    document_group["documents"]
                )


                file_text = ""


                for doc in file_documents:

                    metadata = (
                        doc.metadata or {}
                    )


                    page = metadata.get(
                        "page"
                    )

                    slide = metadata.get(
                        "slide"
                    )

                    sheet = metadata.get(
                        "sheet"
                    )


                    if page is not None:

                        try:

                            location = (
                                f"Page {int(page) + 1}"
                            )

                        except (
                            ValueError,
                            TypeError
                        ):

                            location = (
                                f"Page {page}"
                            )

                    elif slide is not None:

                        location = (
                            f"Slide {slide}"
                        )

                    elif sheet is not None:

                        location = (
                            f"Sheet: {sheet}"
                        )

                    else:

                        location = "Document"


                    file_text += (
                        f"\n\n--- {location} ---\n"
                        f"{doc.page_content}\n"
                    )


                try:

                    report = (
                        generate_analysis_report(
                            text=file_text,
                            filename=filename
                        )
                    )


                    st.session_state.analysis_reports[
                        filename
                    ] = report


                except RateLimitError as e:

                    st.session_state.analysis_reports[
                        filename
                    ] = (
                        "# Document Analysis Report\n\n"
                        f"Unable to analyze `{filename}` "
                        "- Groq daily token limit reached.\n\n"
                        f"{str(e)}"
                    )


                    st.warning(
                        f"Rate limit hit while analyzing "
                        f"{filename} — stopping remaining "
                        "analyses for now."
                    )

                    break


                except Exception as e:

                    st.session_state.analysis_reports[
                        filename
                    ] = (
                        "# Document Analysis Report\n\n"
                        f"Unable to analyze `{filename}`.\n\n"
                        f"Error: {str(e)}"
                    )


        st.rerun()


    # ========================================================
    # Display Reports
    # ========================================================

    if st.session_state.analysis_reports:

        st.divider()

        st.subheader(
            "📊 Individual Document Reports"
        )


        for filename, report in (
            st.session_state.analysis_reports.items()
        ):

            st.markdown(
                f"## 📄 {filename}"
            )


            st.markdown(
                report
            )


            safe_name = os.path.splitext(
                filename
            )[0]


            st.download_button(
                label=(
                    f"Download "
                    f"{filename} Summary"
                ),
                data=report,
                file_name=(
                    f"{safe_name}_summary.txt"
                ),
                mime="text/plain",
                key=(
                    f"download_summary_{filename}"
                )
            )


            st.divider()


    # ========================================================
    # Uploaded Documents
    # ========================================================

    st.subheader(
        "📁 Uploaded Documents"
    )


    for file in (
        st.session_state.uploaded_files
    ):

        extension = os.path.splitext(
            file.name
        )[1].upper()


        st.write(
            f"📄 {file.name} "
            f"({extension})"
        )


# ============================================================
# Voice Assistant
# ============================================================

st.subheader(
    "🎙 Voice Assistant"
)


if st.button(
    "🎙 Start Voice Assistant",
    key="start_voice_assistant"
):

    wave = st.empty()

    transcript_box = st.container()


    # --------------------------------------------------------
    # ONE CLICK = ONE QUESTION
    # --------------------------------------------------------

    wave.markdown(
        "### 🎤 Listening... "
        "(bolna shuru karein)"
    )


    question = whisper_listen()


    wave.empty()


    # --------------------------------------------------------
    # No valid speech
    # --------------------------------------------------------

    if not question:

        transcript_box.warning(
            "Kuch samajh nahi aaya. "
            "Voice Assistant dobara start karein."
        )


    # --------------------------------------------------------
    # Stop command
    # --------------------------------------------------------

    elif question.strip().lower() in (
        "stop",
        "exit",
        "quit",
        "band karo",
        "ruk jao"
    ):

        transcript_box.info(
            "Voice Assistant stopped."
        )


    # --------------------------------------------------------
    # Process ONE question
    # --------------------------------------------------------

    else:

        transcript_box.markdown(
            f"**You:** {question}"
        )


        # ====================================================
        # System Command Check
        #
        # "open youtube", "open <any website>", "youtube
        # kholo", time, date, reminders, "search ..." are
        # handled here. Websites open in a NEW TAB.
        # Anything else goes to the RAG engine.
        # ====================================================

        command_handled, command_message = (
            try_handle_system_command(
                question,
                announce=False
            )
        )

        if command_handled:

            answer = command_message

            sources = []

            source_type = "system"

        else:

            # ================================================
            # RAG Answer
            # ================================================

            try:

                answer, sources, source_type = (
                    run_rag_engine(
                        question,
                        st.session_state.db
                    )
                )


            except Exception as e:

                answer = (
                    "Sorry, jawaab dete waqt "
                    f"error aaya: {e}"
                )

                sources = []

                source_type = "none"


        # ====================================================
        # Save Voice Conversation
        # ====================================================

        st.session_state.voice_conversation.append(
            HumanMessage(
                content=question
            )
        )


        st.session_state.voice_conversation.append(
            AIMessage(
                content=answer
            )
        )


        # ====================================================
        # Display Original Markdown Answer
        # ====================================================

        transcript_box.markdown(
            f"**Assistant:** {answer}"
        )


        # ====================================================
        # Document Sources
        # ====================================================

        if (
            source_type == "document"
            and sources
        ):

            transcript_box.markdown(
                "**📄 Document Sources**"
            )


            seen_sources = set()


            for doc in sources:

                label = (
                    get_document_source_label(
                        doc
                    )
                )


                if label not in seen_sources:

                    seen_sources.add(
                        label
                    )

                    transcript_box.write(
                        label
                    )


        # ====================================================
        # Web Sources
        # ====================================================

        elif (
            source_type == "web"
            and sources
        ):

            transcript_box.markdown(
                "**🌐 Web Sources**"
            )


            for index, source in enumerate(
                sources,
                start=1
            ):

                title = source.get(
                    "title",
                    "Unknown"
                )


                url = source.get(
                    "url",
                    ""
                )


                if url:

                    transcript_box.markdown(
                        f"{index}. "
                        f"[{title}]({url})"
                    )

                else:

                    transcript_box.write(
                        f"{index}. {title}"
                    )


        # ====================================================
        # Document + Web Sources
        # ====================================================

        elif (
            source_type == "both"
            and isinstance(
                sources,
                dict
            )
        ):

            doc_sources = sources.get(
                "documents",
                []
            )


            web_sources = sources.get(
                "web",
                []
            )


            # ------------------------------------------------
            # Document sources
            # ------------------------------------------------

            if doc_sources:

                transcript_box.markdown(
                    "**📄 Document Sources**"
                )


                seen_sources = set()


                for doc in doc_sources:

                    label = (
                        get_document_source_label(
                            doc
                        )
                    )


                    if label not in seen_sources:

                        seen_sources.add(
                            label
                        )

                        transcript_box.write(
                            label
                        )


            # ------------------------------------------------
            # Web sources
            # ------------------------------------------------

            if web_sources:

                transcript_box.markdown(
                    "**🌐 Web Sources**"
                )


                for index, source in enumerate(
                    web_sources,
                    start=1
                ):

                    title = source.get(
                        "title",
                        "Unknown"
                    )


                    url = source.get(
                        "url",
                        ""
                    )


                    if url:

                        transcript_box.markdown(
                            f"{index}. "
                            f"[{title}]({url})"
                        )

                    else:

                        transcript_box.write(
                            f"{index}. {title}"
                        )


        # ====================================================
        # TEXT TO SPEECH
        #
        # Do NOT send raw Markdown answer to TTS.
        # ====================================================

        voice_answer = (
            clean_text_for_voice(
                answer
            )
        )


        if voice_answer:

            speak_edge_tts(
                voice_answer,
                container=transcript_box
            )


# ============================================================
# Display Previous Chat Messages
# ============================================================

st.divider()


for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        source_type = message.get(
            "source_type"
        )

        sources = message.get(
            "sources",
            []
        )


        # ----------------------------------------------------
        # Web Sources
        # ----------------------------------------------------

        if (
            message["role"]
            == "assistant"
            and source_type
            == "web"
            and sources
        ):

            st.markdown(
                "### 🌐 Web Sources"
            )


            for index, source in enumerate(
                sources,
                start=1
            ):

                title = source.get(
                    "title",
                    "Unknown source"
                )


                url = source.get(
                    "url",
                    ""
                )


                if url:

                    st.markdown(
                        f"{index}. "
                        f"[{title}]({url})"
                    )

                else:

                    st.write(
                        f"{index}. {title}"
                    )


        # ----------------------------------------------------
        # Document Sources
        # ----------------------------------------------------

        elif (
            message["role"]
            == "assistant"
            and source_type
            == "document"
            and sources
        ):

            st.markdown(
                "### 📄 Document Sources"
            )


            seen_sources = set()


            for source in sources:

                if hasattr(
                    source,
                    "metadata"
                ):

                    label = (
                        get_document_source_label(
                            source
                        )
                    )


                    metadata = (
                        source.metadata or {}
                    )


                    filename = metadata.get(
                        "filename",
                        metadata.get(
                            "source",
                            "Unknown file"
                        )
                    )


                    source_key = (
                        str(filename),
                        str(
                            metadata.get(
                                "page"
                            )
                        ),
                        str(
                            metadata.get(
                                "slide"
                            )
                        ),
                        str(
                            metadata.get(
                                "sheet"
                            )
                        )
                    )


                else:

                    filename = source.get(
                        "filename",
                        source.get(
                            "source",
                            "Unknown file"
                        )
                    )


                    source_key = (
                        str(filename),
                        str(
                            source.get(
                                "page"
                            )
                        ),
                        str(
                            source.get(
                                "slide"
                            )
                        ),
                        str(
                            source.get(
                                "sheet"
                            )
                        )
                    )


                    label = (
                        f"📄 {filename}"
                    )


                if source_key not in seen_sources:

                    seen_sources.add(
                        source_key
                    )

                    st.write(
                        label
                    )


# ============================================================
# Chat Input
# ============================================================

user_question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if user_question:

    # --------------------------------------------------------
    # Generate chat name for first question
    # --------------------------------------------------------

    if not st.session_state.messages:
        st.session_state.chat_name = (
            generate_chat_name(
                user_question
            )
        )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )


    lower_q = (
        user_question.lower()
    )


    # ========================================================
    # SYSTEM COMMAND CHECK  (NEW)
    #
    # Same logic as the voice assistant, so typing
    # "open youtube" / "open <any website>" / "github kholo"
    # opens the site in a NEW TAB instead of asking the RAG
    # engine. Time, date, reminders and "search ..." work too.
    # If it is not a system command, we fall through to image
    # generation / RAG below.
    # ========================================================

    command_handled, command_message = (
        try_handle_system_command(
            user_question,
            announce=False
        )
    )


    if command_handled:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                command_message
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": command_message,
                "source_type": "system",
                "sources": []
            }
        )


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    elif (
        lower_q.startswith(
            "generate an image"
        )
        or
        lower_q.startswith(
            "create an image"
        )
    ):

        prompt = (
            user_question
            .replace(
                "generate an image of",
                ""
            )
            .replace(
                "create an image of",
                ""
            )
            .replace(
                "generate an image",
                ""
            )
            .replace(
                "create an image",
                ""
            )
            .strip()
        )


        if not prompt:

            prompt = (
                "an abstract image"
            )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Generating image..."
            ):

                try:

                    image_base64 = (
                        generate_image(
                            prompt
                        )
                    )


                    image_bytes = (
                        base64.b64decode(
                            image_base64
                        )
                    )
                    st.image(
                        image_bytes,
                        caption=prompt,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(
                        "Could not generate image: "
                        f"{e}"
                    )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Generated image: {prompt}"
                )
            }
        )

    # ========================================================
    # NORMAL RAG QUESTION
    # ========================================================

    else:

        with st.chat_message(
            "assistant"
        ):

            answer, sources, source_type = (
                answer_question(
                    user_question
                )
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source_type": source_type,
                "sources": sources
            }
        )


    # ========================================================
    # Save Chat
    # ========================================================

    save_chat(
        st.session_state.username,
        st.session_state.chat_name,
        st.session_state.messages
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "AI Document Assistant | "
    "RAG + Web Search | "
    "Streamlit + LangChain + FAISS + "
    "Hugging Face Embeddings + Groq"
)