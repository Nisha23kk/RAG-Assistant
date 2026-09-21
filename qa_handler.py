"""
qa_handler.py

RAG question-answering helpers used by app.py's main chat area:

- get_chat_llm             : cached Groq chat client
- get_document_source_label: formats a citation label for one retrieved chunk
- answer_question          : runs the RAG engine and renders the answer,
                              sources and performance metrics in the UI
"""

import os
import time

import streamlit as st
from langchain_groq import ChatGroq

from rag_engine import answer_question as run_rag_engine


# ============================================================
# Cached LLM Client
# ============================================================

@st.cache_resource
def get_chat_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv(
            "GROQ_API_KEY"
        )
    )


# ============================================================
# Document Source Helper
# ============================================================

def get_document_source_label(doc):

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


    return (
        f"📄 "
        f"{os.path.basename(str(filename))} "
        f"({location})"
    )


# ============================================================
# RAG Answer Helper
# ============================================================

def answer_question(question):
    """
    Unified AI question answering.

    Priority:

    1. Uploaded documents
    2. Web search if document answer is unavailable
    3. Web search directly when no document is uploaded
    """

    retrieval_start = time.time()


    try:

        answer, sources, source_type = (
            run_rag_engine(
                question,
                st.session_state.db
            )
        )


    except Exception as e:

        error_text = str(e)


        if (
            "429" in error_text
            or "rate_limit"
            in error_text.lower()
        ):

            answer = (
                "⚠️ Groq's daily token limit "
                "for this model has been reached, "
                "so I can't answer right now. "
                "Try again later or switch to "
                "a smaller model."
            )

        else:

            answer = (
                "Sorry, something went wrong "
                "while answering: "
                f"{error_text}"
            )


        sources = []

        source_type = None


    total_time = (
        time.time()
        - retrieval_start
    )


    placeholder = st.empty()

    placeholder.markdown(
        answer
    )


    # ========================================================
    # Document Source
    # ========================================================

    if source_type == "document":

        st.markdown(
            "### 📄 Relevant Document Sources"
        )


        seen_sources = set()


        for doc in sources:

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


            source_key = (
                str(filename),
                str(page),
                str(slide),
                str(sheet)
            )


            if source_key not in seen_sources:

                seen_sources.add(
                    source_key
                )


                st.write(
                    get_document_source_label(
                        doc
                    )
                )


    # ========================================================
    # Web Source
    # ========================================================

    elif source_type == "web":

        st.markdown(
            "### 🌐 Web Sources"
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

                st.markdown(
                    f"{index}. "
                    f"[{title}]({url})"
                )

            else:

                st.write(
                    f"{index}. {title}"
                )


    # ========================================================
    # Performance Metrics
    # ========================================================

    st.markdown(
        "### Performance Metrics"
    )


    m1, m2 = st.columns(2)


    with m1:

        st.metric(
            "Response Time",
            f"{total_time:.2f}s"
        )


    with m2:

        if source_type == "document":

            st.metric(
                "Answer Source",
                "📄 Uploaded Files"
            )

        elif source_type == "web":

            st.metric(
                "Answer Source",
                "🌐 Web Search"
            )

        else:

            st.metric(
                "Answer Source",
                "None"
            )


    return (
        answer,
        sources,
        source_type
    )
