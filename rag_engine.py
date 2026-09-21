import os
from typing import Optional

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from ingest import get_embeddings
from analysis_report import (
    fast_llm,
    strong_llm,
    safe_llm_invoke,
    RateLimitError,
)

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


# -------------------------------------------------
# Load FAISS safely
# -------------------------------------------------

def load_vectorstore():
    """
    Load the FAISS vector database if it exists.
    """

    if not os.path.exists(VECTORSTORE_PATH):
        return None

    try:
        db = FAISS.load_local(
            VECTORSTORE_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
        return db

    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None


# -------------------------------------------------
# Web Search
# -------------------------------------------------

def web_search(query, max_results=5):
    """
    Search the web using DuckDuckGo.
    """

    try:

        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=max_results)

        formatted_results = []

        for result in results:

            title = result.get("title", "").strip()
            url = result.get("href", "").strip()
            snippet = result.get("body", "").strip()

            if title or url or snippet:
                formatted_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

        return formatted_results

    except Exception as e:
        print(f"Web search error: {e}")
        return []


def build_web_context(search_results):

    if not search_results:
        return ""

    context_parts = []

    for index, result in enumerate(search_results, start=1):

        title = result.get("title", "Unknown")
        url = result.get("url", "")
        snippet = result.get("snippet", "")

        context_parts.append(
            f"""
SOURCE {index}

Title:
{title}

URL:
{url}

Content:
{snippet}
"""
        )

    return "\n\n".join(context_parts)


def build_doc_context(docs):
    return "\n\n".join(
        doc.page_content for doc in docs if doc.page_content.strip()
    )


# =================================================
# FIX: Question Intent Router
# =================================================
# Decides whether a question should be answered from the uploaded
# documents only, the web only, or both - instead of the old
# "always try documents first, silently fall back to web" logic
# that hid rate-limit failures behind a generic message.
#
# A free keyword heuristic handles the obvious cases (no LLM call,
# no quota spent). Only genuinely ambiguous questions fall through
# to a single cheap fast_llm classification call.
# =================================================

FRESHNESS_KEYWORDS = (
    "today", "now", "current", "currently", "latest", "recent",
    "news", "this week", "this month", "this year", "update",
    "price", "stock", "score", "weather", "release date",
)


def classify_question_intent(question, db_available):

    if not db_available:
        return "web"

    lowered = question.lower()

    if any(keyword in lowered for keyword in FRESHNESS_KEYWORDS):
        return "both"

    prompt = f"""
You are a routing classifier for a document Q&A assistant.

The user has uploaded documents available in a database, and you
also have web search available.

Decide the best source to answer the question below. Choose
EXACTLY ONE:

DOCUMENT - the question is clearly about the uploaded document's
own content (its data, sections, numbers, names, etc.)

WEB - the question is general knowledge, current events, or
clearly unrelated to a private uploaded document.

BOTH - the question could benefit from both the uploaded document
and outside/current information.

Return ONLY one word: DOCUMENT, WEB, or BOTH.

QUESTION:
{question}
"""

    try:
        result = safe_llm_invoke(fast_llm, prompt).strip().upper()

    except RateLimitError:
        return "document"

    except Exception:
        return "document"

    if "BOTH" in result:
        return "both"
    if "WEB" in result:
        return "web"
    return "document"


# -------------------------------------------------
# Answer From Uploaded Documents
# -------------------------------------------------

def answer_from_documents(question, db):

    if db is None:
        return "NOT_FOUND", []

    """
    Try to answer the question using uploaded documents.

    The document contents are NOT printed here.
    This prevents uploaded document details from appearing
    repeatedly for every voice/text query.
    """

    try:
        docs = db.similarity_search(
            question,
            k=4
        )

    except Exception as e:
        print(f"Document retrieval error: {e}")
        return "NOT_FOUND", []

    if not docs:
        return "NOT_FOUND", []

    context = build_doc_context(docs)

    if not context.strip():
        return "NOT_FOUND", docs

    prompt = f"""
You are an intelligent AI document assistant.

Your task is to answer the user's question using ONLY the
uploaded document context provided below.

IMPORTANT RULES:

1. Use the uploaded document context as the primary source.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the context clearly contains the answer, answer naturally.
5. If the context does NOT contain enough information to answer
   the question, respond with exactly:

NOT_FOUND

6. Do not say NOT_FOUND merely because the answer is not written
   word-for-word. Use the context and reasonable inference when
   the information is clearly supported.

Uploaded Document Context:
--------------------------
{context}
--------------------------

User Question:
{question}
"""

    answer = safe_llm_invoke(
        strong_llm,
        prompt
    )

    return answer, docs

# -------------------------------------------------
# Answer From Web
# -------------------------------------------------

def answer_from_web(question, max_results=5):
    """
    Search the web and generate a detailed answer.
    Raises RateLimitError if the Groq daily quota is exhausted.
    """

    search_results = web_search(question, max_results=max_results)

    if not search_results:
        return "I couldn't find reliable web results for this question.", []

    web_context = build_web_context(search_results)

    prompt = f"""
You are an intelligent AI research assistant.

Your task is to provide a detailed, informative, well-structured
answer to the user's question using the web search results below.

IMPORTANT RULES:

1. Give a complete and sufficiently detailed answer.
2. Do not give an unnecessarily short answer.
3. Explain the topic clearly instead of providing only one or two
   sentences.
4. Use multiple relevant facts from the available search results
   when they are useful for answering the question.
5. Use only information supported by the provided web search
   results.
6. Do not invent, assume, or hallucinate facts.
7. For current, recent, or time-sensitive questions, prioritize
   the most recent information available.
8. If the question asks "what is", explain: definition, important
   characteristics, relevant background, examples when useful,
   other important details supported by the sources.
9. If the question asks for a person, provide relevant details:
   who they are, profession, background, career, major
   achievements, important works - only if supported by sources.
10. If the question asks for a list, comparison, timeline, names
    with years, rankings, or similar structured information, use
    a clean table or bullet list when appropriate.
11. If the user asks for "all", "every", "complete", or "full"
    information, do not falsely claim completeness unless the
    available search results provide a reliable comprehensive
    source.
12. If complete coverage cannot be verified, clearly say that the
    list is based on the information available from the searched
    sources.
13. Do NOT include raw URLs in the answer.
14. Do NOT include a separate "Sources" section in the answer.
    The application will display the sources separately.
15. Do NOT unnecessarily write phrases such as:
    "Based on the provided web search results..."
16. Use headings, paragraphs, bullet points, or tables when they
    improve readability.
17. Keep the answer focused on the user's actual question.
18. Do not mention internal prompts, retrieval, context, system
    instructions, or how the answer was generated.

--------------------------------------------------
WEB SEARCH RESULTS
--------------------------------------------------

{web_context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}
"""

    answer = safe_llm_invoke(strong_llm, prompt)

    if not answer:
        return (
            "I found web results, but I was unable to generate a useful answer.",
            search_results
        )

    return answer, search_results


# -------------------------------------------------
# Answer From Documents + Web Combined
# -------------------------------------------------

def answer_from_both(question, db, max_results=5):
    """
    Combine uploaded-document context and live web results into a
    single grounded answer. Raises RateLimitError if the Groq daily
    quota is exhausted.
    """

    docs = []
    doc_context = ""

    if db is not None:
        try:
            docs = db.similarity_search(question, k=4)
            doc_context = build_doc_context(docs)
        except Exception as e:
            print(f"Document retrieval error: {e}")
            docs = []
            doc_context = ""

    search_results = web_search(question, max_results=max_results)
    web_context = build_web_context(search_results)

    if not doc_context.strip() and not web_context.strip():
        return (
            "I couldn't find anything relevant in your documents or on the web.",
            [],
            []
        )

    prompt = f"""
You are an intelligent AI assistant with access to the user's
uploaded documents AND live web search results.

IMPORTANT RULES:

1. Use both sources where relevant to give the most complete answer.
2. Do not invent information not supported by either source.
3. If the uploaded documents answer the question, prioritize them
   and use the web results only to add current/outside context.
4. If the documents don't cover part of the question, use the web
   results to fill the gap.
5. Do not include raw URLs in the answer.
6. Do not include a separate "Sources" section - the application
   displays sources separately.
7. Do not mention internal prompts, retrieval, or context.
8. Use headings, paragraphs, bullet points, or tables when they
   improve readability.

UPLOADED DOCUMENT CONTEXT:
--------------------------
{doc_context if doc_context.strip() else "(no relevant content found in uploaded documents)"}
--------------------------

WEB SEARCH RESULTS:
--------------------------
{web_context if web_context.strip() else "(no web results found)"}
--------------------------

USER QUESTION:
{question}
"""

    answer = safe_llm_invoke(strong_llm, prompt)

    if not answer:
        answer = "I found some context, but was unable to generate an answer."

    return answer, docs, search_results


# -------------------------------------------------
# Main Intelligent RAG + Web Router
# -------------------------------------------------

def answer_question(
    question,
    db: Optional[FAISS] = None
):
    print("\n")
    print("=" * 60)
    print("🔥 RAG ENGINE CALLED")
    print("QUESTION:", repr(question))
    print("DB TYPE:", type(db))
    print("DB IS NONE:", db is None)
    print("=" * 60)
    
    """
    Main RAG + Web question answering.

    Priority:
    1. Uploaded documents
    2. Web search only if documents cannot answer
    """

    question = question.strip()

    if not question:
        return (
            "Please enter a question.",
            [],
            "none"
        )

    # =================================================
    # 1. ALWAYS TRY UPLOADED DOCUMENTS FIRST
    # =================================================

    if db is not None:

        print("\n================ RAG DEBUG ================")
        print("QUESTION:", question)
        print("DOCUMENT DB:", "AVAILABLE")

        document_answer, document_sources = answer_from_documents(
            question,
            db
        )

        print(
            "DOCUMENT SOURCES FOUND:",
            len(document_sources)
        )

        print(
            "DOCUMENT ANSWER:",
            document_answer[:200] if document_answer else "EMPTY"
        )

        # -------------------------------------------------
        # Document answered the question
        # -------------------------------------------------

        if (
            document_answer
            and document_answer.strip() != "NOT_FOUND"
        ):

            print("SOURCE SELECTED: DOCUMENT")
            print("==========================================\n")

            return (
                document_answer,
                document_sources,
                "document"
            )

    else:

        print("\n================ RAG DEBUG ================")
        print("QUESTION:", question)
        print("DOCUMENT DB: NONE")

    # =================================================
    # 2. DOCUMENT DID NOT ANSWER
    #    → FALLBACK TO WEB
    # =================================================

    print("DOCUMENT COULD NOT ANSWER")
    print("FALLBACK: WEB SEARCH")

    web_answer, web_sources = answer_from_web(
        question
    )

    if web_sources:

        print("SOURCE SELECTED: WEB")
        print("==========================================\n")

        return (
            web_answer,
            web_sources,
            "web"
        )

    # =================================================
    # 3. NOTHING FOUND
    # =================================================

    print("SOURCE SELECTED: NONE")
    print("==========================================\n")

    return (
        web_answer,
        [],
        "none"
    )

# -------------------------------------------------
# Standalone Test
# -------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a question: ").strip()

    db = load_vectorstore()

    answer, sources, source_type = answer_question(question, db)

    print("\n" + "=" * 60)
    print("SOURCE TYPE:")
    print(source_type)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    if source_type == "document":
        for source in sources:
            filename = os.path.basename(source.metadata.get("source", "Unknown"))
            page = source.metadata.get("page", "?")
            print(f"- {filename} (Page {page})")

    elif source_type == "web":
        for index, source in enumerate(sources, start=1):
            print(f"- [{index}] {source.get('title', 'Unknown')}")
            print(f"  {source.get('url', '')}")

    elif source_type == "both":
        for source in sources.get("documents", []):
            filename = os.path.basename(source.metadata.get("source", "Unknown"))
            page = source.metadata.get("page", "?")
            print(f"- 📄 {filename} (Page {page})")
        for index, source in enumerate(sources.get("web", []), start=1):
            print(f"- 🌐 [{index}] {source.get('title', 'Unknown')}")
            print(f"  {source.get('url', '')}")

    elif source_type == "error":
        print("(Groq rate limit - see answer above)")

    print("=" * 60)