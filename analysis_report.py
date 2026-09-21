import os
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# =================================================
# Load Environment Variables
# =================================================

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

fast_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1200,
)

strong_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1500,
)


class RateLimitError(Exception):
    pass


def safe_llm_invoke(llm, prompt, retries=2, base_delay=2):

    last_error = None

    for attempt in range(retries + 1):

        try:
            response = llm.invoke(prompt)
            return response.content.strip()

        except Exception as e:

            error_text = str(e)
            last_error = e

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            )

            if is_rate_limit:
                # No point retrying - the wait time is usually long.
                wait_match = re.search(
                    r"try again in ([0-9a-zA-Z.]+)",
                    error_text
                )
                wait_hint = (
                    f" (Groq says: retry in {wait_match.group(1)})"
                    if wait_match else ""
                )
                raise RateLimitError(
                    "Groq daily token limit reached for this model."
                    f"{wait_hint} Try again later, switch to a "
                    "smaller model, or upgrade your Groq tier."
                ) from e

            # Transient/network error - short retry
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
                continue

    raise last_error


# =================================================
# Supported Document Types
# =================================================

VALID_DOCUMENT_TYPES = [
    "Research Paper",
    "Resume / CV",
    "Project Report",
    "Technical Documentation",
    "Invoice / Financial Document",
    "Dataset / CSV / Excel",
    "Presentation / PPT",
    "Academic Notes / Textbook",
    "Legal Document",
    "Business Document",
    "Certificate",
    "User Manual",
    "General Document"
]


# =================================================
# Chunk Settings
# =================================================
# FIX: Bigger chunks = fewer LLM calls = less token overhead spent on
# repeated prompt boilerplate, and fewer total requests => lower
# chance of hitting per-minute / per-day rate limits.

CHUNK_SIZE = 15000
CHUNK_OVERLAP = 400


# =================================================
# Split Text Into Chunks
# =================================================

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if not text:
        return []

    text = text.strip()

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if end < text_length:

            last_newline = chunk.rfind("\n")

            if last_newline > chunk_size * 0.7:
                end = start + last_newline
                chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# =================================================
# Detect Document Type
# =================================================

def detect_document_type(text):

    prompt = f"""
You are an intelligent document classification system.

Identify the type of the provided document.

Choose EXACTLY ONE category:

1. Research Paper
2. Resume / CV
3. Project Report
4. Technical Documentation
5. Invoice / Financial Document
6. Dataset / CSV / Excel
7. Presentation / PPT
8. Academic Notes / Textbook
9. Legal Document
10. Business Document
11. Certificate
12. User Manual
13. General Document

Classification guidelines:

Research Paper: Abstract, Introduction, Methodology, Experiments, Results, Discussion, References
Resume / CV: Name, Contact information, Education, Skills, Experience, Projects, Certifications
Project Report: Objectives, Requirements, Architecture, Implementation, Technologies, Modules, Testing, Results, Future scope
Technical Documentation: APIs, Technical specifications, Configuration, Architecture, Developer documentation
Invoice / Financial Document: Invoice, Billing, Items, Quantity, Price, Tax, Total amount, Financial transactions
Dataset / CSV / Excel: Rows, Columns, Records, Variables, Tabular data
Presentation / PPT: Slides, Slide titles, Bullet points, Presentation topics
Academic Notes / Textbook: Educational explanations, Definitions, Concepts, Chapters, Examples, Study material
Legal Document: Contracts, Agreements, Clauses, Terms, Legal provisions
Business Document: Business plans, Proposals, Company reports, Business strategies
Certificate: Certificate title, Recipient, Issuing organization, Date, Certification statement
User Manual: Installation, Setup, Usage instructions, Troubleshooting, Product instructions
General Document: Use only when no other category clearly fits.

Return ONLY the category name.

DOCUMENT CONTENT:

{text[:6000]}
"""

    try:

        document_type = safe_llm_invoke(fast_llm, prompt)

        for valid_type in VALID_DOCUMENT_TYPES:

            if valid_type.lower() in document_type.lower():
                return valid_type

        return "General Document"

    except RateLimitError:
        raise

    except Exception:
        return "General Document"


# =================================================
# Analyze Individual Chunk
# =================================================

def analyze_chunk(chunk, chunk_number, total_chunks, document_type):

    prompt = f"""
You are analyzing part {chunk_number} of {total_chunks}
of a document.

Document Type:
{document_type}

Analyze ONLY the provided section.

Do NOT invent information.

Extract the most important information from this section.

Focus on:
- Important concepts, Definitions, Topics, Names, Dates, Numbers
- Formulas, Technologies, Organizations, Results, Examples
- Important statements, Conclusions, Technical details

If something is not present, do not invent it.

Create a concise structured summary that can later be
combined with summaries from other sections.

DOCUMENT SECTION:

{chunk}
"""

    try:
        return safe_llm_invoke(fast_llm, prompt)

    except RateLimitError:
        raise

    except Exception as e:
        return f"Unable to analyze section {chunk_number}: {str(e)}"


# =================================================
# Generate Final Report
# =================================================

def generate_final_report(summaries, filename, document_type):

    combined_summaries = "\n\n".join(
        f"===== SECTION {i + 1} SUMMARY =====\n{summary}"
        for i, summary in enumerate(summaries)
    )

    prompt = f"""
You are an advanced AI document analysis assistant.

Generate a final analysis report for ONE document.

==================================================
DOCUMENT INFORMATION
==================================================

Filename:
{filename}

Document Type:
{document_type}


==================================================
IMPORTANT RULES
==================================================

Use ONLY the information contained in the
section summaries below.

Do NOT invent facts.
Do NOT assume missing information.
Do NOT change numerical values.
Do NOT change names.

If information is unavailable, write:
"Not specified in the document."

The report must be based on the actual document type.

Create approximately 6-10 useful sections.

Use Markdown with headings, subheadings, bullet points,
numbered lists, tables when useful, important numbers,
names, dates, technologies, organizations, results and
technical details.


==================================================
DOCUMENT-TYPE ANALYSIS
==================================================

For Academic Notes / Textbook, focus on: Subject/topic, Main concepts, Definitions,
Important theories, Examples, Key points, Important formulas, Learning summary

For Research Paper, focus on: Research objective, Problem statement, Methodology,
Dataset/materials, Experiments, Results, Findings, Limitations, Conclusion

For Resume / CV, focus on: Candidate overview, Education, Skills, Experience,
Projects, Certifications, Achievements, Technologies

For Project Report, focus on: Project overview, Problem statement, Objectives,
Requirements, Technologies, Architecture, Modules, Implementation, Testing,
Results, Future scope

For Technical Documentation, focus on: Purpose, Architecture, Technical
specifications, APIs, Configuration, Dependencies, Procedures, Troubleshooting

For Legal Document, focus on: Purpose, Parties, Clauses, Obligations, Rights,
Terms, Conditions, Important provisions

For Business Document, focus on: Business purpose, Objectives, Strategy,
Products/services, Financial information, Risks, Opportunities, Conclusions

For User Manual, focus on: Product/system overview, Requirements, Installation,
Configuration, Usage, Procedures, Troubleshooting, Safety, Maintenance

For Certificate, focus on: Certificate type, Recipient, Issuing organization,
Achievement, Date, Validity information

For Invoice / Financial Document, focus on: Parties, Invoice details,
Items/services, Quantity, Prices, Taxes, Discounts, Total, Payment information

For Dataset / CSV / Excel, focus on: Dataset overview, Records, Columns,
Variables, Data types, Categories, Numerical information, Patterns,
Potential use cases

For Presentation / PPT, focus on: Topic, Objective, Main themes, Important
concepts, Key messages, Data/statistics, Conclusions


==================================================
OUTPUT FORMAT
==================================================

Start EXACTLY with:

# Document Analysis Report

## Document Information

| Field | Details |
|---|---|
| Filename | {filename} |
| Document Type | {document_type} |

Then write:

## Overview

Provide a clear overview of the document.

Then create the most relevant sections based on the document type.

Finally end with:

## Overall Summary

Explain what the document is about, its main purpose, most important
information, major findings, and overall conclusion.


==================================================
SECTION SUMMARIES
==================================================

{combined_summaries}

==================================================
END
==================================================
"""

    try:

        report = safe_llm_invoke(strong_llm, prompt)

        if not report:
            return (
                f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
                "The AI was unable to generate an analysis report."
            )

        return report

    except RateLimitError as e:

        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"## Document Type\n\n{document_type}\n\n"
            f"### Rate limit reached\n\n{str(e)}"
        )

    except Exception as e:

        return (
            f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n"
            f"## Document Type\n\n{document_type}\n\n"
            f"Unable to generate the final analysis report.\n\n"
            f"### Error\n\n{str(e)}"
        )


# =================================================
# Generate Dynamic Document Analysis Report
# =================================================

def generate_analysis_report(text, filename="Unknown Document"):
    """
    Generate a detailed AI analysis report for ONE document.

    PDF -> Chunking -> Individual chunk analysis (fast model)
         -> Summary aggregation -> Final report (strong model)

    FIX: for short documents that fit in a single chunk, the
    intermediate "analyze this chunk" call is skipped entirely and
    the raw text is fed straight to the final report step - saving
    one full LLM call per small document.
    """

    if not text or not text.strip():
        return f"# Document Analysis Report\n\n## Document\n\n{filename}\n\nNo readable text was found in this document."

    try:
        document_type = detect_document_type(text)
    except RateLimitError as e:
        return f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n### Rate limit reached\n\n{str(e)}"

    chunks = split_text_into_chunks(text)

    if not chunks:
        return f"# Document Analysis Report\n\n## Document\n\n{filename}\n\nNo readable text was found in this document."

    total_chunks = len(chunks)

    try:

        if total_chunks == 1:
            # FIX: single-chunk shortcut - skip the intermediate summary call
            summaries = [chunks[0][:6000]]
        else:
            summaries = []

            for index, chunk in enumerate(chunks):

                print(f"Analyzing document section {index + 1}/{total_chunks}...")

                summary = analyze_chunk(
                    chunk=chunk,
                    chunk_number=index + 1,
                    total_chunks=total_chunks,
                    document_type=document_type
                )

                summaries.append(summary)

                # FIX: small delay only when there are multiple chunks left,
                # and only 0.3s (fast model has much higher rate limits)
                if index < total_chunks - 1:
                    time.sleep(0.3)

    except RateLimitError as e:
        return f"# Document Analysis Report\n\n## Document\n\n{filename}\n\n## Document Type\n\n{document_type}\n\n### Rate limit reached\n\n{str(e)}"

    print("Generating final document analysis report...")

    return generate_final_report(
        summaries=summaries,
        filename=filename,
        document_type=document_type
    )
