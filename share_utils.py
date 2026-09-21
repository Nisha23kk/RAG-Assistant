import html
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)


def conversation_to_text(messages):
    lines = []

    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg.get("content", ""))

        lines.append(f"{role}: {content}")

    return "\n\n".join(lines)


def conversation_to_pdf(messages):
    """
    Generate conversation PDF completely in memory.

    No PDF file is created in the project folder.
    Returns PDF bytes for Streamlit download.
    """

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    for msg in messages:

        role = "User" if msg["role"] == "user" else "Assistant"

        content = str(msg.get("content", ""))

        # Prevent ReportLab HTML/XML parsing errors
        content = html.escape(content)

        # Preserve line breaks
        content = content.replace("\n", "<br/>")

        story.append(
            Paragraph(
                f"<b>{html.escape(role)}:</b>",
                styles["Heading3"]
            )
        )

        if content.strip():
            story.append(
                Paragraph(
                    content,
                    styles["Normal"]
                )
            )

        story.append(
            Spacer(1, 12)
        )

    doc.build(story)

    pdf_buffer.seek(0)

    pdf_data = pdf_buffer.getvalue()

    pdf_buffer.close()

    return pdf_data