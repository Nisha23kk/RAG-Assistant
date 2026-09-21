import os

from rag_engine import (
    answer_question,
    load_vectorstore
)


# -------------------------------------------------
# Load Vector Database
# -------------------------------------------------

db = load_vectorstore()


# -------------------------------------------------
# Main Program
# -------------------------------------------------

def main():

    print("=" * 60)
    print("        AI RAG + WEB SEARCH ASSISTANT")
    print("=" * 60)

    if db is not None:

        print(
            "\nUploaded document database detected."
        )

        print(
            "The assistant will search uploaded "
            "documents first."
        )

        print(
            "If the answer is not found, "
            "web search will be used."
        )

    else:

        print(
            "\nNo uploaded document database found."
        )

        print(
            "The assistant will use web search."
        )

    print(
        "\nType 'exit' to quit."
    )

    # -------------------------------------------------
    # Continuous Question Loop
    # -------------------------------------------------

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in (
            "exit",
            "quit",
            "q"
        ):

            print(
                "\nGoodbye!"
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        # -------------------------------------------------
        # Ask Assistant
        # -------------------------------------------------

        try:

            answer, sources, source_type = answer_question(
                question,
                db
            )

            print(
                "\nAssistant:"
            )

            print(
                answer
            )

            # -------------------------------------------------
            # Document Sources
            # -------------------------------------------------

            if source_type == "document":

                print(
                    "\n📄 Document Sources:"
                )

                seen_sources = set()

                for source in sources:

                    filename = os.path.basename(
                        source.metadata.get(
                            "source",
                            "Unknown"
                        )
                    )

                    page = source.metadata.get(
                        "page",
                        "?"
                    )

                    source_key = (
                        filename,
                        page
                    )

                    if source_key not in seen_sources:

                        seen_sources.add(
                            source_key
                        )

                        print(
                            f"- {filename} "
                            f"(Page {page})"
                        )

            # -------------------------------------------------
            # Web Sources
            # -------------------------------------------------

            elif source_type == "web":

                print(
                    "\n🌐 Web Sources:"
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

                    print(
                        f"[{index}] {title}"
                    )

                    if url:

                        print(
                            f"    {url}"
                        )

            # -------------------------------------------------
            # No Sources
            # -------------------------------------------------

            else:

                print(
                    "\nNo sources available."
                )

        except Exception as e:

            print(
                "\nError:"
            )

            print(
                str(e)
            )


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":

    main()