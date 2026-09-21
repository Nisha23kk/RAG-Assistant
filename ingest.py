import os
import tempfile
import shutil
from functools import lru_cache
import torch
from langchain_core.documents import Document
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, CSVLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------------------------
# Supported File Types
# -------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".xls",
    ".csv",
    ".docx",
    ".doc",
    ".txt",
    ".pptx",
}


@lru_cache(maxsize=1)
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using embedding device: {device}")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )


# -------------------------------------------------
# Load Excel
# -------------------------------------------------
def load_excel(file_path):
    """
    Load Excel workbook and convert each sheet
    into a LangChain Document.
    """

    import pandas as pd

    documents = []

    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        if df.empty:
            continue

        text = df.to_csv(index=False)

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                        "file_type": "excel",
                        "sheet": sheet_name,
                    }
                )
            )

    return documents


# -------------------------------------------------
# Load DOCX
# -------------------------------------------------

def load_docx(file_path):
    """
    Load Microsoft Word .docx files.
    """

    from docx import Document as DocxDocument

    documents = []

    doc = DocxDocument(file_path)

    paragraphs = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    # Also read tables
    for table in doc.tables:

        for row in table.rows:

            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
            )

            if row_text.strip():
                paragraphs.append(row_text)

    full_text = "\n".join(paragraphs)

    if full_text.strip():

        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "file_type": "docx",
                }
            )
        )

    return documents


# -------------------------------------------------
# Load DOC
# -------------------------------------------------
def load_doc(file_path):
    """
    Load old Microsoft Word .doc files.

    Requires LibreOffice to be installed.
    """

    import subprocess

    temp_dir = tempfile.mkdtemp()

    try:

        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                temp_dir,
                file_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        converted_file = os.path.join(
            temp_dir,
            os.path.splitext(os.path.basename(file_path))[0] + ".docx"
        )
        if not os.path.exists(converted_file):
            raise FileNotFoundError(
                "LibreOffice could not convert the DOC file."
            )
        documents = load_docx(converted_file)
        # Preserve original filename
        for doc in documents:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["file_type"] = "doc"
        return documents
    except FileNotFoundError:

        raise RuntimeError(
            "LibreOffice is not installed or 'soffice' "
            "is not available in PATH."
        )

    except Exception as e:

        raise RuntimeError(f"Could not process DOC file: {e}")

    finally:

        shutil.rmtree(temp_dir, ignore_errors=True)


# -------------------------------------------------
# Load PPTX
# -------------------------------------------------

def load_pptx(file_path):
    """
    Load PowerPoint .pptx files.
    Each slide is converted into a LangChain Document.
    """

    from pptx import Presentation

    documents = []

    presentation = Presentation(file_path)

    for slide_number, slide in enumerate(presentation.slides, start=1):

        slide_text = []

        for shape in slide.shapes:

            if hasattr(shape, "text"):

                text = shape.text.strip()

                if text:
                    slide_text.append(text)

        content = "\n".join(slide_text)

        if content.strip():

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                        "file_type": "pptx",
                        "slide": slide_number,
                    }
                )
            )

    return documents


# -------------------------------------------------
# Load TXT
# -------------------------------------------------

def load_txt(file_path):
    """
    Load plain text files.
    """

    loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    for doc in documents:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "txt"

    return documents


# -------------------------------------------------
# Load CSV
# -------------------------------------------------

def load_csv(file_path):
    """
    Load CSV files.
    """

    loader = CSVLoader(file_path, encoding="utf-8")

    documents = loader.load()

    for doc in documents:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "csv"

    return documents


# -------------------------------------------------
# Load Any Supported File
# -------------------------------------------------

def load_single_file(file_path):
    """
    Detect file type and load the file.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        for doc in documents:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["file_type"] = "pdf"

        return documents

    elif extension in [".xlsx", ".xls"]:
        return load_excel(file_path)

    elif extension == ".csv":
        return load_csv(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".doc":
        return load_doc(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    elif extension == ".pptx":
        return load_pptx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")


# -------------------------------------------------
# Load Multiple Files
# -------------------------------------------------

def load_documents(folder_path):
    """
    Load all supported files from a folder.
    """

    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        extension = os.path.splitext(filename)[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        try:

            file_documents = load_single_file(file_path)
            documents.extend(file_documents)
            print(f"Loaded: {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


# -------------------------------------------------
# Split Documents
# -------------------------------------------------

def split_documents(documents):
    # FIX: slightly larger chunk size => fewer chunks => faster
    # embedding + fewer FAISS entries, without hurting retrieval quality.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# -------------------------------------------------
# Create FAISS Vector Store
# -------------------------------------------------

def create_vectorstore(chunks):

    embeddings = get_embeddings()  # FIX: cached, not re-loaded every time

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vectorstore")

    print("Vector database created successfully!")

    return db


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    docs = load_documents("data/papers")

    print(f"Loaded {len(docs)} document sections")

    if len(docs) == 0:
        print("No supported documents were found.")
        exit()

    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    if len(chunks) == 0:
        print("No chunks were created.")
        exit()

    create_vectorstore(chunks)
