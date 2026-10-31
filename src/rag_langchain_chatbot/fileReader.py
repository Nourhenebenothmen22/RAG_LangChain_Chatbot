
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


def _load_pdf(file_path: Path):
    """Load a PDF and display its content and metadata."""
    try:
        documents = PyPDFLoader(str(file_path)).load()

        for doc in documents:
            print(doc.page_content)
            print(doc.metadata)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading PDF '{file_path}': {e}") from e


def read_pdf(file_path: str):
    """Load a single PDF."""

    pdf_path = Path(file_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF format: {file_path}")

    return _load_pdf(pdf_path)


def read_multiple_pdfs(directory: str):
    """Load multiple PDFs from a directory."""

    pdf_directory = Path(directory)

    if not pdf_directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    pdf_files = list(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    all_documents = []

    for pdf_file in pdf_files:
        all_documents.extend(_load_pdf(pdf_file))

    return all_documents
