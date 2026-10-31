
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


def read_pdf(file_path: str):
    """Load a single PDF and display its content and metadata."""

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        print(doc.page_content)
        print(doc.metadata)

    return documents


def read_multiple_pdfs(directory: str):
    """Load multiple PDFs and display their content and metadata."""

    all_documents = []

    for pdf_file in Path(directory).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        for doc in documents:
            print(doc.page_content)
            print(doc.metadata)

        all_documents.extend(documents)

    return all_documents
