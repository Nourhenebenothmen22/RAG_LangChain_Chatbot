"""Document loading adapter maintaining backwards compatibility while leveraging modular loaders."""

import logging
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from rag_langchain_chatbot.loaders import PDFLoader, get_loader_registry

logger = logging.getLogger(__name__)


def read_pdf(file_path: Union[str, Path]) -> List[Document]:
    """Load a single PDF document.
    
    Backwards-compatible wrapper delegating to PDFLoader.
    """
    loader = PDFLoader()
    return loader.load(file_path)


def read_multiple_pdfs(directory: Union[str, Path]) -> List[Document]:
    """Load all PDF documents from a specified directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    loader = PDFLoader()
    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        all_documents.extend(loader.load(pdf_file))

    return all_documents


def read_all_documents(directory: Union[str, Path]) -> List[Document]:
    """Load all eligible documents across all supported formats (PDF, DOCX, TXT, CSV, JSON, DB)."""
    registry = get_loader_registry()
    sources = registry.discover_sources(directory)
    if not sources:
        logger.warning("No eligible documents discovered in: %s", directory)
        return []

    documents: List[Document] = []
    for src in sources:
        try:
            docs = registry.load_source(src)
            documents.extend(docs)
        except Exception as e:
            logger.error("Error loading source '%s': %s", src, e)

    return documents
