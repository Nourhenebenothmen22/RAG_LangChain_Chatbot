"""Modular ingestion loaders for diverse data sources and file formats.

Supports:
- PDF documents (with page metadata)
- DOCX documents (paragraphs and tables)
- Plain text & Markdown (.txt, .md, .markdown)
- Structured database records (SQLite tables, CSV, JSON)
- Dynamic source discovery and SHA-256 content hashing
"""

import csv
import hashlib
import json
import logging
import re
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file for change detection and versioning."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute SHA-256 checksum of an in-memory string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BaseDocumentLoader(ABC):
    """Abstract base loader for document and record ingestion."""

    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Document]:
        """Load documents from source and return LangChain Document instances."""
        pass


def _clean_page_text(raw_text: str) -> str:
    """Clean and normalize extracted page text, removing page numbers and line break artifacts."""
    if not raw_text:
        return ""
    # Normalize non-breaking spaces and carriage returns
    text = raw_text.replace("\xa0", " ").replace("\r", "")
    # Fix soft hyphenation across line breaks (e.g. "responsi-\nbilité" -> "responsibilité")
    text = re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', text)
    # Remove solitary page number headers/footers (e.g. "Page 1 / 10", "Page 2", "1 / 5")
    text = re.sub(r'^\s*(?:Page\s+)?\d+(?:\s*(?:/|sur|of|-)\s*\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class PDFLoader(BaseDocumentLoader):
    """Loads PDF files using PyPDFLoader with layout cleaning and metadata extraction."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        file_path = Path(source)
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {source}")

        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        file_hash = compute_file_sha256(file_path)
        stat = file_path.stat()

        for doc in docs:
            doc.page_content = _clean_page_text(doc.page_content)
            doc.metadata["source"] = str(file_path.resolve())
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "pdf"
            doc.metadata["file_size_bytes"] = stat.st_size
            doc.metadata["source_hash"] = file_hash
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0

        logger.info("Loaded %d pages from PDF: %s", len(docs), file_path.name)
        return docs


class DocxLoader(BaseDocumentLoader):
    """Loads DOCX documents by parsing document.xml inside the zip package.
    
    Extracts structured headings, paragraphs, and tables formatted as clean Markdown
    without requiring external C-dependencies.
    """

    def load(self, source: Union[str, Path]) -> List[Document]:
        file_path = Path(source)
        if not file_path.is_file():
            raise FileNotFoundError(f"DOCX file not found: {source}")

        file_hash = compute_file_sha256(file_path)
        stat = file_path.stat()

        try:
            with zipfile.ZipFile(file_path) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
            
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            
            body = tree.find(".//w:body", ns)
            elements = list(body) if body is not None else tree.findall(".//w:p", ns)

            blocks: List[str] = []
            for elem in elements:
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "p":
                    p_pr = elem.find("w:pPr", ns)
                    style_val = ""
                    if p_pr is not None:
                        p_style = p_pr.find("w:pStyle", ns)
                        if p_style is not None:
                            style_val = p_style.attrib.get(f"{{{ns['w']}}}val", "") or p_style.attrib.get("val", "")

                    texts = [t.text for t in elem.findall(".//w:t", ns) if t.text]
                    p_text = "".join(texts).strip()
                    if p_text:
                        p_text = re.sub(r'[\r\t\xa0]+', ' ', p_text)
                        lower_style = style_val.lower()
                        if "heading1" in lower_style or "titre1" in lower_style or lower_style == "title":
                            p_text = f"# {p_text}"
                        elif "heading2" in lower_style or "titre2" in lower_style:
                            p_text = f"## {p_text}"
                        elif "heading3" in lower_style or "titre3" in lower_style:
                            p_text = f"### {p_text}"
                        blocks.append(p_text)

                elif tag == "tbl":
                    table_rows: List[List[str]] = []
                    for tr in elem.findall(".//w:tr", ns):
                        cells: List[str] = []
                        for tc in tr.findall(".//w:tc", ns):
                            cell_texts = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                            cleaned_cell = " ".join("".join(cell_texts).split()).replace("|", "\\|")
                            cells.append(cleaned_cell)
                        if any(cells):
                            table_rows.append(cells)

                    if table_rows:
                        max_cols = max(len(r) for r in table_rows)
                        norm_rows = [r + [""] * (max_cols - len(r)) for r in table_rows]
                        header = "| " + " | ".join(norm_rows[0]) + " |"
                        sep = "| " + " | ".join(["---"] * max_cols) + " |"
                        body_rows = ["| " + " | ".join(r) + " |" for r in norm_rows[1:]]
                        tbl_md = "\n".join([header, sep] + body_rows)
                        blocks.append(tbl_md)

            # Fallback if body iteration was empty
            if not blocks:
                for p_elem in tree.findall(".//w:p", ns):
                    texts = [t.text for t in p_elem.findall(".//w:t", ns) if t.text]
                    if texts:
                        blocks.append("".join(texts).strip())

            full_text = "\n\n".join(blocks).strip()
            if not full_text:
                full_text = f"[Empty DOCX document: {file_path.name}]"

            doc = Document(
                page_content=full_text,
                metadata={
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_type": "docx",
                    "file_size_bytes": stat.st_size,
                    "source_hash": file_hash,
                    "page": 0,
                }
            )
            logger.info("Loaded DOCX document: %s (%d chars)", file_path.name, len(full_text))
            return [doc]

        except Exception as e:
            logger.error("Failed to load DOCX '%s': %s", file_path, e)
            raise RuntimeError(f"Error loading DOCX '{file_path}': {e}") from e


class TextLoader(BaseDocumentLoader):
    """Loads plain text, Markdown (.md), and plain text documents."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        file_path = Path(source)
        if not file_path.is_file():
            raise FileNotFoundError(f"Text file not found: {source}")

        file_hash = compute_file_sha256(file_path)
        stat = file_path.stat()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path.resolve()),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lstrip(".").lower() or "text",
                "file_size_bytes": stat.st_size,
                "source_hash": file_hash,
                "page": 0,
            }
        )
        logger.info("Loaded text document: %s (%d chars)", file_path.name, len(content))
        return [doc]


class StructuredDatabaseLoader(BaseDocumentLoader):
    """Loads structured relational records from a SQLite database or tabular file.
    
    Transforms rows into semantic key-value documents while preserving column metadata.
    """

    def __init__(self, table_name: Optional[str] = None, text_columns: Optional[List[str]] = None):
        self.table_name = table_name
        self.text_columns = text_columns

    def load(self, source: Union[str, Path]) -> List[Document]:
        """Load rows from a SQLite database file or table query."""
        path_str = str(source)
        db_path = Path(path_str.replace("sqlite:///", "").replace("sqlite://", ""))

        if not db_path.is_file():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        documents: List[Document] = []

        try:
            cursor = conn.cursor()
            # Discover all valid user tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            valid_tables = {row["name"] for row in cursor.fetchall()}

            if self.table_name:
                if self.table_name not in valid_tables:
                    raise ValueError(f"Table '{self.table_name}' does not exist in database {db_path.name}")
                tables = [self.table_name]
            else:
                tables = sorted(list(valid_tables))

            for tbl in tables:
                # Enforce strict identifier whitelist against SQL injection
                if not re.match(r"^[a-zA-Z0-9_]+$", tbl) or tbl not in valid_tables:
                    logger.warning("Rejected unsafe table name during ingestion: %s", tbl)
                    continue

                safe_tbl = f'"{tbl}"'
                cursor.execute(f"PRAGMA table_info({safe_tbl});")
                all_columns = [c["name"] for c in cursor.fetchall()]
                columns = [c for c in all_columns if not self.text_columns or c in self.text_columns] or all_columns

                cursor.execute(f"SELECT * FROM {safe_tbl};")
                rows = cursor.fetchall()

                for row_idx, row in enumerate(rows):
                    row_dict = dict(row)
                    row_hash = compute_text_sha256(json.dumps(row_dict, sort_keys=True, default=str))

                    # Format human-readable semantic block
                    content_parts = []
                    for col in columns:
                        val = row_dict.get(col)
                        if val is not None and str(val).strip():
                            content_parts.append(f"{col}: {val}")

                    row_content = f"Table: {tbl}\n" + "\n".join(content_parts)
                    doc_id = row_dict.get("id") or row_dict.get("rowid") or row_idx

                    doc = Document(
                        page_content=row_content,
                        metadata={
                            "source": f"{db_path.name}#{tbl}",
                            "source_type": "database_record",
                            "table_name": tbl,
                            "record_id": str(doc_id),
                            "file_name": db_path.name,
                            "source_hash": row_hash,
                            "page": row_idx,
                        }
                    )
                    documents.append(doc)

            logger.info("Loaded %d records across %d table(s) from database %s", len(documents), len(tables), db_path.name)
            return documents

        finally:
            conn.close()


class TabularFileLoader(BaseDocumentLoader):
    """Loads CSV or JSON data records into structured Document chunks."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        file_path = Path(source)
        if not file_path.is_file():
            raise FileNotFoundError(f"Tabular file not found: {source}")

        file_hash = compute_file_sha256(file_path)
        stat = file_path.stat()
        documents: List[Document] = []

        if file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_path.name,
                            "file_type": "csv_record",
                            "record_id": str(idx),
                            "file_size_bytes": stat.st_size,
                            "source_hash": file_hash,
                            "page": idx,
                        }
                    ))
        elif file_path.suffix.lower() in (".json", ".jsonl"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read().strip()
                if raw.startswith("["):
                    records = json.loads(raw)
                else:
                    records = [json.loads(line) for line in raw.split("\n") if line.strip()]

                for idx, rec in enumerate(records):
                    if isinstance(rec, dict):
                        content = "\n".join(f"{k}: {v}" for k, v in rec.items() if v is not None)
                    else:
                        content = str(rec)

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_path.name,
                            "file_type": "json_record",
                            "record_id": str(idx),
                            "file_size_bytes": stat.st_size,
                            "source_hash": file_hash,
                            "page": idx,
                        }
                    ))

        logger.info("Loaded %d tabular records from %s", len(documents), file_path.name)
        return documents


class LoaderRegistry:
    """Dynamic source registry that automatically selects loaders by file type or URI."""

    def __init__(self):
        self._loaders: Dict[str, BaseDocumentLoader] = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".markdown": TextLoader(),
            ".csv": TabularFileLoader(),
            ".json": TabularFileLoader(),
            ".jsonl": TabularFileLoader(),
            ".sqlite": StructuredDatabaseLoader(),
            ".sqlite3": StructuredDatabaseLoader(),
            ".db": StructuredDatabaseLoader(),
        }

    def register_loader(self, extension_or_scheme: str, loader: BaseDocumentLoader) -> None:
        """Register a custom loader for an extension or URI scheme."""
        key = extension_or_scheme.lower()
        if not key.startswith(".") and not key.startswith("http"):
            key = f".{key}"
        self._loaders[key] = loader

    def get_loader_for_path(self, path: Union[str, Path]) -> Optional[BaseDocumentLoader]:
        """Find appropriate loader for given path or URI."""
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        if ext in self._loaders:
            return self._loaders[ext]
        return None

    def load_source(self, path: Union[str, Path]) -> List[Document]:
        """Load documents from any recognized source path."""
        loader = self.get_loader_for_path(path)
        if not loader:
            raise ValueError(f"No suitable loader registered for source: {path}")
        return loader.load(path)

    def discover_sources(self, directory: Union[str, Path], recursive: bool = True) -> List[Path]:
        """Discover all eligible files in a directory matching registered extensions."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []

        pattern = "**/*" if recursive else "*"
        eligible: List[Path] = []
        for p in dir_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in self._loaders:
                # Ignore internal Chroma and metrics databases
                if p.name.endswith(".sqlite3") and ("chroma" in str(p) or "metrics" in str(p)):
                    continue
                eligible.append(p)

        return sorted(eligible)


_GLOBAL_LOADER_REGISTRY: Optional[LoaderRegistry] = None


def get_loader_registry() -> LoaderRegistry:
    """Return the global LoaderRegistry singleton instance."""
    global _GLOBAL_LOADER_REGISTRY
    if _GLOBAL_LOADER_REGISTRY is None:
        _GLOBAL_LOADER_REGISTRY = LoaderRegistry()
    return _GLOBAL_LOADER_REGISTRY
