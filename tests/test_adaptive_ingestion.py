"""Tests for modular multi-format ingestion loaders and source registry."""

import json
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_langchain_chatbot.loaders import (
    DocxLoader,
    LoaderRegistry,
    PDFLoader,
    StructuredDatabaseLoader,
    TabularFileLoader,
    TextLoader,
    compute_file_sha256,
    get_loader_registry,
)


def test_text_loader(tmp_path: Path):
    """Test loading plain text and markdown documents."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world from text loader.", encoding="utf-8")

    loader = TextLoader()
    docs = loader.load(txt_file)

    assert len(docs) == 1
    assert docs[0].page_content == "Hello world from text loader."
    assert docs[0].metadata["file_type"] == "txt"
    assert "source_hash" in docs[0].metadata


def test_docx_loader(tmp_path: Path):
    """Test loading DOCX documents by generating a valid docx zip package."""
    docx_file = tmp_path / "test_doc.docx"

    # Create minimal WordprocessingML XML
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>First paragraph in DOCX.</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Second paragraph with data.</w:t></w:r></w:p>'
        '</w:body>'
        '</w:document>'
    )

    with zipfile.ZipFile(docx_file, "w") as z:
        z.writestr("word/document.xml", xml_content)

    loader = DocxLoader()
    docs = loader.load(docx_file)

    assert len(docs) == 1
    assert "First paragraph in DOCX." in docs[0].page_content
    assert "Second paragraph with data." in docs[0].page_content
    assert docs[0].metadata["file_type"] == "docx"


def test_structured_database_loader(tmp_path: Path):
    """Test loading structured rows from a SQLite database table."""
    db_file = tmp_path / "test_warehouse.sqlite3"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, policy_no TEXT, amount REAL, status TEXT);")
    conn.execute("INSERT INTO claims VALUES (101, 'POL-789', 1500.50, 'APPROVED');")
    conn.execute("INSERT INTO claims VALUES (102, 'POL-790', 4200.00, 'UNDER_REVIEW');")
    conn.commit()
    conn.close()

    loader = StructuredDatabaseLoader()
    docs = loader.load(db_file)

    assert len(docs) == 2
    assert "Table: claims" in docs[0].page_content
    assert "policy_no: POL-789" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "database_record"
    assert docs[0].metadata["table_name"] == "claims"
    assert docs[0].metadata["record_id"] == "101"

    # Test with custom text_columns and specific table
    filtered_loader = StructuredDatabaseLoader(table_name="claims", text_columns=["policy_no", "status"])
    filtered_docs = filtered_loader.load(db_file)
    assert len(filtered_docs) == 2
    assert "policy_no: POL-789" in filtered_docs[0].page_content
    assert "amount:" not in filtered_docs[0].page_content


def test_tabular_file_loader(tmp_path: Path):
    """Test loading CSV and JSON records."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("item,price\nLaptop,1200\nMouse,25\n", encoding="utf-8")

    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"role": "ADMIN", "permissions": "all"}]), encoding="utf-8")

    loader = TabularFileLoader()
    csv_docs = loader.load(csv_file)
    json_docs = loader.load(json_file)

    assert len(csv_docs) == 2
    assert "item: Laptop" in csv_docs[0].page_content

    assert len(json_docs) == 1
    assert "role: ADMIN" in json_docs[0].page_content


def test_loader_registry_discovery(tmp_path: Path):
    """Test dynamic source discovery across subdirectories."""
    registry = get_loader_registry()

    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "doc1.txt").write_text("content 1", encoding="utf-8")
    (sub / "doc2.md").write_text("content 2", encoding="utf-8")
    (sub / "ignored.bin").write_bytes(b"\x00\x01")

    discovered = registry.discover_sources(tmp_path)
    discovered_names = [f.name for f in discovered]

    assert "doc1.txt" in discovered_names
    assert "doc2.md" in discovered_names
    assert "ignored.bin" not in discovered_names
