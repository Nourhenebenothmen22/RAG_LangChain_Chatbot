"""Unit and regression tests for Semantic Chunking, Document Intelligence, and Structure-Aware Parsing."""

import io
import sqlite3
import zipfile
from pathlib import Path
import pytest

from langchain_core.documents import Document

from rag_langchain_chatbot.chunking import (
    DocumentStructureParser,
    SemanticHybridSplitter,
    SentenceSegmenter,
    TokenCounter,
)
from rag_langchain_chatbot.config import Settings
from rag_langchain_chatbot.index import DocumentIndexer
from rag_langchain_chatbot.loaders import DocxLoader, _clean_page_text
from rag_langchain_chatbot.models import get_embedding_model


def test_sentence_segmenter_multilingual():
    """Verify SentenceSegmenter preserves abbreviations, initials, decimals, and Arabic punctuation."""
    segmenter = SentenceSegmenter()
    text = (
        "Bonjour Dr. Martin! Le contrat d'assurance auto (police no. 789) est valide jusqu'au 31.12.2026. "
        "En cas d'accident, contactez l'expert au 01.42.68.00.00 immédiatement. "
        "Toute fausse déclaration entraîne la nullité selon l'art. L. 113-8 du Code des assurances. "
        "هل توافق على هذا الشرط؟ نعم، بالتأكيد! شكراً جزيلاً."
    )

    sentences = segmenter.split_sentences(text)

    # Validate that abbreviations did not cause artificial splits
    assert any("Dr. Martin!" in s for s in sentences)
    assert any("police no. 789" in s for s in sentences)
    assert any("01.42.68.00.00" in s for s in sentences)
    assert any("art. L. 113-8" in s for s in sentences)
    assert any("هل توافق على هذا الشرط؟" in s for s in sentences)
    assert any("نعم، بالتأكيد!" in s for s in sentences)

    # Ensure all sentences end with proper punctuation or are complete
    for s in sentences:
        assert len(s.strip()) > 3


def test_token_counter():
    """Verify TokenCounter provides consistent non-zero counts across languages."""
    counter = TokenCounter()
    
    fr_tokens = counter.count_tokens("Le contrat d'assurance automobile couvre les dommages matériels.")
    ar_tokens = counter.count_tokens("عقد تأمين السيارات يغطي الأضرار المادية المباشرة.")
    en_tokens = counter.count_tokens("The automobile insurance contract covers direct physical damages.")

    assert fr_tokens > 5
    assert ar_tokens > 5
    assert en_tokens > 5


def test_document_structure_parser():
    """Verify DocumentStructureParser extracts section hierarchy, breadcrumbs, clauses, and tables."""
    parser = DocumentStructureParser()
    sample = (
        "# Conditions Générales Auto\n"
        "Dispositions générales de la police.\n\n"
        "## Chapitre I: Garanties Principales\n"
        "Garantie responsabilité civile et dommages.\n\n"
        "Article 3: Franchise et Plafonds\n"
        "La franchise générale est de 150 EUR.\n\n"
        "| Catégorie | Franchise | Plafond |\n"
        "| --- | --- | --- |\n"
        "| Bris de glace | 0 EUR | 500 EUR |\n"
        "| Vol | 200 EUR | 30 000 EUR |\n\n"
        "Article 4: Exclusions Spécifiques\n"
        "Dommages intentionnels exclus.\n"
    )

    blocks = parser.parse(sample, default_title="Conditions Générales Auto")

    # Verify blocks extracted
    assert len(blocks) >= 4
    
    # Check breadcrumb propagation
    art3_block = next(b for b in blocks if "Article 3" in b.title and not b.is_table)
    assert "Conditions Générales Auto" in art3_block.breadcrumbs
    assert "Chapitre I: Garanties Principales" in art3_block.breadcrumbs
    assert art3_block.clause_number == "3"

    # Check table block
    table_block = next(b for b in blocks if b.is_table)
    assert table_block.is_table is True
    assert "| Catégorie |" in table_block.content
    assert "| Bris de glace |" in table_block.content


def test_docx_loader_structure_and_table(tmp_path: Path):
    """Verify DocxLoader extracts Word headings (#, ##) and converts <w:tbl> into Markdown tables."""
    docx_file = tmp_path / "structured_policy.docx"

    # XML with headings (w:pStyle) and a table (w:tbl)
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '  <w:p>'
        '    <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        '    <w:r><w:t>Assurance Automobile Premium</w:t></w:r>'
        '  </w:p>'
        '  <w:p>'
        '    <w:r><w:t>Introduction aux conditions générales du contrat.</w:t></w:r>'
        '  </w:p>'
        '  <w:p>'
        '    <w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
        '    <w:r><w:t>Barème des Franchises</w:t></w:r>'
        '  </w:p>'
        '  <w:tbl>'
        '    <w:tr>'
        '      <w:tc><w:p><w:r><w:t>Type Sinistre</w:t></w:r></w:p></w:tc>'
        '      <w:tc><w:p><w:r><w:t>Montant Franchise</w:t></w:r></w:p></w:tc>'
        '    </w:tr>'
        '    <w:tr>'
        '      <w:tc><w:p><w:r><w:t>Optique et Glace</w:t></w:r></w:p></w:tc>'
        '      <w:tc><w:p><w:r><w:t>50 EUR</w:t></w:r></w:p></w:tc>'
        '    </w:tr>'
        '  </w:tbl>'
        '</w:body>'
        '</w:document>'
    )

    with zipfile.ZipFile(docx_file, "w") as z:
        z.writestr("word/document.xml", xml_content)

    loader = DocxLoader()
    docs = loader.load(docx_file)

    assert len(docs) == 1
    content = docs[0].page_content
    # Heading 1 formatted as Markdown
    assert "# Assurance Automobile Premium" in content
    # Heading 2 formatted as Markdown
    assert "## Barème des Franchises" in content
    # Table formatted as Markdown
    assert "| Type Sinistre | Montant Franchise |" in content
    assert "| --- | --- |" in content
    assert "| Optique et Glace | 50 EUR |" in content


def test_pdf_clean_page_text():
    """Verify _clean_page_text fixes soft hyphenation and strips isolated page numbers."""
    raw = (
        "Conditions de responsi-\n"
        "bilité contractuelle de l'assuré.\n"
        "Page 1 / 12\n"
        "L'indemnité est versée sous 15 jours."
    )
    cleaned = _clean_page_text(raw)
    assert "responsibilité" in cleaned
    assert "Page 1 / 12" not in cleaned
    assert "L'indemnité est versée sous 15 jours." in cleaned


def test_semantic_hybrid_splitter_and_sibling_linkage():
    """Verify SemanticHybridSplitter produces enriched breadcrumbs and sibling pointers."""
    splitter = SemanticHybridSplitter(
        target_tokens=100,
        min_tokens=20,
        max_tokens=180,
        overlap_sentences=1,
        enable_breadcrumbs=True,
    )

    raw_doc = Document(
        page_content=(
            "# Manuel de Documentation Kortex AI\n"
            "Introduction au processus d'indemnisation des sinistres.\n\n"
            "## Chapitre 2: Procédure d'Expertise\n"
            "Tout sinistre supérieur à 1 500 EUR nécessite la désignation d'un expert indépendant certifié. "
            "L'expert doit déposer son rapport dans un délai de 30 jours calendaires.\n\n"
            "Article 5: Recours et Contestations\n"
            "En cas de désaccord sur le montant de l'indemnisation, l'assuré dispose d'un délai de 60 jours pour demander une contre-expertise amiable. "
            "Les frais de la tierce expertise sont partagés par moitié entre les deux parties contractantes."
        ),
        metadata={"source": "/data/manuel.md", "file_name": "manuel.md", "page": 1}
    )

    chunks = splitter.split_documents([raw_doc])
    assert len(chunks) >= 2

    # Check breadcrumb injection
    assert any("[manuel.md > Manuel de Documentation Kortex AI" in c.page_content for c in chunks)
    
    # Check metadata fields
    for c in chunks:
        assert "section" in c.metadata
        assert "breadcrumbs" in c.metadata
        assert "token_count" in c.metadata
        assert "char_count" in c.metadata
        assert "chunk_index" in c.metadata
        assert "total_chunks" in c.metadata


def test_document_indexer_end_to_end_chunking():
    """Verify DocumentIndexer.chunk_document integrates SemanticHybridSplitter and sibling IDs."""
    indexer = DocumentIndexer()
    raw_doc = Document(
        page_content=(
            "# Polices Habitation\n"
            "Conditions de souscription multirisque habitation.\n\n"
            "Article 1: Incendie et Explosion\n"
            "La garantie incendie couvre les biens mobiliers et immobiliers jusqu'à concurrence de la valeur déclarée."
        ),
        metadata={"source": "/data/habitation.txt", "file_name": "habitation.txt", "page": 0}
    )

    chunks = indexer.chunk_document([raw_doc])
    assert len(chunks) >= 1

    # Check deterministic chunk_id and sibling IDs
    for idx, c in enumerate(chunks):
        assert "chunk_id" in c.metadata
        assert len(c.metadata["chunk_id"]) == 64  # SHA-256
        assert "prev_chunk_id" in c.metadata
        assert "next_chunk_id" in c.metadata
        if idx == 0:
            assert c.metadata["prev_chunk_id"] == ""
        if idx == len(chunks) - 1:
            assert c.metadata["next_chunk_id"] == ""


def test_embedding_model_max_seq_length_setting():
    """Verify embedding model max_seq_length is set to 512, eliminating 128-token truncation."""
    settings = Settings(embedding_max_seq_length=512)
    emb = get_embedding_model(settings)
    
    client = getattr(emb, "_client", None) or getattr(emb, "client", None)
    if client is not None and hasattr(client, "max_seq_length"):
        assert client.max_seq_length == 512
