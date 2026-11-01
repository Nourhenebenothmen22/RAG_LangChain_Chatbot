"""Tests for diacritic folding, multilingual keyword matching, and page 0 chunk identification."""

from langchain_core.documents import Document
from rag_langchain_chatbot.bm25 import BM25Index, normalize_diacritics
from rag_langchain_chatbot.index import generate_deterministic_chunk_id


def test_diacritic_folding():
    """Verify normalize_diacritics removes accents while preserving underlying letters."""
    assert normalize_diacritics("déterministe") == "deterministe"
    assert normalize_diacritics("pièces") == "pieces"
    assert normalize_diacritics("rôle") == "role"
    assert normalize_diacritics("coût") == "cout"


def test_bm25_cross_lingual_diacritic_search():
    """Verify BM25 matches unaccented queries against accented text."""
    docs = [
        Document(
            page_content="BF-08 : Chiffrage déterministe en Python (pièces + main-d'œuvre - franchise).",
            metadata={"chunk_id": "chunk_calc", "source": "spec.pdf"}
        ),
        Document(
            page_content="Section 3 : Tarification de base 490 TND par mois.",
            metadata={"chunk_id": "chunk_price", "source": "spec.pdf"}
        ),
    ]

    index = BM25Index(documents=docs)
    
    # Query without accents
    results_unaccented = index.search("chiffrage deterministe pieces", top_k=1)
    assert len(results_unaccented) == 1
    assert results_unaccented[0][0].metadata["chunk_id"] == "chunk_calc"

    # Query with accents
    results_accented = index.search("chiffrage déterministe pièces", top_k=1)
    assert len(results_accented) == 1
    assert results_accented[0][0].metadata["chunk_id"] == "chunk_calc"


def test_deterministic_chunk_id_page_zero():
    """Verify page 0 is preserved as a distinct page identifier and does not collapse into empty string."""
    doc_page_0 = Document(page_content="Cover page content", metadata={"source": "doc.pdf", "page": 0})
    doc_no_page = Document(page_content="Cover page content", metadata={"source": "doc.pdf"})

    id_page_0 = generate_deterministic_chunk_id(doc_page_0)
    id_no_page = generate_deterministic_chunk_id(doc_no_page)

    # Page 0 has explicit page identifier, no page has empty string
    assert id_page_0 != id_no_page
    assert len(id_page_0) == 64
