"""High-performance Hybrid Retriever combining Dense Vector Search, Lexical BM25, RRF and Cross-Encoder."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langsmith import traceable

from rag_langchain_chatbot.bm25 import BM25Index, get_bm25_index
from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.index import generate_deterministic_chunk_id
from rag_langchain_chatbot.models import get_vector_store
from rag_langchain_chatbot.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@traceable(run_type="parser", name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_k: int = 10,
    source_names: Optional[List[str]] = None
) -> List[Tuple[Document, float]]:
    """Combine multiple ranked document lists using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_score(d) = SUM_m [ 1 / (k + rank_m(d)) ]
    
    Preserves dense and sparse contribution provenance in document metadata.
    
    Args:
        ranked_lists: List of candidate document rankings (each is a list of (Document, original_score)).
        k: Smoothing constant (default 60, per standard information retrieval benchmarks).
        top_k: Number of fused results to return.
        source_names: Optional list of names corresponding to each ranked list (e.g. ["dense", "bm25"]).
        
    Returns:
        List of (Document, rrf_score) sorted in descending order of fused score.
    """
    fused_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Document] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    names = source_names or [f"retriever_{i}" for i in range(len(ranked_lists))]

    for list_idx, ranked_list in enumerate(ranked_lists):
        channel_name = names[list_idx] if list_idx < len(names) else f"channel_{list_idx}"
        seen_in_current_list = set()

        for rank, (doc, raw_score) in enumerate(ranked_list, start=1):
            doc_id = doc.metadata.get("chunk_id")
            if not doc_id:
                doc_id = generate_deterministic_chunk_id(doc)
                doc.metadata["chunk_id"] = doc_id

            if doc_id in seen_in_current_list:
                continue

            seen_in_current_list.add(doc_id)

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_registry[doc_id] = doc
                provenance[doc_id] = {}

            fused_scores[doc_id] += 1.0 / (k + rank)
            provenance[doc_id][f"{channel_name}_rank"] = rank
            provenance[doc_id][f"{channel_name}_score"] = round(raw_score, 4)

    # Sort documents by fused RRF score
    sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    results: List[Tuple[Document, float]] = []
    for doc_id, score in sorted_items[:top_k]:
        doc = doc_registry[doc_id]
        doc.metadata["rrf_score"] = round(score, 6)
        # Attach channel contribution rankings
        for key, val in provenance[doc_id].items():
            doc.metadata[key] = val
        results.append((doc, score))

    return results


class HybridRetriever:
    """Production Hybrid Retriever combining Dense Embeddings, BM25 Lexical search, RRF, and Cross-Encoder."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.vector_store = get_vector_store(self.settings)
        self.bm25_index: BM25Index = get_bm25_index(self.settings)
        self.reranker = CrossEncoderReranker(self.settings)

    @traceable(run_type="retriever", name="dense_vector_search")
    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve candidate documents using dense vector similarity."""
        k = top_k or self.settings.dense_top_k
        docs_and_scores = self.vector_store.similarity_search_with_score(query=query, k=k)
        return [(doc, float(score)) for doc, score in docs_and_scores]

    @traceable(run_type="retriever", name="bm25_lexical_search")
    def retrieve_bm25(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve candidate documents using BM25 lexical keyword matching."""
        k = top_k or self.settings.bm25_top_k
        return self.bm25_index.search(query=query, top_k=k)

    @traceable(run_type="retriever", name="hybrid_retriever")
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: bool = True
    ) -> List[Tuple[Document, float]]:
        """Execute the complete Hybrid RAG retrieval pipeline with full provenance tracking.
        
        Steps:
            1. Dense vector search (top-k candidates)
            2. BM25 lexical search (top-k candidates)
            3. Reciprocal Rank Fusion with dense/sparse provenance
            4. Cross-Encoder reranking (high-precision final top-k)
            
        Args:
            query: The user search query.
            top_k: Final number of documents to return.
            rerank: Whether to apply Cross-Encoder reranking.
            
        Returns:
            List of (Document, score) pairs.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_clean = query.strip()
        final_k = top_k or self.settings.reranker_top_k

        # 1. Fetch dense candidates
        dense_candidates = self.retrieve_dense(
            query=query_clean,
            top_k=self.settings.dense_top_k
        )

        # 2. Fetch lexical BM25 candidates
        bm25_candidates = self.retrieve_bm25(
            query=query_clean,
            top_k=self.settings.bm25_top_k
        )

        # Handle case where both retrievers return empty results
        if not dense_candidates and not bm25_candidates:
            logger.info("Both dense and BM25 retrievers returned empty results for query: %s", query_clean)
            return []

        # 3. Fuse rankings via RRF with channel provenance
        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=[dense_candidates, bm25_candidates],
            k=self.settings.rrf_k,
            top_k=self.settings.fused_top_k,
            source_names=["dense", "sparse"]
        )

        if not rerank or not fused_candidates:
            return fused_candidates[:final_k]

        # 4. High-Precision Cross-Encoder Reranking
        candidate_docs = [doc for doc, _ in fused_candidates]
        reranked_results = self.reranker.rerank(
            query=query_clean,
            documents=candidate_docs,
            top_k=final_k,
            min_score=self.settings.min_relevance_score
        )

        # Attach rerank score to document metadata
        for doc, score in reranked_results:
            doc.metadata["rerank_score"] = round(score, 4)

        return reranked_results


def hybrid_retrieval(
    query: str,
    top_k: Optional[int] = None,
    rerank: bool = True,
    settings: Optional[Settings] = None
) -> List[Tuple[Document, float]]:
    """Convenience functional interface for hybrid retrieval."""
    retriever = HybridRetriever(settings=settings)
    return retriever.retrieve(query=query, top_k=top_k, rerank=rerank)
