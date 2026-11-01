"""High-performance Hybrid Retriever combining Dense Vector Search, Lexical BM25, RRF and Cross-Encoder.

Fully integrated with LangChain's native BaseRetriever, LCEL runnable pipelines,
and concurrent multi-channel retrieval.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langsmith import traceable
from pydantic import ConfigDict, Field, PrivateAttr

from rag_langchain_chatbot.bm25 import BM25Index, get_bm25_index
from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.index import generate_deterministic_chunk_id
from rag_langchain_chatbot.models import get_vector_store
from rag_langchain_chatbot.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def calculate_adaptive_weights(query: str) -> Dict[str, float]:
    """Dynamically calibrate dense vs sparse channel weighting based on query characteristics.
    
    - Highly lexical queries (containing exact codes, alphanumeric IDs, numbers, legal references)
      benefit from higher sparse (BM25) weighting.
    - Long natural language or conceptual questions benefit from higher dense vector weighting.
    """
    query_clean = query.strip()
    words = query_clean.split()
    
    # Check for presence of numbers, codes, or specific alphanumeric tokens (e.g. T301, BF-06, 150 EUR)
    has_digits = bool(re.search(r"\d+", query_clean))
    has_codes = bool(re.search(r"\b[A-Z0-9_\-]{2,}\b", query_clean))
    is_short = len(words) <= 4

    dense_weight = 1.0
    bm25_weight = 1.0

    if has_codes or (has_digits and is_short):
        # Specific identifier/code query: boost lexical sparse matching
        bm25_weight = 1.3
        dense_weight = 0.9
    elif len(words) >= 8:
        # Long conversational/explanatory question: boost dense semantic vector matching
        dense_weight = 1.2
        bm25_weight = 0.85

    return {"dense": dense_weight, "sparse": bm25_weight}


@traceable(run_type="parser", name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_k: int = 10,
    source_names: Optional[List[str]] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[Document, float]]:
    """Combine multiple ranked document lists using Reciprocal Rank Fusion (RRF) with optional adaptive weighting.
    
    Formula: RRF_score(d) = SUM_m [ w_m / (k + rank_m(d)) ]
    
    Preserves dense and sparse contribution provenance in document metadata.
    
    Args:
        ranked_lists: List of candidate document rankings (each is a list of (Document, original_score)).
        k: Smoothing constant (default 60, per standard information retrieval benchmarks).
        top_k: Number of fused results to return.
        source_names: Optional list of names corresponding to each ranked list (e.g. ["dense", "sparse"]).
        channel_weights: Optional map of channel names to importance weights (e.g. {"dense": 1.2, "sparse": 0.8}).
        
    Returns:
        List of (Document, rrf_score) sorted in descending order of fused score.
    """
    fused_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Document] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    names = source_names or [f"retriever_{i}" for i in range(len(ranked_lists))]
    weights = channel_weights or {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        channel_name = names[list_idx] if list_idx < len(names) else f"channel_{list_idx}"
        channel_weight = weights.get(channel_name, 1.0)
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

            fused_scores[doc_id] += channel_weight * (1.0 / (k + rank))
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


class HybridRetriever(BaseRetriever):
    """Production Hybrid Retriever combining Dense Embeddings, BM25 Lexical search, RRF, and Cross-Encoder.
    
    Inherits from LangChain BaseRetriever to enable seamless composition in LCEL chains
    (`retriever | prompt | llm`) while preserving backwards-compatible functional interfaces.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    settings: Settings = Field(default_factory=get_settings)
    top_k: Optional[int] = Field(default=None)
    rerank: bool = Field(default=True)

    _vector_store: Any = PrivateAttr()
    _bm25_index: Optional[BM25Index] = PrivateAttr(default=None)
    _reranker: Optional[CrossEncoderReranker] = PrivateAttr(default=None)

    def __init__(
        self,
        settings: Optional[Settings] = None,
        top_k: Optional[int] = None,
        rerank: bool = True,
        **kwargs: Any
    ):
        cfg = settings or get_settings()
        super().__init__(settings=cfg, top_k=top_k, rerank=rerank, **kwargs)
        self._vector_store = get_vector_store(self.settings)
        self._bm25_index = get_bm25_index(self.settings)
        self._reranker = CrossEncoderReranker(self.settings)

    @property
    def vector_store(self) -> Any:
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = get_bm25_index(self.settings)
        return self._bm25_index

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings)
        return self._reranker

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """LangChain BaseRetriever execution entrypoint for LCEL chains."""
        results = self.retrieve(query=query, top_k=self.top_k, rerank=self.rerank)
        return [doc for doc, _ in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Async LangChain BaseRetriever execution entrypoint."""
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)

    @traceable(run_type="retriever", name="dense_vector_search")
    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve candidate documents using dense vector similarity.
        
        Converts Chroma raw distance to standard similarity: 1 / (1 + distance).
        """
        k = top_k or self.settings.dense_top_k
        docs_and_scores = self.vector_store.similarity_search_with_score(query=query, k=k)
        
        # Convert distance to similarity score
        results = []
        for doc, score in docs_and_scores:
            distance = float(score)
            similarity = 1.0 / (1.0 + max(0.0, distance))
            results.append((doc, round(similarity, 4)))
        return results

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
            1. Concurrent Dense vector search & BM25 lexical search (ThreadPoolExecutor)
            2. Reciprocal Rank Fusion with dense/sparse provenance
            3. Cross-Encoder neural reranking (high-precision final top-k)
            
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

        # 1. Fetch dense and lexical BM25 candidates concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.retrieve_dense, query_clean, self.settings.dense_top_k)
            future_bm25 = executor.submit(self.retrieve_bm25, query_clean, self.settings.bm25_top_k)
            
            try:
                dense_candidates = future_dense.result()
            except Exception as e:
                logger.warning("Dense search encountered error (%s). Falling back to sparse.", e)
                dense_candidates = []

            try:
                bm25_candidates = future_bm25.result()
            except Exception as e:
                logger.warning("BM25 search encountered error (%s). Falling back to dense.", e)
                bm25_candidates = []

        # Handle case where both retrievers return empty results
        if not dense_candidates and not bm25_candidates:
            logger.info("Both dense and BM25 retrievers returned empty results for query: %s", query_clean)
            return []

        # 2. Fuse rankings via RRF with dynamic query-adaptive channel weighting
        channel_weights = calculate_adaptive_weights(query_clean) if self.settings.enable_adaptive_rrf else None
        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=[dense_candidates, bm25_candidates],
            k=self.settings.rrf_k,
            top_k=self.settings.fused_top_k,
            source_names=["dense", "sparse"],
            channel_weights=channel_weights,
        )

        if not rerank or not fused_candidates:
            return fused_candidates[:final_k]

        # 3. High-Precision Cross-Encoder Reranking
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
