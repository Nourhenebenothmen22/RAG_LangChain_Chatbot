"""Cross-Encoder reranking module for precision scoring of candidate documents."""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[tuple[str, str], CrossEncoder] = {}


def get_cross_encoder(settings: Optional[Settings] = None) -> CrossEncoder:
    """Return a cached CrossEncoder instance."""
    cfg = settings or get_settings()
    key = (cfg.reranker_model_name, cfg.reranker_device)

    if key not in _RERANKER_CACHE:
        logger.info(
            "Loading Cross-Encoder model: %s on %s",
            cfg.reranker_model_name,
            cfg.reranker_device
        )
        _RERANKER_CACHE[key] = CrossEncoder(
            model_name_or_path=cfg.reranker_model_name,
            device=cfg.reranker_device
        )

    return _RERANKER_CACHE[key]


class CrossEncoderReranker:
    """Applies high-precision cross-attention scoring to candidate documents."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily initialize and return the CrossEncoder model."""
        if self._model is None:
            self._model = get_cross_encoder(self.settings)
        return self._model

    @traceable(run_type="parser", name="cross_encoder_rerank")
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank a list of documents against a query using the Cross-Encoder.
        
        Args:
            query: The user query string.
            documents: List of candidate Document objects.
            top_k: Maximum number of reranked documents to return.
            min_score: Optional minimum relevance score threshold.
            
        Returns:
            List of (Document, score) sorted in descending order of relevance.
        """
        if not documents:
            return []

        limit = top_k if top_k is not None else self.settings.reranker_top_k
        score_threshold = (
            min_score if min_score is not None else self.settings.min_relevance_score
        )

        pairs = [(query, doc.page_content) for doc in documents]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.settings.reranker_batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error("Error during cross-encoder prediction: %s. Returning unreranked.", e)
            return [(doc, 0.0) for doc in documents[:limit]]

        ranked_results: List[Tuple[Document, float]] = []
        for doc, score in zip(documents, scores):
            score_float = float(score)
            if score_threshold is not None and score_float < score_threshold:
                continue

            # Enrich metadata with rerank score
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": score_float}
            )
            ranked_results.append((doc_copy, score_float))

        # Sort descending by score
        ranked_results.sort(key=lambda item: item[1], reverse=True)

        return ranked_results[:limit]
