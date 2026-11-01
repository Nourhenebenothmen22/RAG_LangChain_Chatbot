"""Multilingual Cross-Encoder reranking module for precision scoring of candidate documents."""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[tuple[str, str], CrossEncoder] = {}

# Known local fallback cross-encoders to ensure offline resiliency
_FALLBACK_RERANKERS = [
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-2-v2",
]


def get_cross_encoder(settings: Optional[Settings] = None) -> CrossEncoder:
    """Return a cached CrossEncoder instance with offline-first loading and resilient fallback.
    
    Supports multilingual rerankers (e.g., BAAI/bge-reranker-v2-m3) across French,
    English, Arabic, and Tunisian dialect. If the configured model is unavailable
    offline and cannot be downloaded, falls back safely to cached models with logging.
    """
    cfg = settings or get_settings()
    key = (cfg.reranker_model_name, cfg.reranker_device)

    if key not in _RERANKER_CACHE:
        logger.info(
            "Initializing Cross-Encoder model: %s on %s",
            cfg.reranker_model_name,
            cfg.reranker_device
        )
        model = None

        # 1. Attempt offline-first load of configured model
        try:
            model = CrossEncoder(
                model_name_or_path=cfg.reranker_model_name,
                device=cfg.reranker_device,
                local_files_only=True
            )
            logger.info("Successfully loaded Cross-Encoder from local cache: %s", cfg.reranker_model_name)
        except Exception as local_err:
            logger.debug(
                "Local files check for '%s' failed (%s). Attempting remote download/hub connection...",
                cfg.reranker_model_name, local_err
            )
            # 2. Check network connectivity before attempting remote hub download
            is_online = False
            try:
                import socket
                with socket.create_connection(("huggingface.co", 443), timeout=1.0):
                    is_online = True
            except Exception:
                logger.info(
                    "Network unreachable; skipping remote download for '%s' and checking local fallbacks.",
                    cfg.reranker_model_name
                )

            if is_online:
                try:
                    model = CrossEncoder(
                        model_name_or_path=cfg.reranker_model_name,
                        device=cfg.reranker_device
                    )
                    logger.info("Successfully initialized Cross-Encoder: %s", cfg.reranker_model_name)
                except Exception as remote_err:
                    logger.warning(
                        "Configured Cross-Encoder '%s' could not be loaded remotely: %s. "
                        "Searching for local cached fallback model...",
                        cfg.reranker_model_name, remote_err
                    )

        # 3. If configured model could not be loaded, search for cached local fallbacks
        if model is None:
            for fallback_candidate in _FALLBACK_RERANKERS:
                try:
                    model = CrossEncoder(
                        model_name_or_path=fallback_candidate,
                        device=cfg.reranker_device,
                        local_files_only=True
                    )
                    logger.warning(
                        "Engaged fallback cross-encoder '%s' because primary model '%s' was unavailable.",
                        fallback_candidate, cfg.reranker_model_name
                    )
                    break
                except Exception:
                    continue

        # 4. If all loading attempts fail, raise RuntimeError with actionable diagnostics
        if model is None:
            raise RuntimeError(
                f"Failed to load configured Cross-Encoder '{cfg.reranker_model_name}' "
                f"and no local cached fallback cross-encoder could be found in HuggingFace cache."
            )

        _RERANKER_CACHE[key] = model

    return _RERANKER_CACHE[key]


class CrossEncoderReranker:
    """Applies high-precision cross-attention scoring to candidate documents."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> Optional[CrossEncoder]:
        """Lazily initialize and return the CrossEncoder model with safe failure handling."""
        if self._model is None:
            try:
                self._model = get_cross_encoder(self.settings)
            except Exception as e:
                logger.warning("CrossEncoder model unavailable (%s). Reranking will pass through unreranked.", e)
                return None
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

        active_model = self.model
        if active_model is None:
            logger.info("Cross-encoder model not loaded; returning unreranked candidates.")
            return [(doc, doc.metadata.get("rrf_score", 0.0)) for doc in documents[:limit]]

        pairs = [(query, doc.page_content) for doc in documents]

        try:
            import torch
            with torch.inference_mode():
                scores = active_model.predict(
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
