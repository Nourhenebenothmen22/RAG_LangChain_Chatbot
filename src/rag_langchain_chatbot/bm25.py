"""Production Lexical BM25 Manager with serialization and fast querying."""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Token regex preserving words, alphanumeric terms, and decimal numbers (e.g., 0.6 or 0,6)
_TOKEN_PATTERN = re.compile(r"\b\w+(?:[.,]\w+)*\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase lexical terms, normalizing decimal separators."""
    if not text:
        return []
    # Normalize French comma-decimals (0,6) to standard dot-decimals (0.6) for consistent matching
    normalized = re.sub(r"(\d+),(\d+)", r"\1.\2", text.lower())
    return _TOKEN_PATTERN.findall(normalized)


class BM25Index:
    """Persistent BM25 lexical search index over LangChain Document objects."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = documents or []
        self.tokenized_corpus: List[List[str]] = [
            tokenize(doc.page_content) for doc in self.documents
        ]
        self.bm25: Optional[BM25Okapi] = (
            BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            if self.tokenized_corpus
            else None
        )

    def build_from_documents(self, documents: List[Document]) -> "BM25Index":
        """Build or rebuild the BM25 index from a list of documents."""
        self.documents = list(documents)
        self.tokenized_corpus = [
            tokenize(doc.page_content) for doc in self.documents
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = None
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist BM25 index and indexed documents to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ],
        }
        if target_path.suffix == ".json":
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            with open(target_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Persisted BM25 index with %d documents to %s", len(self.documents), target_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        """Load a persisted BM25 index from disk."""
        target_path = Path(path)
        if not target_path.is_file():
            raise FileNotFoundError(f"BM25 index file not found at: {target_path}")

        try:
            if target_path.suffix == ".json":
                with open(target_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                with open(target_path, "rb") as f:
                    payload = pickle.load(f)
        except Exception:
            with open(target_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        raw_docs = payload.get("documents", [])
        if raw_docs and isinstance(raw_docs[0], dict):
            documents = [
                Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in raw_docs
            ]
        else:
            documents = raw_docs

        instance = cls(documents=documents, k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        logger.info("Loaded BM25 index with %d documents from %s", len(instance.documents), target_path)
        return instance

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents matching the query with BM25 scores."""
        if not self.bm25 or not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_k = min(top_k, len(self.documents))

        if top_k <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        query_set = set(query_tokens)

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            overlap = query_set.intersection(self.tokenized_corpus[idx])
            
            # If standard BM25 score > 0, use it
            if score > 0:
                results.append((self.documents[idx], score))
            # Handle edge cases (e.g. tiny test corpora where log(1)=0 gives zero IDF)
            elif overlap:
                fallback_score = float(len(overlap))
                results.append((self.documents[idx], fallback_score))

        return results


_BM25_CACHE: Optional[BM25Index] = None


def get_bm25_index(settings: Optional[Settings] = None, reload: bool = False) -> BM25Index:
    """Return the cached BM25 index instance, loading from disk if needed."""
    global _BM25_CACHE
    cfg = settings or get_settings()

    if _BM25_CACHE is None or reload:
        if cfg.bm25_index_path.is_file():
            _BM25_CACHE = BM25Index.load(cfg.bm25_index_path)
        else:
            logger.warning(
                "BM25 index not found at %s. Returning uninitialized index.",
                cfg.bm25_index_path
            )
            _BM25_CACHE = BM25Index(k1=cfg.bm25_k1, b=cfg.bm25_b)

    return _BM25_CACHE
