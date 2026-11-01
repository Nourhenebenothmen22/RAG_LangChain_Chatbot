"""Production Lexical BM25 Manager with secure JSON serialization and fast querying."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple, Union

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Token regex preserving words, alphanumeric terms, and decimal numbers (e.g., 0.6 or 0,6)
_TOKEN_PATTERN = re.compile(r"\b\w+(?:[.,]\w+)*\b", flags=re.UNICODE)


def normalize_diacritics(text: str) -> str:
    """Normalize unicode text by stripping diacritics/accents (e.g., 'déterministe' -> 'deterministe')."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase lexical terms, normalizing decimal separators and diacritics.
    
    Normalizations applied:
    1. French comma-decimals (0,6) -> standard dot-decimals (0.6).
    2. Unicode diacritics folding (é, è, ê, ô -> e, e, e, o) for robust cross-lingual matching.
    3. Token extraction preserving alphanumeric and decimal formats.
    """
    if not text:
        return []
    # Normalize comma decimals first
    normalized = re.sub(r"(\d+),(\d+)", r"\1.\2", text.lower())
    # Fold diacritics for language-agnostic keyword matching
    folded = normalize_diacritics(normalized)
    tokens = _TOKEN_PATTERN.findall(folded)
    return tokens


class BM25Index:
    """Persistent BM25 lexical search index over LangChain Document objects using safe JSON storage."""

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
        """Persist BM25 index and indexed documents to disk using safe, tamper-resistant JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "format": "json",
            "k1": self.k1,
            "b": self.b,
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ],
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # If caller specified .pkl, also maintain a .json companion
        if target_path.suffix == ".pkl":
            json_target = target_path.with_suffix(".json")
            with open(json_target, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Persisted safe JSON BM25 index with %d documents to %s", len(self.documents), target_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        """Load a persisted BM25 index from disk with safe JSON parsing and legacy pickle migration."""
        target_path = Path(path)
        
        # If file does not exist directly, check potential extensions
        if not target_path.is_file():
            json_equivalent = target_path.with_suffix(".json")
            if json_equivalent.is_file():
                target_path = json_equivalent
            else:
                pkl_equivalent = target_path.with_suffix(".pkl")
                if pkl_equivalent.is_file():
                    target_path = pkl_equivalent
                else:
                    raise FileNotFoundError(f"BM25 index file not found at: {path}")

        # Try reading as JSON first
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Legacy binary pickle BM25 index detected at %s. Migrating to safe JSON format.", target_path)
            json_target = target_path.with_suffix(".json")
            return cls._load_and_migrate_legacy_pickle(target_path, json_target)

        raw_docs = payload.get("documents", [])
        if raw_docs and isinstance(raw_docs[0], dict):
            documents = [
                Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in raw_docs
            ]
        else:
            documents = raw_docs

        instance = cls(documents=documents, k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        logger.info("Loaded safe JSON BM25 index with %d documents from %s", len(instance.documents), target_path)
        return instance

    @classmethod
    def _load_and_migrate_legacy_pickle(cls, pkl_path: Path, json_target_path: Path) -> "BM25Index":
        """Load legacy pickle file once, and immediately migrate it to safe JSON format."""
        import pickle
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        raw_docs = payload.get("documents", [])
        if raw_docs and isinstance(raw_docs[0], dict):
            documents = [
                Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in raw_docs
            ]
        else:
            documents = raw_docs

        instance = cls(documents=documents, k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        # Persist migrated JSON immediately
        instance.save(json_target_path)
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

        query_set = set(query_tokens)
        scored_docs = []
        for idx in range(len(self.documents)):
            bm25_score = float(scores[idx])
            overlap = query_set.intersection(self.tokenized_corpus[idx])
            overlap_score = float(len(overlap))
            
            # Use BM25 score if positive; if zero or negative (IDF<=0 on small corpora), use overlap score
            effective_score = bm25_score if bm25_score > 0 else (overlap_score if overlap_score > 0 else 0.0)
            if effective_score > 0:
                scored_docs.append((idx, effective_score, overlap_score))

        # Sort descending by effective_score, then overlap_score
        scored_docs.sort(key=lambda x: (x[1], x[2]), reverse=True)
        results = [(self.documents[idx], eff_score) for idx, eff_score, _ in scored_docs[:top_k]]
        return results


_BM25_CACHE: Optional[BM25Index] = None


def get_bm25_index(settings: Optional[Settings] = None, reload: bool = False) -> BM25Index:
    """Return the cached BM25 index instance, loading from disk if needed."""
    global _BM25_CACHE
    cfg = settings or get_settings()

    if _BM25_CACHE is None or reload:
        # Check both JSON and legacy pkl path
        primary_path = cfg.bm25_index_path
        fallback_json = primary_path.with_suffix(".json")
        fallback_pkl = primary_path.with_suffix(".pkl")

        if primary_path.is_file():
            _BM25_CACHE = BM25Index.load(primary_path)
        elif fallback_json.is_file():
            _BM25_CACHE = BM25Index.load(fallback_json)
        elif fallback_pkl.is_file():
            _BM25_CACHE = BM25Index.load(fallback_pkl)
        else:
            logger.warning(
                "BM25 index not found at %s. Returning uninitialized index.",
                cfg.bm25_index_path
            )
            _BM25_CACHE = BM25Index(k1=cfg.bm25_k1, b=cfg.bm25_b)

    return _BM25_CACHE
