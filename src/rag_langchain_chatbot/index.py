"""Adaptive document indexing and database synchronization with change detection and deletion tracking."""

import datetime
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_langchain_chatbot.bm25 import BM25Index
from rag_langchain_chatbot.chunking import SemanticHybridSplitter
from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.loaders import (
    LoaderRegistry,
    compute_file_sha256,
    get_loader_registry,
)
from rag_langchain_chatbot.models import get_vector_store

logger = logging.getLogger(__name__)


def generate_deterministic_chunk_id(document: Document) -> str:
    """Generate a stable, content-derived ID for a document chunk.
    
    Format: sha256(source|page_or_record|content_hash)
    Explicitly handles page 0 without falsy coercion.
    """
    source = document.metadata.get("source", "")
    page = document.metadata.get("page")
    record_id = document.metadata.get("record_id")
    page_or_record = str(page) if page is not None else (str(record_id) if record_id is not None else "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    raw_key = f"{source}|{page_or_record}|{content_hash}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class SyncLedger:
    """Persistent SQLite ledger tracking source documents, content hashes, and chunk IDs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_ledger()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_ledger(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_sync_ledger (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    last_synced_at TEXT NOT NULL
                );
            """)
            conn.commit()

    def get_all_records(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve all tracked sources in the ledger."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM source_sync_ledger;").fetchall()
        result = {}
        for r in rows:
            result[r["source_id"]] = {
                "source_type": r["source_type"],
                "content_hash": r["content_hash"],
                "chunk_ids": json.loads(r["chunk_ids"]),
                "chunk_count": r["chunk_count"],
                "last_synced_at": r["last_synced_at"],
            }
        return result

    def record_source(self, source_id: str, source_type: str, content_hash: str, chunk_ids: List[str]) -> None:
        """Upsert source synchronization record."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO source_sync_ledger (
                    source_id, source_type, content_hash, chunk_ids, chunk_count, last_synced_at
                ) VALUES (?, ?, ?, ?, ?, ?);
            """, (source_id, source_type, content_hash, json.dumps(chunk_ids), len(chunk_ids), now_iso))
            conn.commit()

    def remove_source(self, source_id: str) -> None:
        """Delete source synchronization record."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM source_sync_ledger WHERE source_id = ?;", (source_id,))
            conn.commit()

    def clear(self) -> None:
        """Clear all records from the ledger."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM source_sync_ledger;")
            conn.commit()


class DocumentIndexer:
    """Orchestrates multi-source ingestion, incremental sync, deletion handling, and dual indexing."""

    def __init__(self, settings: Optional[Settings] = None, registry: Optional[LoaderRegistry] = None):
        self.settings = settings or get_settings()
        self.registry = registry or get_loader_registry()
        self.vector_store = get_vector_store(self.settings)
        self.ledger = SyncLedger(self.settings.metrics_db_path)
        self.semantic_splitter = SemanticHybridSplitter(
            target_tokens=self.settings.chunk_target_tokens,
            min_tokens=self.settings.chunk_min_tokens,
            max_tokens=self.settings.chunk_max_tokens,
            overlap_sentences=self.settings.chunk_overlap_sentences,
            enable_breadcrumbs=self.settings.enable_context_breadcrumbs,
            tokenizer_model_name=self.settings.embedding_model_name,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=self.settings.chunk_separators,
        )

    def discover_all_sources(self) -> List[Tuple[str, str, str]]:
        """Discover all eligible files and connected database sources.
        
        Returns:
            List of tuples: (source_id, source_type, content_hash)
        """
        discovered = []

        # 1. Discover local files in data directory
        if self.settings.data_dir.exists():
            files = self.registry.discover_sources(self.settings.data_dir)
            for f in files:
                file_hash = compute_file_sha256(f)
                discovered.append((str(f.resolve()), f.suffix.lstrip(".").lower() or "file", file_hash))

        # 2. Discover connected database if configured
        if self.settings.database_url:
            db_path = Path(self.settings.database_url.replace("sqlite:///", "").replace("sqlite://", ""))
            if db_path.is_file():
                db_hash = compute_file_sha256(db_path)
                discovered.append((str(db_path.resolve()), "sqlite_database", db_hash))

        return discovered

    def chunk_document(self, raw_docs: List[Document]) -> List[Document]:
        """Split raw documents into chunks with deterministic IDs, sibling linkage, and preserved metadata."""
        try:
            chunks = self.semantic_splitter.split_documents(raw_docs)
        except Exception as e:
            logger.warning("SemanticHybridSplitter encountered an error (%s); falling back to text_splitter", e)
            chunks = self.text_splitter.split_documents(raw_docs)

        for chunk in chunks:
            chunk_id = generate_deterministic_chunk_id(chunk)
            chunk.metadata["chunk_id"] = chunk_id
            chunk.id = chunk_id

        # Sibling chunk linkage for context window expansion
        for idx, chunk in enumerate(chunks):
            chunk.metadata["prev_chunk_id"] = chunks[idx - 1].metadata["chunk_id"] if idx > 0 else ""
            chunk.metadata["next_chunk_id"] = chunks[idx + 1].metadata["chunk_id"] if idx < len(chunks) - 1 else ""

        return chunks

    def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Synchronize documents and database records with Chroma vector store and BM25 index.
        
        Performs:
        1. Content hash change detection.
        2. Incremental addition of new documents.
        3. Updating of modified documents (purging previous chunks).
        4. Deletion synchronization (purging chunks of deleted files/records).
        5. Rebuilding and persisting the BM25 index.
        """
        if force_reindex:
            logger.info("Force reindex requested. Resetting collection %s and ledger...", self.settings.collection_name)
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                logger.warning("Error deleting collection: %s", e)
            self.vector_store = get_vector_store(self.settings)
            self.ledger.clear()

        ledger_records = self.ledger.get_all_records()
        discovered_sources = self.discover_all_sources()
        discovered_map = {src_id: (src_type, src_hash) for src_id, src_type, src_hash in discovered_sources}

        added_sources = 0
        modified_sources = 0
        deleted_sources = 0
        unchanged_sources = 0
        chunks_added_total = 0

        # Phase 1: Handle Deletions (present in ledger, but missing on disk/db)
        for tracked_id, record in ledger_records.items():
            if tracked_id not in discovered_map:
                logger.info("Source deleted from disk/database: %s. Pruning %d chunks...", tracked_id, record["chunk_count"])
                try:
                    self.vector_store.delete(ids=record["chunk_ids"])
                except Exception as ex:
                    logger.warning("Failed to delete chunks from Chroma for '%s': %s", tracked_id, ex)
                self.ledger.remove_source(tracked_id)
                deleted_sources += 1

        # Phase 2: Ingest New or Modified Sources
        for src_id, (src_type, current_hash) in discovered_map.items():
            prev_record = ledger_records.get(src_id)

            if prev_record is not None and prev_record["content_hash"] == current_hash and not force_reindex:
                unchanged_sources += 1
                continue

            # If modified, prune previous chunks first
            if prev_record is not None:
                logger.info("Source modified: %s. Replacing %d outdated chunks...", src_id, prev_record["chunk_count"])
                try:
                    self.vector_store.delete(ids=prev_record["chunk_ids"])
                except Exception as ex:
                    logger.warning("Failed to delete outdated chunks for '%s': %s", src_id, ex)
                modified_sources += 1
            else:
                logger.info("New source discovered: %s", src_id)
                added_sources += 1

            # Ingest and chunk
            try:
                raw_docs = self.registry.load_source(src_id)
                chunks = self.chunk_document(raw_docs)
                chunk_ids = [c.metadata["chunk_id"] for c in chunks]

                if chunks:
                    self.vector_store.add_documents(documents=chunks, ids=chunk_ids)
                    chunks_added_total += len(chunks)

                self.ledger.record_source(
                    source_id=src_id,
                    source_type=src_type,
                    content_hash=current_hash,
                    chunk_ids=chunk_ids
                )
            except Exception as e:
                logger.error("Failed to ingest source '%s': %s", src_id, e)

        # Phase 3: Rebuild and Persist BM25 Index
        needs_bm25_rebuild = (added_sources > 0 or modified_sources > 0 or deleted_sources > 0 or force_reindex or not self.settings.bm25_index_path.exists())

        if needs_bm25_rebuild:
            logger.info("Rebuilding BM25 index to reflect database state...")
            stored_data = self.vector_store.get(include=["documents", "metadatas"])
            all_indexed_docs: List[Document] = []

            for content, metadata in zip(stored_data["documents"], stored_data["metadatas"]):
                all_indexed_docs.append(Document(page_content=content, metadata=metadata or {}))

            bm25_index = BM25Index(
                documents=all_indexed_docs,
                k1=self.settings.bm25_k1,
                b=self.settings.bm25_b,
            )
            bm25_index.save(self.settings.bm25_index_path)
            total_bm25_docs = len(all_indexed_docs)
        else:
            try:
                bm25_index = BM25Index.load(self.settings.bm25_index_path)
                total_bm25_docs = len(bm25_index.documents)
            except Exception:
                total_bm25_docs = 0

        try:
            total_chroma_count = self.vector_store._collection.count()
        except Exception:
            try:
                total_chroma_count = len(self.vector_store.get().get("ids", []))
            except Exception:
                total_chroma_count = 0
        stats = {
            "sources_discovered": len(discovered_sources),
            "sources_added": added_sources,
            "sources_modified": modified_sources,
            "sources_deleted": deleted_sources,
            "sources_unchanged": unchanged_sources,
            "new_chunks_added": chunks_added_total,
            "total_chroma_chunks": total_chroma_count,
            "total_bm25_documents": total_bm25_docs,
        }
        logger.info("Database synchronization summary: %s", stats)
        return stats


def index_documents(force_reindex: bool = False, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Convenience function to run the full indexing workflow."""
    indexer = DocumentIndexer(settings=settings)
    return indexer.index_documents(force_reindex=force_reindex)


if __name__ == "__main__":
    index_documents()
