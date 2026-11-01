"""Tests for synchronization ledger, change detection, and deletion tracking."""

from pathlib import Path

from rag_langchain_chatbot.index import SyncLedger


def test_sync_ledger_crud(tmp_path: Path):
    """Test inserting, reading, updating, and removing ledger records."""
    db_file = tmp_path / "test_ledger.sqlite3"
    ledger = SyncLedger(db_file)

    # 1. Initially empty
    records = ledger.get_all_records()
    assert len(records) == 0

    # 2. Record new source
    ledger.record_source(
        source_id="file1.pdf",
        source_type="pdf",
        content_hash="hash_v1",
        chunk_ids=["chunk_1", "chunk_2"]
    )

    records = ledger.get_all_records()
    assert "file1.pdf" in records
    assert records["file1.pdf"]["content_hash"] == "hash_v1"
    assert records["file1.pdf"]["chunk_count"] == 2
    assert records["file1.pdf"]["chunk_ids"] == ["chunk_1", "chunk_2"]

    # 3. Update source with new hash
    ledger.record_source(
        source_id="file1.pdf",
        source_type="pdf",
        content_hash="hash_v2",
        chunk_ids=["chunk_1_new", "chunk_2_new", "chunk_3_new"]
    )

    records = ledger.get_all_records()
    assert records["file1.pdf"]["content_hash"] == "hash_v2"
    assert records["file1.pdf"]["chunk_count"] == 3

    # 4. Remove source (simulating deletion synchronization)
    ledger.remove_source("file1.pdf")
    records = ledger.get_all_records()
    assert "file1.pdf" not in records
