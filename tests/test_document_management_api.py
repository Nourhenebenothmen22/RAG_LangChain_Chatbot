"""Integration tests for Document Management & Ingestion API endpoints."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from rag_langchain_chatbot.api import app, get_pipeline


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a TestClient for FastAPI app."""
    return TestClient(app)


def test_list_documents(client: TestClient):
    """Verify listing existing documents returns structured items and chunk counts."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data
    assert "total_chunks" in data
    assert isinstance(data["documents"], list)

    if data["documents"]:
        first = data["documents"][0]
        assert "name" in first
        assert "type" in first
        assert "size_bytes" in first
        assert "chunk_count" in first


def test_upload_unsupported_file_fails(client: TestClient):
    """Verify uploading an unsupported file format returns 400 Bad Request."""
    file_content = b"print('malicious or unsupported script')"
    files = [("files", ("unsupported_script.exe", io.BytesIO(file_content), "application/octet-stream"))]

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_upload_list_and_delete_document_lifecycle(client: TestClient):
    """Verify full end-to-end lifecycle: upload, index, inspect, and delete."""
    pipeline = get_pipeline()
    test_filename = "test_insurance_endorsement_temp.txt"
    test_content = (
        "Kortex AI Endorsement Notice: All windshield glass damage under 200 EUR has zero deductible. "
        "Any comprehensive bodily injury claim exceeding 50000 EUR requires Senior Adjuster approval."
    ).encode("utf-8")

    files = [("files", (test_filename, io.BytesIO(test_content), "text/plain"))]

    try:
        # 1. Upload & Index
        upload_resp = client.post("/api/documents/upload", files=files)
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert test_filename in upload_data["uploaded_files"]
        assert "sync_stats" in upload_data
        assert upload_data["sync_stats"]["sources_added"] >= 1

        # 2. Inspect document list
        list_resp = client.get("/api/documents")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        doc_names = [d["name"] for d in list_data["documents"]]
        assert test_filename in doc_names

        uploaded_doc = next(d for d in list_data["documents"] if d["name"] == test_filename)
        assert uploaded_doc["type"] == "txt"
        assert uploaded_doc["chunk_count"] >= 1
        assert uploaded_doc["size_bytes"] == len(test_content)

        # 3. Synchronize endpoint
        sync_resp = client.post("/api/documents/sync")
        assert sync_resp.status_code == 200
        sync_data = sync_resp.json()
        assert "sync_stats" in sync_data

    finally:
        # 4. Delete document and prune chunks
        delete_resp = client.delete(f"/api/documents/{test_filename}")
        assert delete_resp.status_code == 200
        delete_data = delete_resp.json()
        assert "purged" in delete_data["message"].lower() or "deleted" in delete_data["message"].lower()

        # 5. Verify document is gone from data directory and index
        post_delete_resp = client.get("/api/documents")
        post_delete_names = [d["name"] for d in post_delete_resp.json()["documents"]]
        assert test_filename not in post_delete_names


def test_delete_nonexistent_document_fails(client: TestClient):
    """Verify deleting a nonexistent document returns 404 Not Found."""
    response = client.delete("/api/documents/completely_nonexistent_file_9999.pdf")
    assert response.status_code == 404


def test_delete_path_traversal_blocked(client: TestClient):
    """Verify directory traversal attempts are blocked with 400 Bad Request."""
    response = client.delete("/api/documents/../../etc/passwd")
    assert response.status_code in [400, 404]
