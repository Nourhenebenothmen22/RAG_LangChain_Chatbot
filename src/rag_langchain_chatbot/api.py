"""FastAPI & LangServe API transport layer exposing the Hybrid RAG pipeline.

Adheres strictly to SOLID principles and Clean Architecture:
Acts exclusively as an HTTP transport adapter delegating all RAG operations
(batch query and token streaming) to the authoritative HybridRAGPipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_langchain_chatbot.pipeline import (
    HybridRAGPipeline,
    HybridRAGResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Pydantic Request & Response Data Transfer Objects (DTOs)
# ==============================================================================

class QueryRequest(BaseModel):
    """Query submission payload."""
    query: str = Field(..., min_length=1, description="User query or message")
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Prior conversational turns as [{role/type: 'user'|'assistant', content: str}]"
    )
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override top-k results")
    rerank: bool = Field(True, description="Whether to apply Cross-Encoder reranking")
    generate_answer: bool = Field(True, description="Whether to generate grounded LLM answer")
    conversation_id: Optional[str] = Field(None, description="Client conversation tracking ID")


class RetrievedChunkModel(BaseModel):
    """Retrieved document chunk with provenance metadata."""
    rank: int
    content: str
    score: float
    source: str
    page: Optional[int] = None
    record_id: Optional[str] = None
    chunk_id: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(cls, c: RetrievedChunk) -> "RetrievedChunkModel":
        """Factory method mapping internal domain chunk to API response DTO."""
        return cls(
            rank=c.rank,
            content=c.content,
            score=c.score,
            source=c.source,
            page=c.page,
            record_id=c.record_id,
            chunk_id=c.chunk_id,
            dense_rank=c.dense_rank,
            dense_score=c.dense_score,
            sparse_rank=c.sparse_rank,
            sparse_score=c.sparse_score,
            rrf_score=c.rrf_score,
            rerank_score=c.rerank_score,
            metadata=c.metadata,
        )


class QueryResponseModel(BaseModel):
    """Full API response matching backend HybridRAGResponse semantics."""
    query: str
    standalone_query: Optional[str] = None
    answer: str
    chunks: List[RetrievedChunkModel]
    is_grounded: bool
    intent: str
    detected_language: str
    retrieval_avoided: bool
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model_name: str
    reranked: bool
    security_risk: str
    flagged_security_patterns: List[str]
    error: Optional[str] = None
    conversation_id: Optional[str] = None

    @classmethod
    def from_response(
        cls,
        response: HybridRAGResponse,
        conversation_id: Optional[str] = None
    ) -> "QueryResponseModel":
        """Factory method mapping internal domain response to API response DTO."""
        return cls(
            query=response.query,
            standalone_query=response.standalone_query,
            answer=response.answer,
            chunks=[RetrievedChunkModel.from_chunk(c) for c in response.chunks],
            is_grounded=response.is_grounded,
            intent=response.intent,
            detected_language=response.detected_language,
            retrieval_avoided=response.retrieval_avoided,
            total_latency_ms=response.total_latency_ms,
            retrieval_latency_ms=response.retrieval_latency_ms,
            generation_latency_ms=response.generation_latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            estimated_cost_usd=response.estimated_cost_usd,
            model_name=response.model_name,
            reranked=response.reranked,
            security_risk=response.security_risk,
            flagged_security_patterns=response.flagged_security_patterns,
            error=response.error,
            conversation_id=conversation_id,
        )


class HealthResponse(BaseModel):
    """Health check and configuration inspection status."""
    status: str = "ok"
    version: str = "0.1.0"
    llm_model: str
    embedding_model: str
    reranker_model: str
    hybrid_search_enabled: bool
    chroma_dir: str
    collection_name: str


class DocumentItemModel(BaseModel):
    """Metadata representing an indexed or ingested document."""
    name: str
    type: str
    size_bytes: int
    chunk_count: int
    last_synced_at: Optional[str] = None


class DocumentsListResponse(BaseModel):
    """Document ledger listing response."""
    documents: List[DocumentItemModel]
    total_documents: int
    total_chunks: int


class DocumentUploadResponse(BaseModel):
    """File upload and incremental ingestion response."""
    message: str
    uploaded_files: List[str]
    sync_stats: Dict[str, Any]


class DocumentDeleteResponse(BaseModel):
    """Document deletion and index purge response."""
    message: str
    filename: str
    sync_stats: Dict[str, Any]


class DocumentSyncResponse(BaseModel):
    """Manual synchronization response."""
    message: str
    sync_stats: Dict[str, Any]


# ==============================================================================
# 2. FastAPI Application Initialization
# ==============================================================================

app = FastAPI(
    title="ClaimEye Hybrid RAG API",
    description="High-performance enterprise Hybrid RAG pipeline with pure LLM semantic routing and cross-encoder reranking",
    version="0.1.0",
)

# Enable CORS for Next.js frontend and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton
_PIPELINE: Optional[HybridRAGPipeline] = None


def get_pipeline() -> HybridRAGPipeline:
    """Retrieve or initialize the singleton HybridRAGPipeline instance."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = HybridRAGPipeline()
    return _PIPELINE


# Mount native LangServe LCEL routes if available
try:
    from langserve import add_routes
    add_routes(
        app,
        get_pipeline().as_runnable(),
        path="/rag",
        enable_feedback_endpoint=False,
    )
    logger.info("Mounted native LangServe LCEL routes at /rag")
except Exception as e:
    logger.warning("LangServe mounting deferred or unsupported in this mode: %s", e)


# ==============================================================================
# 3. HTTP Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check system health, pipeline readiness, and model configurations."""
    pipeline = get_pipeline()
    cfg = pipeline.settings
    return HealthResponse(
        status="ok",
        version="0.1.0",
        llm_model=cfg.llm_model_name,
        embedding_model=cfg.embedding_model_name,
        reranker_model=cfg.reranker_model_name,
        hybrid_search_enabled=True,
        chroma_dir=str(cfg.chroma_dir),
        collection_name=cfg.collection_name,
    )


@app.post("/api/query", response_model=QueryResponseModel)
def submit_query(payload: QueryRequest) -> QueryResponseModel:
    """Submit a query to the Hybrid RAG pipeline and return grounded results."""
    pipeline = get_pipeline()
    try:
        response: HybridRAGResponse = pipeline.query(
            query_text=payload.query,
            chat_history=payload.chat_history,
            top_k=payload.top_k,
            rerank=payload.rerank,
            generate_answer=payload.generate_answer,
        )
        return QueryResponseModel.from_response(
            response=response,
            conversation_id=payload.conversation_id
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error executing pipeline query: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(payload: QueryRequest) -> StreamingResponse:
    """Stream token responses and source attribution via Server-Sent Events (SSE)."""
    pipeline = get_pipeline()

    async def event_generator() -> AsyncGenerator[str, None]:
        for event in pipeline.stream(
            query_text=payload.query,
            chat_history=payload.chat_history,
            top_k=payload.top_k,
            rerank=payload.rerank,
            generate_answer=payload.generate_answer,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==============================================================================
# 4. Document Management & Synchronization Endpoints
# ==============================================================================

MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024  # 50MB per file limit


@app.get("/api/documents", response_model=DocumentsListResponse)
def list_documents() -> DocumentsListResponse:
    """Retrieve list of all tracked documents with chunk counts and sync metadata."""
    pipeline = get_pipeline()
    ledger_records = pipeline.indexer.ledger.get_all_records()

    documents: List[DocumentItemModel] = []
    seen_paths = set()
    total_chunks = 0

    # 1. Inspect data directory for active files
    if pipeline.settings.data_dir.exists():
        for f in pipeline.settings.data_dir.rglob("*"):
            if not f.is_file() or f.name.startswith("."):
                continue
            if f.suffix.lower() not in pipeline.settings.supported_extensions:
                continue

            resolved_key = str(f.resolve())
            seen_paths.add(resolved_key)
            record = ledger_records.get(resolved_key)

            chunk_count = record["chunk_count"] if record else 0
            last_synced = record["last_synced_at"] if record else None
            total_chunks += chunk_count

            try:
                display_name = str(f.relative_to(pipeline.settings.data_dir))
            except ValueError:
                display_name = f.name

            documents.append(
                DocumentItemModel(
                    name=display_name,
                    type=f.suffix.lstrip(".").lower() or "file",
                    size_bytes=f.stat().st_size,
                    chunk_count=chunk_count,
                    last_synced_at=last_synced,
                )
            )

    # 2. Add any tracked sources from ledger that may not reside directly in data_dir (e.g. database sources)
    for src_id, record in ledger_records.items():
        if src_id not in seen_paths:
            src_path = Path(src_id)
            size = src_path.stat().st_size if src_path.is_file() else 0
            chunk_cnt = record.get("chunk_count", 0)
            total_chunks += chunk_cnt
            documents.append(
                DocumentItemModel(
                    name=src_path.name,
                    type=record.get("source_type", "database"),
                    size_bytes=size,
                    chunk_count=chunk_cnt,
                    last_synced_at=record.get("last_synced_at"),
                )
            )

    documents.sort(key=lambda d: d.name.lower())
    return DocumentsListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks,
    )


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    """Accept document uploads, write them safely into the data directory, and trigger incremental indexing."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    pipeline = get_pipeline()
    data_dir = pipeline.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    supported_exts = set(ext.lower() for ext in pipeline.settings.supported_extensions)

    uploaded_names: List[str] = []

    for file in files:
        if not file.filename:
            continue

        raw_name = Path(file.filename).name  # Prevent directory traversal attacks
        suffix = Path(raw_name).suffix.lower()

        if suffix not in supported_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{suffix}' for file '{raw_name}'. Allowed formats: {', '.join(sorted(supported_exts))}",
            )

        target_file = data_dir / raw_name

        # Stream file to disk while enforcing max size
        size = 0
        try:
            with target_file.open("wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB buffer
                    size += len(chunk)
                    if size > MAX_DOCUMENT_SIZE_BYTES:
                        target_file.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{raw_name}' exceeds maximum allowed size of 50MB.",
                        )
                    buffer.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            target_file.unlink(missing_ok=True)
            logger.exception("Failed to write uploaded file %s: %s", raw_name, e)
            raise HTTPException(status_code=500, detail=f"Failed to save file '{raw_name}': {str(e)}")

        uploaded_names.append(raw_name)

    # Trigger incremental index update
    try:
        sync_stats = pipeline.index(force_reindex=False)
    except Exception as e:
        logger.exception("Error indexing uploaded documents: %s", e)
        raise HTTPException(status_code=500, detail=f"Indexing failed post-upload: {str(e)}")

    return DocumentUploadResponse(
        message=f"Successfully uploaded and indexed {len(uploaded_names)} document(s).",
        uploaded_files=uploaded_names,
        sync_stats=sync_stats,
    )


@app.delete("/api/documents/{filename:path}", response_model=DocumentDeleteResponse)
def delete_document(filename: str) -> DocumentDeleteResponse:
    """Delete a document from data directory and automatically prune its indexed chunks."""
    pipeline = get_pipeline()
    data_dir = pipeline.settings.data_dir.resolve()

    # Prevent directory traversal
    target_path = (data_dir / filename).resolve()
    if data_dir not in target_path.parents and target_path != data_dir:
        raise HTTPException(status_code=400, detail="Invalid path traversal attempt.")

    # Locate target file
    if not target_path.is_file():
        safe_name = Path(filename).name
        alt_path = (data_dir / safe_name).resolve()
        if alt_path.is_file():
            target_path = alt_path
        else:
            found = False
            for match in data_dir.rglob(safe_name):
                if match.is_file():
                    target_path = match.resolve()
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Document '{filename}' not found on server.")

    try:
        target_path.unlink()
    except Exception as e:
        logger.exception("Failed to delete file %s: %s", target_path, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{filename}': {str(e)}")

    # Incremental sync automatically handles deletions
    try:
        sync_stats = pipeline.index(force_reindex=False)
    except Exception as e:
        logger.exception("Index sync failed after file deletion: %s", e)
        raise HTTPException(status_code=500, detail=f"Index sync failed after deletion: {str(e)}")

    return DocumentDeleteResponse(
        message=f"Document '{filename}' successfully deleted and purged from all indexes.",
        filename=filename,
        sync_stats=sync_stats,
    )


@app.post("/api/documents/sync", response_model=DocumentSyncResponse)
def synchronize_documents() -> DocumentSyncResponse:
    """Trigger manual synchronization of all documents in the data directory."""
    pipeline = get_pipeline()
    try:
        sync_stats = pipeline.index(force_reindex=False)
        return DocumentSyncResponse(
            message="Document synchronization completed successfully.",
            sync_stats=sync_stats,
        )
    except Exception as e:
        logger.exception("Manual document synchronization failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Synchronization failed: {str(e)}")
