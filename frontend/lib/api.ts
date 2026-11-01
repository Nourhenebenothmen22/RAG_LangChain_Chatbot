/** Typed HTTP and SSE API Client for ClaimEye Hybrid RAG Backend */

import {
  DocumentDeleteResponse,
  DocumentsListResponse,
  DocumentSyncResponse,
  DocumentUploadResponse,
  HealthStatus,
  QueryResponse,
  RetrievedChunk,
  StreamEvent,
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class RAGApiClient {
  /** Check backend health status and active configuration */
  static async getHealth(): Promise<HealthStatus> {
    try {
      const res = await fetch(`${API_BASE}/api/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        throw new Error(`Health check failed with status ${res.status}`);
      }
      return await res.json();
    } catch (err: any) {
      console.warn("Backend API unreachable:", err.message);
      throw err;
    }
  }

  /** Submit standard JSON query to the frozen Hybrid RAG pipeline */
  static async submitQuery(params: {
    query: string;
    top_k?: number;
    rerank?: boolean;
    generate_answer?: boolean;
    conversation_id?: string;
  }): Promise<QueryResponse> {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: params.query,
        top_k: params.top_k,
        rerank: params.rerank ?? true,
        generate_answer: params.generate_answer ?? true,
        conversation_id: params.conversation_id,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Query failed with status ${res.status}`);
    }

    return await res.json();
  }

  /** Stream response tokens and source chunks via Server-Sent Events (SSE) */
  static async streamQuery(
    params: {
      query: string;
      top_k?: number;
      rerank?: boolean;
      conversation_id?: string;
    },
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        query: params.query,
        top_k: params.top_k,
        rerank: params.rerank ?? true,
        conversation_id: params.conversation_id,
      }),
      signal,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Stream failed with status ${res.status}`);
    }

    if (!res.body) {
      throw new Error("ReadableStream not supported by browser environment.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Retain last incomplete line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data:")) continue;

          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;

          try {
            const parsedEvent: StreamEvent = JSON.parse(jsonStr);
            onEvent(parsedEvent);
          } catch (jsonErr) {
            console.error("Failed to parse SSE JSON payload:", jsonStr, jsonErr);
          }
        }
      }

      // Process any remaining bytes
      if (buffer.trim().startsWith("data:")) {
        const jsonStr = buffer.trim().slice(5).trim();
        try {
          const parsedEvent: StreamEvent = JSON.parse(jsonStr);
          onEvent(parsedEvent);
        } catch (_) {}
      }
    } finally {
      reader.releaseLock();
    }
  }

  /** Retrieve all indexed documents and sync status */
  static async getDocuments(): Promise<DocumentsListResponse> {
    const res = await fetch(`${API_BASE}/api/documents`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Failed to fetch documents (${res.status})`);
    }
    return await res.json();
  }

  /** Upload one or multiple documents to knowledge base */
  static async uploadDocuments(files: File[]): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    const res = await fetch(`${API_BASE}/api/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed (${res.status})`);
    }
    return await res.json();
  }

  /** Delete a document and purge its chunks */
  static async deleteDocument(filename: string): Promise<DocumentDeleteResponse> {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Failed to delete document (${res.status})`);
    }
    return await res.json();
  }

  /** Trigger manual re-synchronization of documents */
  static async syncDocuments(): Promise<DocumentSyncResponse> {
    const res = await fetch(`${API_BASE}/api/documents/sync`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Sync failed (${res.status})`);
    }
    return await res.json();
  }
}
