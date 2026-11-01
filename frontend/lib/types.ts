/** Type definitions for ClaimEye Hybrid RAG Next.js Client */

export interface RetrievedChunk {
  rank: number;
  content: string;
  score: number;
  source: string;
  page?: number | null;
  record_id?: string | null;
  chunk_id: string;
  dense_rank?: number | null;
  dense_score?: number | null;
  sparse_rank?: number | null;
  sparse_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  metadata?: Record<string, any>;
}

export interface QueryResponse {
  query: string;
  answer: string;
  chunks: RetrievedChunk[];
  is_grounded: boolean;
  intent: string;
  detected_language: string;
  retrieval_avoided: boolean;
  total_latency_ms: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  model_name: string;
  reranked: boolean;
  security_risk: string;
  flagged_security_patterns: string[];
  error?: string | null;
  conversation_id?: string | null;
}

export interface ChatMessageMetadata {
  intent?: string;
  language?: string;
  retrievalAvoided?: boolean;
  totalLatencyMs?: number;
  retrievalLatencyMs?: number;
  generationLatencyMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  estimatedCostUsd?: number;
  modelName?: string;
  securityRisk?: string;
  flaggedPatterns?: string[];
  isGrounded?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  chunks?: RetrievedChunk[];
  metadata?: ChatMessageMetadata;
  isStreaming?: boolean;
  error?: string;
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface HealthStatus {
  status: string;
  version: string;
  llm_model: string;
  embedding_model: string;
  reranker_model: string;
  hybrid_search_enabled: boolean;
  chroma_dir: string;
  collection_name: string;
}

export type StreamEvent =
  | { type: "meta"; intent: string; language: string; retrieval_avoided: boolean; confidence?: number }
  | { type: "sources"; chunks: RetrievedChunk[]; retrieval_latency_ms: number }
  | { type: "token"; content: string }
  | { type: "done"; chunks?: RetrievedChunk[]; retrieval_avoided?: boolean; total_latency_ms?: number; latency_ms?: number; answer?: string }
  | { type: "error"; message: string; security_risk?: string };

export interface DocumentItem {
  name: string;
  type: string;
  size_bytes: number;
  chunk_count: number;
  last_synced_at?: string | null;
}

export interface DocumentsListResponse {
  documents: DocumentItem[];
  total_documents: number;
  total_chunks: number;
}

export interface DocumentUploadResponse {
  message: string;
  uploaded_files: string[];
  sync_stats: Record<string, any>;
}

export interface DocumentDeleteResponse {
  message: string;
  filename: string;
  sync_stats: Record<string, any>;
}

export interface DocumentSyncResponse {
  message: string;
  sync_stats: Record<string, any>;
}
