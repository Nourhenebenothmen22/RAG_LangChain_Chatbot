"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Database,
  FileCheck2,
  FileText,
  Layers,
  Scale,
  ShieldAlert,
} from "lucide-react";
import { RAGApiClient } from "../lib/api";
import {
  ChatMessage,
  ConversationSession,
  HealthStatus,
  RetrievedChunk,
  StreamEvent,
} from "../lib/types";
import { DocumentManagerModal } from "./DocumentManagerModal";
import { Header } from "./Header";
import { InputComposer } from "./InputComposer";
import { MessageBubble } from "./MessageBubble";
import { Sidebar } from "./Sidebar";
import { SourceViewerDrawer } from "./SourceViewerDrawer";

const SUGGESTED_INQUIRIES = [
  {
    num: "01",
    category: "FRAUD RISK POLICY",
    title: "Fraud Escalation Threshold",
    query: "What is the fraud score threshold for human-in-the-loop review?",
    desc: "Inspect strict probability boundaries for automated claims suspension.",
  },
  {
    num: "02",
    category: "FRENCH BARÈME",
    title: "Glass Breakage Deductibles",
    query: "Quel est le montant de la franchise bris de glace selon le barème ?",
    desc: "Extract official schedule values and conditional deductible exclusions.",
  },
  {
    num: "03",
    category: "DETERMINISTIC CODE",
    title: "Damage Quantification Math",
    query: "Comment est calculé le chiffrage déterministe des dommages en Python ?",
    desc: "Verify rule-based algorithms for vehicle component repair estimates.",
  },
  {
    num: "04",
    category: "INSTANT BYPASS",
    title: "Conversational Screening",
    query: "Bonjour ! Qui es-tu et quelles sont tes capacités d'analyse ?",
    desc: "Demonstrate 0ms retrieval avoidance on conversational greetings.",
  },
];

export function ChatWorkspace() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState<ConversationSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerChunks, setDrawerChunks] = useState<RetrievedChunk[]>([]);
  const [docModalOpen, setDocModalOpen] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize and load saved sessions from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("kortex-chat-sessions") || localStorage.getItem("claimeye-chat-sessions");
      if (saved) {
        const parsed: ConversationSession[] = JSON.parse(saved);
        setSessions(parsed);
        if (parsed.length > 0) {
          setActiveSessionId(parsed[0].id);
          setMessages(parsed[0].messages || []);
        } else {
          createNewSession();
        }
      } else {
        createNewSession();
      }
    } catch (_) {
      createNewSession();
    }

    // Check backend health
    RAGApiClient.getHealth()
      .then((data) => setHealth(data))
      .catch((err) => console.warn("Backend not yet reachable on startup:", err));
  }, []);

  // Save sessions to localStorage whenever messages update
  useEffect(() => {
    if (!activeSessionId) return;
    setSessions((prev) => {
      const updated = prev.map((s) => {
        if (s.id === activeSessionId) {
          const title =
            messages.length > 0
              ? messages[0].content.slice(0, 36) + (messages[0].content.length > 36 ? "..." : "")
              : "New Inquiry";
          return { ...s, title, messages, updatedAt: Date.now() };
        }
        return s;
      });
      try {
        localStorage.setItem("kortex-chat-sessions", JSON.stringify(updated));
      } catch (_) {}
      return updated;
    });
  }, [messages, activeSessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const createNewSession = () => {
    const newSession: ConversationSession = {
      id: `session_${Date.now()}`,
      title: "New Inquiry",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setMessages([]);
  };

  const selectSession = (id: string) => {
    const s = sessions.find((item) => item.id === id);
    if (s) {
      setActiveSessionId(s.id);
      setMessages(s.messages || []);
    }
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const remaining = sessions.filter((s) => s.id !== id);
    setSessions(remaining);
    try {
      localStorage.setItem("kortex-chat-sessions", JSON.stringify(remaining));
    } catch (_) {}

    if (activeSessionId === id) {
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id);
        setMessages(remaining[0].messages || []);
      } else {
        createNewSession();
      }
    }
  };

  const clearChat = () => {
    setMessages([]);
    if (activeSessionId) {
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, messages: [], title: "New Inquiry" } : s))
      );
    }
  };

  const openDrawerWithChunks = (chunks: RetrievedChunk[]) => {
    setDrawerChunks(chunks);
    setDrawerOpen(true);
  };

  const handleSendMessage = async (
    query: string,
    options: { topK: number; rerank: boolean }
  ) => {
    if (!query.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `usr_${Date.now()}`,
      role: "user",
      content: query,
      timestamp: Date.now(),
    };

    const assistantMsgId = `asst_${Date.now() + 1}`;
    const assistantPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: Date.now() + 1,
      isStreaming: true,
      metadata: {},
      chunks: [],
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await RAGApiClient.streamQuery(
        {
          query,
          top_k: options.topK,
          rerank: options.rerank,
          conversation_id: activeSessionId || undefined,
        },
        (event: StreamEvent) => {
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantMsgId) return m;

              if (event.type === "meta") {
                return {
                  ...m,
                  metadata: {
                    ...m.metadata,
                    intent: event.intent,
                    language: event.language,
                    retrievalAvoided: event.retrieval_avoided,
                  },
                };
              }

              if (event.type === "sources") {
                return {
                  ...m,
                  chunks: event.chunks,
                  metadata: {
                    ...m.metadata,
                    retrievalLatencyMs: event.retrieval_latency_ms,
                  },
                };
              }

              if (event.type === "token") {
                return {
                  ...m,
                  content: m.content + event.content,
                };
              }

              if (event.type === "done") {
                return {
                  ...m,
                  content: event.answer || m.content,
                  isStreaming: false,
                  chunks: event.chunks || m.chunks,
                  metadata: {
                    ...m.metadata,
                    totalLatencyMs: event.total_latency_ms || event.latency_ms,
                    retrievalAvoided: event.retrieval_avoided ?? m.metadata?.retrievalAvoided,
                  },
                };
              }

              if (event.type === "error") {
                return {
                  ...m,
                  isStreaming: false,
                  error: event.message,
                  metadata: {
                    ...m.metadata,
                    securityRisk: event.security_risk,
                  },
                };
              }

              return m;
            })
          );
        },
        controller.signal
      );
    } catch (err: any) {
      if (err.name === "AbortError") {
        console.log("Stream aborted by user.");
      } else {
        console.error("Query streaming failed:", err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  isStreaming: false,
                  error: err.message || "Failed to communicate with RAG backend.",
                }
              : m
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas-light dark:bg-canvas-dark text-ink-primaryLight dark:text-ink-primaryDark transition-colors">
      {/* Sidebar Rail */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={selectSession}
        onNewSession={createNewSession}
        onDeleteSession={deleteSession}
        onOpenDocuments={() => setDocModalOpen(true)}
        health={health}
      />

      {/* Main Workspace Column */}
      <div className="flex-1 flex flex-col h-full relative overflow-hidden">
        {/* Navigation Masthead */}
        <Header
          onToggleSidebar={() => setSidebarOpen(true)}
          onClearChat={clearChat}
          onOpenDocuments={() => setDocModalOpen(true)}
          hasMessages={messages.length > 0}
        />

        {/* Scrollable Conversation Stream */}
        <main className="flex-1 overflow-y-auto px-4 sm:px-8 py-8 max-w-4xl w-full mx-auto">
          {messages.length === 0 ? (
            /* Swiss Editorial Welcome Section */
            <div className="py-6 sm:py-12 animate-fade-in">
              {/* Technical Issue Tag */}
              <div className="flex items-center justify-between border-b border-border-light dark:border-border-dark pb-3 mb-6 text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark">
                <span>SYSTEM EDITION 02.4 // KORTEX HYBRID RAG</span>
                <span>CHROMA + BM25 + CROSS-ENCODER</span>
              </div>

              {/* Oversized Brand Masthead */}
              <div className="mb-8">
                <div className="flex items-baseline space-x-3 mb-2">
                  <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tighter uppercase text-ink-primaryLight dark:text-ink-primaryDark font-mono">
                    KORTEX
                  </h1>
                  <span className="text-cobalt-DEFAULT dark:text-cobalt-dark font-mono text-xl sm:text-2xl font-bold">
                    [AI]
                  </span>
                </div>

                <p className="text-base sm:text-lg text-ink-secondaryLight dark:text-ink-secondaryDark max-w-2xl leading-relaxed">
                  Autonomous multilingual document intelligence and hybrid retrieval-augmented generation engine.
                  Formulate inquiries across policies, contracts, specifications, and verify grounded source citations.
                </p>
              </div>

              {/* Numbered Query Section Grid */}
              <div className="mt-8">
                <div className="flex items-center justify-between text-[11px] font-mono tracking-widest text-ink-mutedLight dark:text-ink-mutedDark uppercase border-b border-border-light dark:border-border-dark pb-2 mb-4">
                  <span>01 // RECOMMENDED INQUIRIES</span>
                  <span>SELECT DOSSIER PROMPT</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {SUGGESTED_INQUIRIES.map((sq) => (
                    <button
                      key={sq.num}
                      onClick={() => handleSendMessage(sq.query, { topK: 3, rerank: true })}
                      className="border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark p-4 text-left hover:border-cobalt-DEFAULT dark:hover:border-cobalt-dark transition-colors group flex flex-col justify-between shadow-subtle"
                    >
                      <div>
                        <div className="flex items-center justify-between text-[10px] font-mono mb-2">
                          <span className="text-cobalt-DEFAULT dark:text-cobalt-dark font-bold">
                            [{sq.num}] {sq.category}
                          </span>
                          <span className="text-ink-mutedLight dark:text-ink-mutedDark group-hover:text-cobalt-DEFAULT dark:group-hover:text-cobalt-dark transition-colors">
                            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                          </span>
                        </div>
                        <h3 className="font-bold text-sm text-ink-primaryLight dark:text-ink-primaryDark mb-1 font-mono">
                          {sq.title}
                        </h3>
                        <p className="text-xs font-sans text-ink-secondaryLight dark:text-ink-secondaryDark line-clamp-2 leading-relaxed">
                          {sq.desc}
                        </p>
                      </div>

                      <div className="mt-3 pt-2.5 border-t border-border-light dark:border-border-dark text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark truncate">
                        "{sq.query}"
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Architecture Guarantee Footer */}
              <div className="mt-10 p-4 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono text-ink-secondaryLight dark:text-ink-secondaryDark">
                <div className="flex items-center space-x-2">
                  <Scale className="w-4 h-4 text-cobalt-DEFAULT dark:text-cobalt-dark" />
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    GROUNDED ACCURACY GUARANTEE:
                  </span>
                  <span>Zero hallucination via strict context boundary checks.</span>
                </div>
                <button
                  onClick={() => setDocModalOpen(true)}
                  className="text-cobalt-DEFAULT dark:text-cobalt-dark hover:underline flex items-center space-x-1"
                >
                  <span>MANAGE CORPUS</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          ) : (
            /* Active Conversation Bubbles */
            <div className="space-y-4">
              {messages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  onViewSources={openDrawerWithChunks}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Input Composer Dock */}
        <footer className="w-full flex-shrink-0 z-20">
          <InputComposer
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            onStopStreaming={handleStopStreaming}
          />
        </footer>
      </div>

      {/* Source Excerpts Drawer */}
      <SourceViewerDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        chunks={drawerChunks}
      />

      {/* Knowledge Base Modal */}
      <DocumentManagerModal
        isOpen={docModalOpen}
        onClose={() => setDocModalOpen(false)}
      />
    </div>
  );
}
