"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import remarkGfm from "remark-gfm";
import {
  Check,
  CheckCircle2,
  Copy,
  ExternalLink,
  FileText,
  Layers,
  Zap,
} from "lucide-react";
import { ChatMessage, RetrievedChunk } from "../lib/types";

const ReactMarkdown = dynamic(() => import("react-markdown"), {
  ssr: false,
  loading: () => (
    <div className="text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark">
      Rendering analysis...
    </div>
  ),
});

interface MessageBubbleProps {
  message: ChatMessage;
  onViewSources?: (chunks: RetrievedChunk[]) => void;
}

export function MessageBubble({ message, onViewSources }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const meta = message.metadata;
  const chunks = message.chunks || [];
  const hasSources = chunks.length > 0;
  const timeStr = new Date(message.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="w-full mb-8 animate-fade-in">
      {isUser ? (
        /* USER INQUIRY BLOCK */
        <div className="max-w-3xl ml-auto border border-border-strongLight dark:border-border-strongDark bg-surface-light dark:bg-surface-dark shadow-subtle">
          {/* User Header */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark">
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark uppercase">
                INQUIRY // CLIENT
              </span>
              <span>•</span>
              <span>{timeStr}</span>
            </div>
            <button
              onClick={handleCopy}
              className="hover:text-ink-primaryLight dark:hover:text-ink-primaryDark transition-colors"
              title="Copy Query"
            >
              {copied ? <Check className="w-3 h-3 text-emerald-DEFAULT" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>

          {/* User Content */}
          <div className="p-4 text-sm md:text-base font-sans text-ink-primaryLight dark:text-ink-primaryDark leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
        </div>
      ) : (
        /* ASSISTANT DOSSIER BLOCK */
        <div className="max-w-4xl mr-auto border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark shadow-subtle">
          {/* Assistant Header & Masthead */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-cobalt-DEFAULT dark:bg-cobalt-dark" />
              <span className="font-bold text-ink-primaryLight dark:text-ink-primaryDark uppercase tracking-wider">
                CLAIMEYE // SYNTHESIS
              </span>
              <span>•</span>
              <span>{timeStr}</span>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleCopy}
                className="flex items-center space-x-1 hover:text-ink-primaryLight dark:hover:text-ink-primaryDark transition-colors"
                title="Copy Response"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-DEFAULT" />
                    <span className="text-emerald-DEFAULT">COPIED</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>COPY</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Assistant Telemetry Grid (Swiss Metadata Bar) */}
          {meta && (
            <div className="px-4 py-2 border-b border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-ink-secondaryLight dark:text-ink-secondaryDark">
              {meta.intent && (
                <div>
                  <span className="text-ink-mutedLight dark:text-ink-mutedDark mr-1">INTENT:</span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark uppercase">
                    {meta.intent.replace(/_/g, " ")}
                  </span>
                </div>
              )}

              {meta.language && (
                <div>
                  <span className="text-ink-mutedLight dark:text-ink-mutedDark mr-1">LANG:</span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark uppercase">
                    {meta.language}
                  </span>
                </div>
              )}

              {meta.totalLatencyMs !== undefined && (
                <div>
                  <span className="text-ink-mutedLight dark:text-ink-mutedDark mr-1">LATENCY:</span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    {Math.round(meta.totalLatencyMs)}ms
                  </span>
                </div>
              )}

              {meta.totalTokens !== undefined && meta.totalTokens > 0 && (
                <div>
                  <span className="text-ink-mutedLight dark:text-ink-mutedDark mr-1">TOKENS:</span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    {meta.totalTokens.toLocaleString()}
                  </span>
                </div>
              )}

              {meta.estimatedCostUsd !== undefined && meta.estimatedCostUsd > 0 && (
                <div>
                  <span className="text-ink-mutedLight dark:text-ink-mutedDark mr-1">EST. COST:</span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    ${meta.estimatedCostUsd.toFixed(4)}
                  </span>
                </div>
              )}

              {meta.retrievalAvoided && (
                <div className="text-emerald-DEFAULT font-semibold flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>CONVERSATIONAL BYPASS (0ms retrieval)</span>
                </div>
              )}
            </div>
          )}

          {/* Assistant Prose Body */}
          <div className="p-6 editorial-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Assistant Grounded Sources Evidence Dossier */}
          {hasSources && (
            <div className="border-t border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark p-4 sm:p-5">
              <div className="flex items-center justify-between mb-3 text-xs font-mono">
                <div className="flex items-center space-x-2 text-ink-primaryLight dark:text-ink-primaryDark font-bold tracking-wider uppercase">
                  <Layers className="w-3.5 h-3.5 text-cobalt-DEFAULT dark:text-cobalt-dark" />
                  <span>02 // GROUNDED EVIDENCE DOSSIER ({chunks.length} PASSAGES)</span>
                </div>

                <button
                  onClick={() => onViewSources && onViewSources(chunks)}
                  className="text-cobalt-DEFAULT dark:text-cobalt-dark hover:underline flex items-center space-x-1"
                >
                  <span>INSPECT ALL IN DRAWER</span>
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {chunks.map((chunk, idx) => {
                  const filename = chunk.source ? chunk.source.split("/").pop() : "Corpus Document";
                  return (
                    <div
                      key={chunk.chunk_id || idx}
                      onClick={() => onViewSources && onViewSources(chunks)}
                      className="border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark p-3 text-xs font-mono hover:border-cobalt-DEFAULT dark:hover:border-cobalt-dark transition-colors cursor-pointer group"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark flex items-center space-x-1">
                          <span className="text-cobalt-DEFAULT dark:text-cobalt-dark">
                            [REF #{String(chunk.rank).padStart(2, "0")}]
                          </span>
                          <span className="truncate max-w-[140px] text-ink-secondaryLight dark:text-ink-secondaryDark">
                            {filename}
                          </span>
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-ink-mutedLight dark:text-ink-mutedDark">
                          SCORE {chunk.score.toFixed(3)}
                        </span>
                      </div>

                      <p className="font-sans text-ink-secondaryLight dark:text-ink-secondaryDark line-clamp-3 leading-relaxed text-xs">
                        "{chunk.content}"
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
