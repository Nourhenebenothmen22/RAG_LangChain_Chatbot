"use client";

import { useState } from "react";
import { Check, Copy, ExternalLink, FileText, Layers, X } from "lucide-react";
import { RetrievedChunk } from "../lib/types";

interface SourceViewerDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  chunks: RetrievedChunk[];
  activeChunkIndex?: number;
}

export function SourceViewerDrawer({
  isOpen,
  onClose,
  chunks,
  activeChunkIndex = 0,
}: SourceViewerDrawerProps) {
  const [selectedIdx, setSelectedIdx] = useState<number>(activeChunkIndex);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!isOpen) return null;

  const currentChunk = chunks[selectedIdx] || chunks[0];

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end animate-fade-in">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Drawer Panel */}
      <aside className="relative w-full max-w-2xl h-full shadow-elevated flex flex-col z-10 border-l border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark text-ink-primaryLight dark:text-ink-primaryDark transition-colors">
        {/* Drawer Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-cobalt-DEFAULT dark:bg-cobalt-dark" />
              <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-ink-primaryLight dark:text-ink-primaryDark">
                EVIDENCE DOSSIER // EXCERPT INSPECTION
              </h2>
            </div>
            <p className="text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark mt-0.5">
              {chunks.length} PASSAGES RERANKED VIA CROSS-ENCODER
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 border border-border-light dark:border-border-dark hover:border-ink-primaryLight dark:hover:border-ink-primaryDark text-ink-mutedLight dark:text-ink-mutedDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Chunks Navigation Tabs */}
        <div className="flex overflow-x-auto px-6 py-2.5 space-x-2 border-b border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark scrollbar-none font-mono text-xs">
          {chunks.map((chunk, idx) => (
            <button
              key={chunk.chunk_id || idx}
              onClick={() => setSelectedIdx(idx)}
              className={`px-3 py-1.5 border transition-colors whitespace-nowrap flex items-center space-x-1.5 ${
                selectedIdx === idx
                  ? "border-cobalt-DEFAULT dark:border-cobalt-dark bg-cobalt-DEFAULT dark:bg-cobalt-dark text-white font-semibold"
                  : "border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:border-border-strongLight dark:hover:border-border-strongDark"
              }`}
            >
              <span>RANK #{chunk.rank}</span>
              <span className="text-xs opacity-80 font-mono">({chunk.score.toFixed(3)})</span>
            </button>
          ))}
        </div>

        {/* Selected Chunk Details */}
        {currentChunk ? (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Metadata Card */}
            <div className="border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark p-4 space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between pb-3 border-b border-border-light dark:border-border-dark">
                <div className="flex items-center space-x-2 truncate pr-2">
                  <FileText className="w-4 h-4 text-cobalt-DEFAULT dark:text-cobalt-dark flex-shrink-0" />
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark truncate">
                    {currentChunk.source ? currentChunk.source.split("/").pop() : "Unknown Document"}
                  </span>
                </div>
                <button
                  onClick={() => handleCopy(currentChunk.content, currentChunk.chunk_id)}
                  className="flex items-center space-x-1.5 px-2.5 py-1 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark hover:border-cobalt-DEFAULT text-ink-secondaryLight dark:text-ink-secondaryDark transition-colors flex-shrink-0"
                >
                  {copiedId === currentChunk.chunk_id ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-DEFAULT" />
                      <span className="text-emerald-DEFAULT">COPIED</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>COPY TEXT</span>
                    </>
                  )}
                </button>
              </div>

              {/* Technical Score Breakdown */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-xs font-mono">
                <div>
                  <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
                    FINAL RELEVANCE
                  </span>
                  <span className="font-bold text-cobalt-DEFAULT dark:text-cobalt-dark">
                    {currentChunk.score.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
                    RERANK SCORE
                  </span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    {currentChunk.rerank_score != null
                      ? currentChunk.rerank_score.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
                    RRF RECIPROCAL
                  </span>
                  <span className="font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                    {currentChunk.rrf_score != null
                      ? currentChunk.rrf_score.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
                    CHUNK HASH ID
                  </span>
                  <span className="font-semibold text-ink-mutedLight dark:text-ink-mutedDark truncate block max-w-[100px]">
                    {currentChunk.chunk_id ? currentChunk.chunk_id.slice(0, 10) : "N/A"}…
                  </span>
                </div>
              </div>
            </div>

            {/* Verbatim Content Excerpt */}
            <div>
              <div className="flex items-center justify-between mb-2 text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark">
                <span>VERBATIM PASSAGE TEXT</span>
                <span>{currentChunk.content.length} CHARS</span>
              </div>
              <div className="p-4 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark font-mono text-xs leading-relaxed text-ink-primaryLight dark:text-ink-primaryDark whitespace-pre-wrap select-text">
                {currentChunk.content}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark">
            NO PASSAGES RETRIEVED
          </div>
        )}
      </aside>
    </div>
  );
}
