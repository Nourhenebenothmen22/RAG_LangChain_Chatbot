"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, CornerDownLeft, SlidersHorizontal, Square } from "lucide-react";

interface InputComposerProps {
  onSendMessage: (query: string, options: { topK: number; rerank: boolean }) => void;
  isLoading: boolean;
  onStopStreaming?: () => void;
}

export function InputComposer({
  onSendMessage,
  isLoading,
  onStopStreaming,
}: InputComposerProps) {
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState<number>(3);
  const [rerank, setRerank] = useState<boolean>(true);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    onSendMessage(input.trim(), { topK, rerank });
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-6 pt-2">
      {/* Configuration Strip (Technical Swiss Bar) */}
      <div className="mb-2 flex items-center justify-between text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark">
        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            className={`flex items-center space-x-1.5 px-2 py-0.5 border transition-colors ${
              showConfig
                ? "border-cobalt-DEFAULT text-cobalt-DEFAULT dark:border-cobalt-dark dark:text-cobalt-dark bg-cobalt-subtleLight dark:bg-cobalt-subtleDark"
                : "border-border-light dark:border-border-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:border-border-strongLight dark:hover:border-border-strongDark"
            }`}
          >
            <SlidersHorizontal className="w-3 h-3" />
            <span>PARAMS {showConfig ? "[▲]" : "[▼]"}</span>
          </button>

          <span className="hidden sm:inline">
            MODE: <span className="text-ink-primaryLight dark:text-ink-primaryDark">HYBRID RRF</span>
          </span>
          <span className="hidden sm:inline text-border-strongLight dark:text-border-strongDark">|</span>
          <span className="hidden sm:inline">
            RERANK:{" "}
            <span
              className={
                rerank
                  ? "text-emerald-DEFAULT font-semibold"
                  : "text-ink-mutedLight dark:text-ink-mutedDark"
              }
            >
              {rerank ? "CROSS-ENCODER" : "DISABLED"}
            </span>
          </span>
        </div>

        <div className="hidden sm:flex items-center space-x-1 text-[10px]">
          <span>SUBMIT WITH</span>
          <kbd className="px-1.5 py-0.5 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-ink-primaryLight dark:text-ink-primaryDark font-mono">
            ↵ ENTER
          </kbd>
        </div>
      </div>

      {/* Expandable Parameters Drawer */}
      {showConfig && (
        <div className="mb-3 p-3 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-xs font-mono grid grid-cols-1 sm:grid-cols-2 gap-4 animate-fade-in">
          <div>
            <span className="block text-[10px] text-ink-mutedLight dark:text-ink-mutedDark uppercase mb-1.5">
              01 // RETRIEVAL DEPTH (TOP-K)
            </span>
            <div className="flex space-x-2">
              {[3, 5, 10].map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setTopK(k)}
                  className={`px-3 py-1 border text-xs font-mono transition-colors ${
                    topK === k
                      ? "border-cobalt-DEFAULT bg-cobalt-DEFAULT text-white dark:border-cobalt-dark dark:bg-cobalt-dark"
                      : "border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:border-border-strongLight dark:hover:border-border-strongDark"
                  }`}
                >
                  K = {k}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="block text-[10px] text-ink-mutedLight dark:text-ink-mutedDark uppercase mb-1.5">
              02 // CROSS-ENCODER RERANKING
            </span>
            <button
              type="button"
              onClick={() => setRerank(!rerank)}
              className={`px-3 py-1 border text-xs font-mono transition-colors ${
                rerank
                  ? "border-emerald-DEFAULT bg-emerald-subtleLight dark:bg-emerald-subtleDark text-emerald-DEFAULT"
                  : "border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark text-ink-mutedLight dark:text-ink-mutedDark"
              }`}
            >
              {rerank ? "● RERANKER ACTIVE" : "○ RERANKER BYPASSED"}
            </button>
          </div>
        </div>
      )}

      {/* Main Composer Box */}
      <div className="relative border border-border-strongLight dark:border-border-strongDark focus-within:border-cobalt-DEFAULT dark:focus-within:border-cobalt-dark bg-surface-light dark:bg-surface-dark transition-colors shadow-subtle">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Formulate an insurance inquiry, verify policy clauses, or test fraud conditions..."
          rows={1}
          className="w-full bg-transparent resize-none border-0 px-4 py-3 text-sm md:text-base text-ink-primaryLight dark:text-ink-primaryDark placeholder:text-ink-mutedLight dark:placeholder:text-ink-mutedDark focus:outline-none focus:ring-0 max-h-44"
        />

        {/* Action Button Row */}
        <div className="flex items-center justify-between px-3 py-2 border-t border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark">
          <span className="text-[10px] font-mono text-ink-mutedLight dark:text-ink-mutedDark">
            QUERY DOSSIER COMPOSER
          </span>

          <div className="flex items-center space-x-2">
            {isLoading ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="h-8 px-3 border border-vermilion-DEFAULT bg-vermilion-subtleLight dark:bg-vermilion-subtleDark text-vermilion-DEFAULT font-mono text-xs flex items-center space-x-1.5 hover:bg-vermilion-DEFAULT hover:text-white transition-colors"
              >
                <Square className="w-3 h-3 fill-current" />
                <span>HALT</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => handleSubmit()}
                disabled={!input.trim()}
                className={`h-8 px-4 font-mono text-xs font-semibold flex items-center space-x-1.5 transition-colors ${
                  input.trim()
                    ? "bg-cobalt-DEFAULT hover:bg-cobalt-hover dark:bg-cobalt-dark dark:hover:bg-cobalt-hover text-white"
                    : "bg-surface-subtleLight dark:bg-surface-subtleDark border border-border-light dark:border-border-dark text-ink-mutedLight dark:text-ink-mutedDark cursor-not-allowed"
                }`}
              >
                <span>RUN QUERY</span>
                <ArrowUp className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
