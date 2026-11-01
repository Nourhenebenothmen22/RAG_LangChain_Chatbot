"use client";

import {
  Activity,
  CheckCircle2,
  Database,
  Layers,
  Plus,
  Terminal,
  Trash2,
  X,
} from "lucide-react";
import { ConversationSession, HealthStatus } from "../lib/types";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: ConversationSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  onOpenDocuments: () => void;
  health: HealthStatus | null;
}

export function Sidebar({
  isOpen,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onOpenDocuments,
  health,
}: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden animate-fade-in"
          onClick={onClose}
        />
      )}

      {/* Sidebar Rail */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 w-72 flex flex-col bg-surface-light dark:bg-surface-dark border-r border-border-light dark:border-border-dark transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Brand Masthead Section */}
        <div className="p-4 border-b border-border-light dark:border-border-dark flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-base tracking-tighter text-ink-primaryLight dark:text-ink-primaryDark uppercase">
                KORTEX AI
              </span>
              <span className="w-1.5 h-1.5 bg-cobalt-DEFAULT dark:bg-cobalt-dark" />
            </div>
            <p className="text-[10px] font-mono text-ink-mutedLight dark:text-ink-mutedDark tracking-wide mt-0.5">
              ENTERPRISE INTELLIGENCE DOSSIER
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-1 md:hidden border border-border-light dark:border-border-dark text-ink-mutedLight dark:text-ink-mutedDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark"
            aria-label="Close Sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Primary Action: New Inquiry Dossier */}
        <div className="p-3 border-b border-border-light dark:border-border-dark space-y-2">
          <button
            onClick={() => {
              onNewSession();
              if (window.innerWidth < 768) onClose();
            }}
            className="w-full h-9 px-3 bg-ink-primaryLight dark:bg-ink-primaryDark text-canvas-light dark:text-canvas-dark font-mono text-xs font-semibold flex items-center justify-between hover:bg-cobalt-DEFAULT dark:hover:bg-cobalt-dark dark:hover:text-white transition-colors"
          >
            <span className="flex items-center space-x-2">
              <Plus className="w-3.5 h-3.5" />
              <span>NEW INQUIRY</span>
            </span>
            <span className="text-[10px] opacity-70">⌘N</span>
          </button>

          <button
            onClick={() => {
              onOpenDocuments();
              if (window.innerWidth < 768) onClose();
            }}
            className="w-full h-8 px-3 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark hover:border-cobalt-DEFAULT dark:hover:border-cobalt-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark font-mono text-xs flex items-center justify-between transition-colors"
          >
            <span className="flex items-center space-x-2">
              <Database className="w-3.5 h-3.5 text-cobalt-DEFAULT dark:text-cobalt-dark" />
              <span>02 // KNOWLEDGE CORPUS</span>
            </span>
            <span className="text-[10px] text-ink-mutedLight dark:text-ink-mutedDark">→</span>
          </button>
        </div>

        {/* Section 01: Inquiries History */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="flex items-center justify-between px-2 py-1.5 text-[10px] font-mono tracking-widest text-ink-mutedLight dark:text-ink-mutedDark uppercase border-b border-border-light dark:border-border-dark mb-2">
            <span>01 // INQUIRIES DOSSIER</span>
            <span>[{sessions.length.toString().padStart(2, "0")}]</span>
          </div>

          {sessions.length === 0 ? (
            <div className="p-4 text-center text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark border border-dashed border-border-light dark:border-border-dark">
              NO ACTIVE DOSSIERS
            </div>
          ) : (
            sessions.map((session, index) => {
              const isActive = session.id === activeSessionId;
              const dateStr = new Date(session.updatedAt || session.createdAt).toLocaleDateString(
                undefined,
                { month: "numeric", day: "numeric" }
              );
              return (
                <div
                  key={session.id}
                  onClick={() => {
                    onSelectSession(session.id);
                    if (window.innerWidth < 768) onClose();
                  }}
                  className={`group relative p-2.5 text-xs font-mono transition-colors border cursor-pointer flex items-center justify-between ${
                    isActive
                      ? "border-l-2 border-l-cobalt-DEFAULT dark:border-l-cobalt-dark border-y-border-strongLight dark:border-y-border-strongDark border-r-border-strongLight dark:border-r-border-strongDark bg-surface-subtleLight dark:bg-surface-subtleDark text-ink-primaryLight dark:text-ink-primaryDark"
                      : "border-transparent hover:border-border-light dark:hover:border-border-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark"
                  }`}
                >
                  <div className="flex items-baseline space-x-2 truncate pr-2">
                    <span className="text-[10px] text-ink-mutedLight dark:text-ink-mutedDark">
                      {(index + 1).toString().padStart(2, "0")}
                    </span>
                    <span className="truncate">{session.title || "Untitled Inquiry"}</span>
                  </div>

                  <div className="flex items-center space-x-1.5 flex-shrink-0">
                    <span className="text-[10px] text-ink-mutedLight dark:text-ink-mutedDark group-hover:hidden">
                      {dateStr}
                    </span>
                    <button
                      onClick={(e) => onDeleteSession(session.id, e)}
                      title="Delete Session"
                      className="hidden group-hover:flex p-1 hover:text-vermilion-DEFAULT transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Section 03: Architecture Telemetry */}
        <div className="p-3 border-t border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark text-[11px] font-mono text-ink-mutedLight dark:text-ink-mutedDark space-y-2">
          <div className="text-[10px] tracking-wider uppercase text-ink-secondaryLight dark:text-ink-secondaryDark font-semibold">
            03 // SYSTEM ARCHITECTURE
          </div>

          <div className="space-y-1.5 pt-1">
            <div className="flex items-center justify-between">
              <span>PIPELINE</span>
              <span className="text-ink-primaryLight dark:text-ink-primaryDark font-medium">
                HYBRID RAG
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>VDB ENGINE</span>
              <span className="text-ink-primaryLight dark:text-ink-primaryDark font-medium">
                CHROMA (LOCAL)
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>RERANKER</span>
              <span className="text-emerald-DEFAULT font-medium">
                BGE-RERANKER
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>STATUS</span>
              <span className="flex items-center space-x-1 text-emerald-DEFAULT">
                <CheckCircle2 className="w-3 h-3" />
                <span>ONLINE</span>
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
