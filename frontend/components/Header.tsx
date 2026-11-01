"use client";

import { Database, Menu, RotateCcw } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface HeaderProps {
  onToggleSidebar: () => void;
  onClearChat: () => void;
  onOpenDocuments: () => void;
  hasMessages: boolean;
}

export function Header({
  onToggleSidebar,
  onClearChat,
  onOpenDocuments,
  hasMessages,
}: HeaderProps) {
  return (
    <header className="h-13 bg-surface-light dark:bg-surface-dark border-b border-border-light dark:border-border-dark sticky top-0 z-30 flex items-center justify-between px-4 sm:px-6 transition-colors">
      {/* Left: Mobile Sidebar Toggle */}
      <div className="flex items-center space-x-3 sm:space-x-4">
        <button
          onClick={onToggleSidebar}
          className="p-1.5 md:hidden border border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark transition-colors"
          aria-label="Toggle Navigation Sidebar"
        >
          <Menu className="w-4 h-4" />
        </button>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center space-x-2">
        <button
          onClick={onOpenDocuments}
          title="Manage Knowledge Base Documents & Corpus"
          className="h-8 px-3 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark hover:border-cobalt-DEFAULT dark:hover:border-cobalt-dark focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark text-ink-primaryLight dark:text-ink-primaryDark text-xs font-mono font-medium flex items-center space-x-2 transition-colors"
        >
          <Database className="w-3.5 h-3.5 text-cobalt-DEFAULT dark:text-cobalt-dark" />
          <span className="hidden sm:inline">KNOWLEDGE BASE</span>
        </button>

        {hasMessages && (
          <button
            onClick={onClearChat}
            title="Reset Active Inquiry Dossier"
            className="h-8 px-2.5 border border-border-light dark:border-border-dark hover:border-vermilion-DEFAULT focus:outline-none focus:ring-1 focus:ring-vermilion-DEFAULT text-ink-mutedLight dark:text-ink-mutedDark hover:text-vermilion-DEFAULT text-xs font-mono font-medium flex items-center space-x-1.5 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">RESET</span>
          </button>
        )}

        <ThemeToggle />
      </div>
    </header>
  );
}
