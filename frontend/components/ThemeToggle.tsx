"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const saved = (localStorage.getItem("kortex-theme") || localStorage.getItem("claimeye-theme")) as "dark" | "light" | null;
    if (saved) {
      setTheme(saved);
      if (saved === "light") {
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.add("light");
      } else {
        document.documentElement.classList.remove("light");
        document.documentElement.classList.add("dark");
      }
    } else {
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("kortex-theme", next);

    if (next === "light") {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    } else {
      document.documentElement.classList.remove("light");
      document.documentElement.classList.add("dark");
    }
  };

  if (!mounted) {
    return (
      <div className="h-8 w-8 border border-border-light dark:border-border-dark opacity-0" />
    );
  }

  return (
    <button
      onClick={toggleTheme}
      title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
      aria-label="Toggle visual theme"
      className="h-8 px-2.5 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark hover:border-cobalt dark:hover:border-cobalt-dark text-ink-secondaryLight dark:text-ink-secondaryDark hover:text-ink-primaryLight dark:hover:text-ink-primaryDark transition-colors flex items-center space-x-1.5 text-xs font-mono"
    >
      {theme === "dark" ? (
        <>
          <Sun className="w-3.5 h-3.5 text-amber-DEFAULT" />
          <span className="text-[10px] uppercase tracking-wider hidden sm:inline">Light</span>
        </>
      ) : (
        <>
          <Moon className="w-3.5 h-3.5 text-cobalt-DEFAULT" />
          <span className="text-[10px] uppercase tracking-wider hidden sm:inline">Dark</span>
        </>
      )}
    </button>
  );
}
