import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          light: "#FBFBFA",
          dark: "#0C0D10",
        },
        surface: {
          light: "#FFFFFF",
          dark: "#14161E",
          subtleLight: "#F3F2EC",
          subtleDark: "#1A1D28",
        },
        border: {
          light: "#D8D7CE",
          dark: "#2A2F40",
          strongLight: "#B8B7AC",
          strongDark: "#3B4257",
        },
        ink: {
          primaryLight: "#111215",
          primaryDark: "#F4F4F0",
          secondaryLight: "#4E515C",
          secondaryDark: "#9BA0B0",
          mutedLight: "#575B66",
          mutedDark: "#8E95A5",
        },
        cobalt: {
          DEFAULT: "#1A56DB",
          light: "#1A56DB",
          dark: "#3B72FF",
          hover: "#1644B2",
          subtleLight: "#EEF2FF",
          subtleDark: "rgba(59, 114, 255, 0.16)",
        },
        vermilion: {
          DEFAULT: "#D9383A",
          subtleLight: "#FDECEC",
          subtleDark: "rgba(217, 56, 58, 0.14)",
        },
        emerald: {
          DEFAULT: "#15803D",
          subtleLight: "#EDF8F1",
          subtleDark: "rgba(21, 128, 61, 0.14)",
        },
        amber: {
          DEFAULT: "#B45309",
          subtleLight: "#FEF3C7",
          subtleDark: "rgba(180, 83, 9, 0.14)",
        },
      },
      fontFamily: {
        sans: [
          '"Plus Jakarta Sans"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
        serif: ['"Newsreader"', "Georgia", "serif"],
      },
      boxShadow: {
        subtle: "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)",
        card: "0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04)",
        elevated: "0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04)",
      },
      letterSpacing: {
        tighter: "-0.035em",
        tight: "-0.02em",
        normal: "0",
        wide: "0.04em",
        wider: "0.08em",
        widest: "0.14em",
      },
      animation: {
        "fade-in": "fadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
