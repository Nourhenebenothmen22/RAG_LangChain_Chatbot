import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClaimEye — Insurance Intelligence & Forensic Hybrid RAG",
  description: "Swiss Editorial enterprise insurance intelligence engine powered by Hybrid RRF, Cross-Encoder reranking, and grounded policy validation.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const stored = localStorage.getItem('claimeye-theme');
                  if (stored === 'light') {
                    document.documentElement.classList.remove('dark');
                    document.documentElement.classList.add('light');
                  } else {
                    document.documentElement.classList.remove('light');
                    document.documentElement.classList.add('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="antialiased selection:bg-cobalt selection:text-white">
        {children}
      </body>
    </html>
  );
}
