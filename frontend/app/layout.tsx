import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kortex AI — Enterprise Multilingual Hybrid RAG Platform",
  description: "Swiss Editorial enterprise knowledge intelligence engine powered by Hybrid RRF, Cross-Encoder reranking, and grounded document validation.",
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
                  const stored = localStorage.getItem('kortex-theme') || localStorage.getItem('claimeye-theme');
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
