import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dev Log — AI Text X-Ray",
  description:
    "Development log and technical notes for AI Text X-Ray — an open AI text detector and humanizer.",
};

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Nav bar */}
      <header className="sticky top-0 z-10 bg-[#2d2b28] border-b border-white/5">
        <div className="max-w-3xl mx-auto px-6 h-12 flex items-center justify-between">
          <Link
            href="/blog"
            className="text-white/90 text-sm font-semibold tracking-tight hover:text-white transition-colors"
          >
            AI Text X-Ray <span className="text-white/40 font-normal">/ Dev Log</span>
          </Link>
          <Link
            href="/"
            className="text-white/50 text-xs hover:text-white/80 transition-colors"
          >
            Back to App
          </Link>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-3xl mx-auto px-6 py-10">{children}</main>

      {/* Footer */}
      <footer className="max-w-3xl mx-auto px-6 py-8 border-t border-[var(--card-border)]">
        <p className="text-xs text-[var(--muted)]">
          AI Text X-Ray — Free AI text detection and humanization.
        </p>
      </footer>
    </div>
  );
}
