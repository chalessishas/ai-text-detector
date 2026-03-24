"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function PostBody({ content }: { content: string }) {
  return (
    <div className="prose prose-neutral max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold text-[var(--foreground)] mt-8 mb-3">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold text-[var(--foreground)] mt-6 mb-2">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="text-sm text-[var(--foreground)] leading-relaxed mb-4">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="text-sm text-[var(--foreground)] list-disc pl-5 mb-4 space-y-1">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="text-sm text-[var(--foreground)] list-decimal pl-5 mb-4 space-y-1">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">{children}</li>
          ),
          code: ({ className, children }) => {
            const isBlock = className?.includes("language-");
            if (isBlock) {
              return (
                <code className="block bg-[#2d2b28] text-[#e8e2d9] rounded-lg p-4 text-xs leading-relaxed overflow-x-auto mb-4">
                  {children}
                </code>
              );
            }
            return (
              <code className="bg-[var(--accent-light)] text-[var(--accent)] px-1.5 py-0.5 rounded text-xs font-mono">
                {children}
              </code>
            );
          },
          pre: ({ children }) => <pre className="mb-4">{children}</pre>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-[var(--accent)] pl-4 text-sm text-[var(--muted)] italic mb-4">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-[var(--accent)] hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[var(--foreground)]">
              {children}
            </strong>
          ),
          hr: () => (
            <hr className="border-[var(--card-border)] my-6" />
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4">
              <table className="w-full text-sm border-collapse">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="text-left text-xs font-semibold text-[var(--foreground)] border-b border-[var(--card-border)] px-3 py-2">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="text-sm text-[var(--foreground)] border-b border-[var(--card-border)] px-3 py-2">
              {children}
            </td>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
