"use client";

import { useState } from "react";
import type { TokenData, GLTRData } from "@/lib/analysis";

interface Props {
  tokens: TokenData[];
  gltr: GLTRData;
}

function getRankColor(rank: number): string {
  if (rank <= 10) return "bg-emerald-200/70 text-emerald-900";
  if (rank <= 100) return "bg-yellow-200/70 text-yellow-900";
  if (rank <= 1000) return "bg-red-200/70 text-red-900";
  return "bg-purple-200/70 text-purple-900";
}

function getRankLabel(rank: number): string {
  if (rank <= 10) return "Top 10";
  if (rank <= 100) return "Top 100";
  if (rank <= 1000) return "Top 1000";
  return "1000+";
}

export default function GLTROverlay({ tokens, gltr }: Props) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
        <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
          GLTR Token Rank Distribution
        </h3>
        <p className="text-xs text-[var(--muted)] mb-4">
          AI text overwhelmingly uses top-10 predicted tokens. Human text is
          more diverse.
        </p>

        <div className="flex h-6 rounded-full overflow-hidden mb-3">
          {gltr.top10 > 0 && (
            <div
              style={{ width: `${gltr.top10}%` }}
              className="bg-emerald-400"
            />
          )}
          {gltr.top100 > 0 && (
            <div
              style={{ width: `${gltr.top100}%` }}
              className="bg-yellow-400"
            />
          )}
          {gltr.top1000 > 0 && (
            <div
              style={{ width: `${gltr.top1000}%` }}
              className="bg-red-400"
            />
          )}
          {gltr.above1000 > 0 && (
            <div
              style={{ width: `${gltr.above1000}%` }}
              className="bg-purple-400"
            />
          )}
        </div>

        <div className="flex flex-wrap gap-3 text-xs text-[var(--muted)]">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-emerald-400" />
            Top 10: {gltr.top10.toFixed(1)}%
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-yellow-400" />
            Top 100: {gltr.top100.toFixed(1)}%
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-400" />
            Top 1000: {gltr.top1000.toFixed(1)}%
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-purple-400" />
            1000+: {gltr.above1000.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
        <h4 className="text-xs font-medium text-[var(--muted)] mb-3">
          Color-coded text
        </h4>
        <div className="flex flex-wrap gap-0.5 leading-relaxed max-h-[200px] overflow-y-auto p-1">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`inline-block px-1 py-0.5 rounded text-[13px] cursor-default transition-all ${getRankColor(t.rank)} ${hoveredIndex === i ? "ring-2 ring-[var(--accent)] scale-110 z-10" : ""}`}
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              {t.token}
            </span>
          ))}
        </div>
        {hoveredIndex !== null && tokens[hoveredIndex] && (
          <div className="mt-3 pt-3 border-t border-[var(--card-border)] text-xs text-[var(--muted)] flex items-center gap-3 flex-wrap">
            <span className="font-mono text-[var(--foreground)] bg-[var(--background)] px-2 py-1 rounded">
              {tokens[hoveredIndex].token}
            </span>
            <span>Rank: {tokens[hoveredIndex].rank}</span>
            <span>PPL: {tokens[hoveredIndex].perplexity.toFixed(1)}</span>
            <span>Entropy: {tokens[hoveredIndex].entropy.toFixed(2)}</span>
            <span className="text-[var(--accent)]">
              {getRankLabel(tokens[hoveredIndex].rank)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
