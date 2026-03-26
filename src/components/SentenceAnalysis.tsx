"use client";

import { useState } from "react";
import type { SentenceScoreData } from "@/lib/analysis";
import { explainSentenceScore } from "@/lib/analysis";

interface Props {
  sentenceScores: SentenceScoreData[];
  overallAiScore?: number;
}

function getHighlightColor(score: number): string {
  if (score > 70) return "bg-red-100/80";
  if (score > 30) return "bg-amber-100/60";
  return "bg-emerald-100/40";
}

function getImpactLevel(score: number): { label: string; bars: number } {
  if (score > 70) return { label: "High AI impact", bars: 3 };
  if (score > 30) return { label: "Medium AI impact", bars: 2 };
  return { label: "Low AI impact", bars: 1 };
}

function ImpactBars({ bars }: { bars: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {[1, 2, 3].map((i) => (
        <span
          key={i}
          className={`w-1 h-3 rounded-full ${
            i <= bars
              ? bars === 3
                ? "bg-red-400"
                : bars === 2
                  ? "bg-amber-400"
                  : "bg-emerald-400"
              : "bg-gray-200"
          }`}
        />
      ))}
    </span>
  );
}

export default function SentenceAnalysis({ sentenceScores }: Props) {
  const [selected, setSelected] = useState<number | null>(null);
  const [expandedExplain, setExpandedExplain] = useState<number | null>(null);

  const aiSentences = sentenceScores
    .filter((s) => s.aiScore > 30)
    .sort((a, b) => b.aiScore - a.aiScore);
  const humanSentences = sentenceScores
    .filter((s) => s.aiScore <= 30)
    .sort((a, b) => a.aiScore - b.aiScore);

  const aiCount = sentenceScores.filter((s) => s.aiScore > 70).length;
  const mixedCount = sentenceScores.filter(
    (s) => s.aiScore > 30 && s.aiScore <= 70
  ).length;
  const humanCount = sentenceScores.filter((s) => s.aiScore <= 30).length;

  return (
    <div className="space-y-4">
      {/* Highlighted text */}
      <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
        <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
          Sentence-Level Analysis
        </h3>
        <p className="text-xs text-[var(--muted)] mb-4">
          {aiCount > 0
            ? `${aiCount}/${sentenceScores.length} sentences show strong AI patterns.`
            : mixedCount > 0
              ? `${mixedCount}/${sentenceScores.length} sentences show some AI-like patterns.`
              : `All ${sentenceScores.length} sentences appear human-written.`}
          {" "}Click a sentence to see details.
        </p>

        <div className="flex gap-4 text-xs mb-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-red-200" />
            AI-like: {aiCount}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-amber-200" />
            Mixed: {mixedCount}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-200" />
            Human-like: {humanCount}
          </span>
        </div>

        <div className="text-sm leading-[1.8]">
          {sentenceScores.map((s, i) => (
            <span
              key={i}
              className={`${getHighlightColor(s.aiScore)} ${selected === i ? "ring-2 ring-[var(--accent)]" : ""} rounded px-0.5 cursor-pointer transition-all inline`}
              onClick={() => {
                setSelected(i);
                setExpandedExplain(expandedExplain === i ? null : i);
              }}
              onMouseEnter={() => setSelected(i)}
              onMouseLeave={() => { if (expandedExplain !== i) setSelected(null); }}
            >
              {s.text}{" "}
            </span>
          ))}
        </div>
      </div>

      {/* Impact ranking — AI sentences */}
      <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
        <h4 className="text-xs font-semibold text-[var(--foreground)] mb-1">
          Top sentences driving AI probability
        </h4>
        <p className="text-[10px] text-[var(--muted)] mb-3">
          Ranked by AI impact. Click &ldquo;Explain&rdquo; to see why.
        </p>
        <div className="space-y-1.5 max-h-[260px] overflow-y-auto">
          {aiSentences.map((s) => {
            const impact = getImpactLevel(s.aiScore);
            const isExpanded = expandedExplain === s.index;
            const reasons = explainSentenceScore(s);
            return (
              <div key={s.index}>
                <div
                  className={`flex items-center gap-3 p-2.5 rounded-xl text-xs transition-all cursor-pointer ${
                    selected === s.index
                      ? "bg-[var(--background)] ring-1 ring-[var(--accent)]"
                      : "hover:bg-[var(--background)]"
                  }`}
                  onMouseEnter={() => setSelected(s.index)}
                  onMouseLeave={() => { if (!isExpanded) setSelected(null); }}
                >
                  <ImpactBars bars={impact.bars} />
                  <span className="text-[var(--foreground)] flex-1 min-w-0 line-clamp-2">
                    {s.text}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedExplain(isExpanded ? null : s.index);
                      setSelected(s.index);
                    }}
                    className="shrink-0 text-[10px] text-[var(--accent)] hover:underline"
                  >
                    {isExpanded ? "Hide" : "Explain"}
                  </button>
                  <span className="shrink-0 font-mono text-[var(--muted)] tabular-nums w-8 text-right">
                    {s.aiScore.toFixed(0)}%
                  </span>
                </div>
                {isExpanded && (
                  <div className="ml-8 mr-12 mb-2 p-3 bg-red-50/50 rounded-lg border border-red-100">
                    <p className="text-[10px] font-semibold text-red-700 mb-1">Why we think it&apos;s AI:</p>
                    {reasons.map((r, ri) => (
                      <p key={ri} className="text-[11px] text-red-800/80 leading-relaxed">
                        {r}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Human sentences */}
      {humanSentences.length > 0 && (
        <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
          <h4 className="text-xs font-semibold text-[var(--foreground)] mb-1">
            Top sentences driving Human probability
          </h4>
          <div className="space-y-1.5 max-h-[200px] overflow-y-auto">
            {humanSentences.map((s) => {
              const impact = getImpactLevel(s.aiScore);
              const isExpanded = expandedExplain === s.index;
              const reasons = explainSentenceScore(s);
              return (
                <div key={s.index}>
                  <div
                    className={`flex items-center gap-3 p-2.5 rounded-xl text-xs transition-all cursor-pointer ${
                      selected === s.index
                        ? "bg-[var(--background)] ring-1 ring-[var(--accent)]"
                        : "hover:bg-[var(--background)]"
                    }`}
                    onMouseEnter={() => setSelected(s.index)}
                    onMouseLeave={() => { if (!isExpanded) setSelected(null); }}
                  >
                    <ImpactBars bars={impact.bars} />
                    <span className="text-[var(--foreground)] flex-1 min-w-0 line-clamp-2">
                      {s.text}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedExplain(isExpanded ? null : s.index);
                        setSelected(s.index);
                      }}
                      className="shrink-0 text-[10px] text-[var(--accent)] hover:underline"
                    >
                      {isExpanded ? "Hide" : "Explain"}
                    </button>
                    <span className="shrink-0 font-mono text-[var(--muted)] tabular-nums w-8 text-right">
                      {s.aiScore.toFixed(0)}%
                    </span>
                  </div>
                  {isExpanded && (
                    <div className="ml-8 mr-12 mb-2 p-3 bg-emerald-50/50 rounded-lg border border-emerald-100">
                      <p className="text-[10px] font-semibold text-emerald-700 mb-1">Why we think it&apos;s human:</p>
                      {reasons.map((r, ri) => (
                        <p key={ri} className="text-[11px] text-emerald-800/80 leading-relaxed">
                          {r}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
