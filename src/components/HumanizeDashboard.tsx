"use client";

import { useState } from "react";
import type {
  HumanizeResult,
  HumanizeSentenceDetail,
  MethodKey,
  MethodResult,
} from "@/app/api/humanize/route";

const METHODS: {
  key: MethodKey;
  label: string;
  bg: string;
  border: string;
  text: string;
  hoverBorder: string;
  desc: string;
}[] = [
  // ── Corpus-based (human text as base) ──
  {
    key: "corpus",
    label: "Corpus",
    bg: "bg-blue-50/60",
    border: "border-blue-200",
    text: "text-blue-700",
    hoverBorder: "hover:border-blue-200",
    desc: "Semantic match, 100% human",
  },
  {
    key: "structure",
    label: "Structure",
    bg: "bg-purple-50/60",
    border: "border-purple-200",
    text: "text-purple-700",
    hoverBorder: "hover:border-purple-200",
    desc: "Same grammar, 100% human",
  },
  {
    key: "transplant",
    label: "Transplant",
    bg: "bg-emerald-50/60",
    border: "border-emerald-200",
    text: "text-emerald-700",
    hoverBorder: "hover:border-emerald-200",
    desc: "Corpus + your entities/numbers",
  },
  {
    key: "inject",
    label: "Inject",
    bg: "bg-cyan-50/60",
    border: "border-cyan-200",
    text: "text-cyan-700",
    hoverBorder: "hover:border-cyan-200",
    desc: "Corpus + your objects swapped in",
  },
  {
    key: "harvest",
    label: "Harvest",
    bg: "bg-orange-50/60",
    border: "border-orange-200",
    text: "text-orange-700",
    hoverBorder: "hover:border-orange-200",
    desc: "Best clauses from multiple humans",
  },
  {
    key: "remix",
    label: "Remix",
    bg: "bg-indigo-50/60",
    border: "border-indigo-200",
    text: "text-indigo-700",
    hoverBorder: "hover:border-indigo-200",
    desc: "All content words from humans",
  },
  {
    key: "anchor",
    label: "Anchor",
    bg: "bg-lime-50/60",
    border: "border-lime-200",
    text: "text-lime-700",
    hoverBorder: "hover:border-lime-200",
    desc: "Corpus sentence + your facts appended",
  },
  // ⚠️ 以下四种方法经验证无效（在原文上做局部替换，AI检测器仍可识别），已从后端禁用：
  // phrase, collocation, noise, splice
];

type Choice = MethodKey | "original";

interface Props {
  result: HumanizeResult;
}

export default function HumanizeDashboard({ result }: Props) {
  const [choices, setChoices] = useState<Choice[]>(() =>
    result.details.map((d) => {
      for (const m of METHODS) {
        if (d.methods[m.key]) return m.key;
      }
      return "original";
    })
  );
  const [copied, setCopied] = useState(false);

  function getOutput(d: HumanizeSentenceDetail, choice: Choice): string {
    if (choice === "original") return d.original;
    const m = d.methods[choice as MethodKey];
    return m?.text || d.original;
  }

  function getFinalOutput(): string {
    return result.details
      .map((d, i) => getOutput(d, choices[i] || "original"))
      .join(" ");
  }

  function setChoice(idx: number, choice: Choice) {
    const next = [...choices];
    next[idx] = choice;
    setChoices(next);
  }

  function setAll(choice: Choice) {
    setChoices(result.details.map(() => choice));
  }

  function handleCopy() {
    navigator.clipboard.writeText(getFinalOutput()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  }

  // Count methods across all sentences
  const methodCounts: Record<string, number> = { original: 0 };
  for (const m of METHODS) methodCounts[m.key] = 0;
  for (const c of choices) methodCounts[c] = (methodCounts[c] || 0) + 1;

  return (
    <div className="space-y-5">
      {/* Final output */}
      <div className="bg-white rounded-2xl p-5 border border-emerald-200 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[var(--foreground)]">
            Final Output
          </h3>
          <button
            onClick={handleCopy}
            className="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-medium px-4 py-1.5 rounded-lg transition-all"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <p className="text-sm leading-relaxed text-[var(--foreground)] whitespace-pre-wrap max-h-[200px] overflow-y-auto">
          {getFinalOutput()}
        </p>
        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[var(--card-border)]">
          {METHODS.map((m) =>
            methodCounts[m.key] > 0 ? (
              <span
                key={m.key}
                className={`text-[10px] px-2 py-0.5 rounded-full ${m.bg} ${m.text}`}
              >
                {m.label}: {methodCounts[m.key]}
              </span>
            ) : null
          )}
          {methodCounts.original > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
              Original: {methodCounts.original}
            </span>
          )}
        </div>
      </div>

      {/* Sentence cards */}
      <div className="bg-white rounded-2xl border border-[var(--card-border)] shadow-sm overflow-hidden">
        {/* Header with bulk actions */}
        <div className="px-5 py-3 border-b border-[var(--card-border)] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-[var(--foreground)]">
              Sentence Dashboard
            </h3>
            <p className="text-[11px] text-[var(--muted)] mt-0.5">
              {result.sentenceCount} sentences &middot; click a method to select
            </p>
          </div>
          <div className="flex gap-1 flex-wrap justify-end">
            {METHODS.map((m) => (
              <button
                key={m.key}
                onClick={() => setAll(m.key)}
                className={`text-[10px] px-2 py-1 rounded ${m.bg} ${m.text} hover:opacity-80 transition-opacity`}
              >
                All {m.label}
              </button>
            ))}
            <button
              onClick={() => setAll("original")}
              className="text-[10px] px-2 py-1 rounded bg-gray-100 text-gray-600 hover:opacity-80 transition-opacity"
            >
              All Original
            </button>
          </div>
        </div>

        {/* Sentence list */}
        <div className="divide-y divide-[var(--card-border)] max-h-[700px] overflow-y-auto">
          {result.details.map((d, i) => (
            <SentenceCard
              key={i}
              index={i}
              detail={d}
              choice={choices[i] || "original"}
              onChoose={(c) => setChoice(i, c)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function SentenceCard({
  index,
  detail,
  choice,
  onChoose,
}: {
  index: number;
  detail: HumanizeSentenceDetail;
  choice: Choice;
  onChoose: (c: Choice) => void;
}) {
  const availableMethods = METHODS.filter((m) => detail.methods[m.key]);

  return (
    <div className="px-5 py-4">
      {/* Original sentence */}
      <div className="flex items-start gap-2 mb-3">
        <span className="text-[10px] font-bold text-[var(--muted)] mt-0.5 shrink-0">
          #{index + 1}
        </span>
        <p className="text-xs text-[var(--muted)] line-through opacity-60 leading-relaxed">
          {detail.original}
        </p>
      </div>

      {/* Method grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 ml-5">
        {availableMethods.map((m) => {
          const data = detail.methods[m.key]!;
          const isActive = choice === m.key;
          return (
            <MethodCard
              key={m.key}
              method={m}
              data={data}
              isActive={isActive}
              onClick={() => onChoose(m.key)}
            />
          );
        })}
        {/* Original option */}
        <button
          onClick={() => onChoose("original")}
          className={`text-left rounded-xl border p-3 transition-all ${
            choice === "original"
              ? "border-gray-400 bg-gray-100 ring-1 ring-gray-400"
              : "border-gray-200 bg-gray-50/30 hover:border-gray-300"
          }`}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span
              className={`text-[10px] font-semibold ${
                choice === "original" ? "text-gray-800" : "text-gray-500"
              }`}
            >
              Original
            </span>
          </div>
          <p className="text-xs leading-relaxed text-gray-600 line-clamp-3">
            {detail.original}
          </p>
        </button>
      </div>
    </div>
  );
}

function MethodCard({
  method,
  data,
  isActive,
  onClick,
}: {
  method: (typeof METHODS)[number];
  data: MethodResult;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`text-left rounded-xl border p-3 transition-all ${
        isActive
          ? `${method.border} ${method.bg} ring-1 ring-current`
          : `border-[var(--card-border)] ${method.hoverBorder} bg-white`
      }`}
    >
      {/* Label + score */}
      <div className="flex items-center justify-between mb-1.5">
        <span
          className={`text-[10px] font-semibold ${
            isActive ? method.text : "text-[var(--muted)]"
          }`}
        >
          {method.label}
        </span>
        {data.score !== undefined && (
          <span className="text-[9px] text-[var(--muted)] tabular-nums">
            {(data.score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Text */}
      <p
        className={`text-xs leading-relaxed line-clamp-3 ${
          isActive ? method.text : "text-[var(--foreground)]"
        }`}
      >
        {data.text}
      </p>

      {/* Metadata */}
      <div className="mt-2 space-y-0.5">
        {data.swaps && data.swaps.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {data.swaps.map((s, j) => (
              <span
                key={j}
                className="text-[9px] bg-black/5 rounded px-1 py-0.5"
              >
                <span className="line-through opacity-50">{s.from}</span>
                {" → "}
                <span className="font-medium">{s.to}</span>
              </span>
            ))}
          </div>
        )}
        {data.template && (
          <p className="text-[9px] text-[var(--muted)] italic truncate">
            tpl: {data.template}
          </p>
        )}
        {data.splitPoint !== undefined && (
          <p className="text-[9px] text-[var(--muted)]">
            split at word {data.splitPoint}
          </p>
        )}
      </div>
    </button>
  );
}
