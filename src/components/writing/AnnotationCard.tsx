"use client";

import type { Annotation, Trait, Severity } from "@/lib/writing/types";

const TRAIT_LABELS: Record<Trait, string> = {
  ideas: "Ideas",
  organization: "Organization",
  voice: "Voice",
  wordChoice: "Word Choice",
  fluency: "Fluency",
  conventions: "Conventions",
  presentation: "Presentation",
};

const TRAIT_DOT_COLORS: Record<Trait, string> = {
  ideas: "bg-blue-500",
  organization: "bg-purple-500",
  voice: "bg-orange-500",
  wordChoice: "bg-teal-500",
  fluency: "bg-green-500",
  conventions: "bg-gray-500",
  presentation: "bg-pink-500",
};

const SEVERITY_ICONS: Record<Severity, string> = {
  good: "\u2713",
  question: "?",
  suggestion: "\u26A1",
  issue: "\u26A0\uFE0F",
};

const SEVERITY_COLORS: Record<Severity, string> = {
  good: "text-green-600 bg-green-50",
  question: "text-blue-600 bg-blue-50",
  suggestion: "text-amber-600 bg-amber-50",
  issue: "text-red-600 bg-red-50",
};

interface AnnotationCardProps {
  annotation: Annotation;
  expanded?: boolean;
  expandedDetail?: {
    detail: string;
    suggestion?: string;
    question: string;
  } | null;
  onExpand: (annotationId: string) => void;
  onClick: (annotationId: string) => void;
  isFocused?: boolean;
}

export default function AnnotationCard({
  annotation,
  expanded = false,
  expandedDetail,
  onExpand,
  onClick,
  isFocused = false,
}: AnnotationCardProps) {
  return (
    <div
      onClick={() => onClick(annotation.id)}
      className={`bg-[var(--card)] border rounded-xl p-3 space-y-2 cursor-pointer transition-all hover:shadow-sm ${
        isFocused
          ? "border-amber-400 ring-2 ring-amber-300 animate-pulse-ring"
          : "border-[var(--card-border)]"
      }`}
    >
      {/* Header: trait badge + severity icon */}
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${TRAIT_DOT_COLORS[annotation.trait]}`}
        />
        <span className="text-[10px] font-medium text-[var(--muted)] uppercase tracking-wide">
          {TRAIT_LABELS[annotation.trait]}
        </span>
        <span
          className={`ml-auto w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${SEVERITY_COLORS[annotation.severity]}`}
        >
          {SEVERITY_ICONS[annotation.severity]}
        </span>
      </div>

      {/* Message */}
      <p className="text-xs text-[var(--foreground)] leading-relaxed">
        {annotation.message}
      </p>

      {/* Rewrite preview */}
      {annotation.rewrite && (
        <div className="rounded-lg bg-[var(--background)] px-3 py-2 border border-[var(--card-border)]">
          <span className="text-[10px] font-semibold uppercase text-[var(--muted)] tracking-wide">
            Suggested rewrite
          </span>
          <p className="text-xs text-[var(--foreground)] mt-0.5 leading-relaxed italic">
            {annotation.rewrite}
          </p>
        </div>
      )}

      {/* Learn more / Expanded detail */}
      {!expanded && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onExpand(annotation.id);
          }}
          className="text-xs text-[var(--accent)] hover:underline font-medium"
        >
          Learn more
        </button>
      )}

      {expanded && expandedDetail && (
        <div className="space-y-2 pt-1 border-t border-[var(--card-border)]">
          <p className="text-xs text-[var(--foreground)] leading-relaxed">
            {expandedDetail.detail}
          </p>
          {expandedDetail.suggestion && (
            <div className="rounded-lg bg-green-50 px-3 py-2 border border-green-100">
              <span className="text-[10px] font-semibold uppercase text-green-500 tracking-wide">
                Suggestion
              </span>
              <p className="text-xs text-green-700 mt-0.5 leading-relaxed">
                {expandedDetail.suggestion}
              </p>
            </div>
          )}
          <div className="rounded-lg bg-blue-50 px-3 py-2 border border-blue-100">
            <span className="text-[10px] font-semibold uppercase text-blue-500 tracking-wide">
              Think about
            </span>
            <p className="text-xs text-blue-700 mt-0.5 leading-relaxed">
              {expandedDetail.question}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
