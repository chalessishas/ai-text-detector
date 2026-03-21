"use client";

import type { DailyTip, Trait } from "@/lib/writing/types";

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

interface DailyTipCardProps {
  tip: DailyTip;
  onTryIt: (exercisePrompt: string) => void;
  onSkip: () => void;
  onDisable: () => void;
}

export default function DailyTipCard({
  tip,
  onTryIt,
  onSkip,
  onDisable,
}: DailyTipCardProps) {
  return (
    <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-xl p-4 space-y-3">
      {/* Trait badge */}
      <div className="flex items-center gap-2">
        <span
          className={`w-2.5 h-2.5 rounded-full ${TRAIT_DOT_COLORS[tip.trait]}`}
        />
        <span className="text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
          {TRAIT_LABELS[tip.trait]}
        </span>
        <span className="ml-auto text-xs text-[var(--muted)]">Daily Tip</span>
      </div>

      {/* Tip text */}
      <p className="text-sm text-[var(--foreground)] leading-relaxed">
        {tip.tip}
      </p>

      {/* Before/After example */}
      {tip.example && (
        <div className="space-y-1.5">
          <div className="rounded-lg bg-red-50 border border-red-100 px-3 py-2">
            <span className="text-[10px] font-semibold uppercase text-red-400 tracking-wide">
              Before
            </span>
            <p className="text-xs text-red-700 mt-0.5 leading-relaxed">
              {tip.example.before}
            </p>
          </div>
          <div className="rounded-lg bg-green-50 border border-green-100 px-3 py-2">
            <span className="text-[10px] font-semibold uppercase text-green-500 tracking-wide">
              After
            </span>
            <p className="text-xs text-green-700 mt-0.5 leading-relaxed">
              {tip.example.after}
            </p>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 pt-1">
        {tip.exercisePrompt && (
          <button
            onClick={() => onTryIt(tip.exercisePrompt!)}
            className="px-3 py-1.5 text-xs font-medium text-white bg-[var(--accent)] rounded-lg hover:opacity-90 transition-all"
          >
            Try it &rarr;
          </button>
        )}
        <button
          onClick={onSkip}
          className="px-3 py-1.5 text-xs font-medium text-[var(--foreground)] bg-[var(--background)] border border-[var(--card-border)] rounded-lg hover:bg-[var(--card-border)] transition-all"
        >
          Skip
        </button>
        <button
          onClick={onDisable}
          className="ml-auto text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        >
          Don&apos;t show tips
        </button>
      </div>
    </div>
  );
}
