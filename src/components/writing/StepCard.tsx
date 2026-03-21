"use client";

import { useState } from "react";
import type { StepCard as StepCardType } from "@/lib/writing/types";

interface StepCardProps {
  card: StepCardType;
  onComplete: (cardId: string) => void;
}

export default function StepCardComponent({ card, onComplete }: StepCardProps) {
  const [checked, setChecked] = useState<boolean[]>(
    () => card.checklist?.map(() => false) ?? []
  );

  function toggleCheck(index: number) {
    setChecked((prev) => {
      const next = [...prev];
      next[index] = !next[index];
      return next;
    });
  }

  return (
    <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-xl p-4 space-y-3">
      {/* Progress + mnemonic */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--muted)]">
          Step {card.stepIndex + 1} of {card.totalSteps}
        </span>
        {card.mnemonic && (
          <span className="ml-auto px-2 py-0.5 text-[10px] font-bold uppercase bg-[var(--accent-light)] text-[var(--accent)] rounded-full">
            {card.mnemonic}
          </span>
        )}
      </div>

      {/* Title */}
      <h4 className="text-sm font-semibold text-[var(--foreground)]">
        {card.title}
      </h4>

      {/* Instructions */}
      <p className="text-xs text-[var(--muted)] leading-relaxed">
        {card.instructions}
      </p>

      {/* Checklist */}
      {card.checklist && card.checklist.length > 0 && (
        <ul className="space-y-1.5">
          {card.checklist.map((item, i) => (
            <li key={i} className="flex items-start gap-2">
              <button
                onClick={() => toggleCheck(i)}
                className={`mt-0.5 w-4 h-4 rounded border flex-shrink-0 flex items-center justify-center transition-colors ${
                  checked[i]
                    ? "bg-[var(--accent)] border-[var(--accent)] text-white"
                    : "border-[var(--card-border)]"
                }`}
              >
                {checked[i] && (
                  <svg
                    width="10"
                    height="10"
                    viewBox="0 0 12 12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M2 6l3 3 5-5" />
                  </svg>
                )}
              </button>
              <span
                className={`text-xs leading-relaxed ${
                  checked[i]
                    ? "line-through text-[var(--muted)]"
                    : "text-[var(--foreground)]"
                }`}
              >
                {item}
              </span>
            </li>
          ))}
        </ul>
      )}

      {/* Example */}
      {card.example && (
        <blockquote className="border-l-2 border-[var(--card-border)] pl-3 text-xs text-[var(--muted)] italic leading-relaxed">
          {card.example}
        </blockquote>
      )}

      {/* Done button */}
      {!card.completed ? (
        <button
          onClick={() => onComplete(card.id)}
          className="w-full py-1.5 text-xs font-medium text-[var(--accent)] border border-[var(--accent)] rounded-lg hover:bg-[var(--accent-light)] transition-all"
        >
          Done &#10003;
        </button>
      ) : (
        <div className="text-xs text-green-600 font-medium text-center py-1.5">
          &#10003; Completed
        </div>
      )}
    </div>
  );
}
