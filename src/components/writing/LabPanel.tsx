"use client";

import { useState } from "react";
import { labExamples } from "@/lib/writing/lab-examples";
import type { Genre, LabExample, Trait } from "@/lib/writing/types";

interface LabPanelProps {
  onYourTurn: (coldText: string, topic: string, genre: Genre) => void;
  loading: boolean;
  onLabRewrite: (
    text: string,
    temperatures: number[]
  ) => Promise<{ temperature: number; text: string; explanation: string }[]>;
}

const TRAIT_LABELS: Record<Trait, string> = {
  ideas: "Ideas",
  organization: "Organization",
  voice: "Voice",
  wordChoice: "Word Choice",
  fluency: "Fluency",
  conventions: "Conventions",
  presentation: "Presentation",
};

const MAX_REWRITES = 5;

export default function LabPanel({
  onYourTurn,
  loading,
  onLabRewrite,
}: LabPanelProps) {
  const [selected, setSelected] = useState<LabExample>(labExamples[0]);
  const [rewrites, setRewrites] = useState<
    { temperature: number; text: string; explanation: string }[] | null
  >(null);
  const [rewriteLoading, setRewriteLoading] = useState(false);
  const [rewriteCount, setRewriteCount] = useState(0);

  const isRateLimited = rewriteCount >= MAX_REWRITES;

  function handleSelectExample(example: LabExample) {
    setSelected(example);
    setRewrites(null);
  }

  async function handleSeeTemperature() {
    if (isRateLimited || rewriteLoading) return;
    setRewriteLoading(true);
    try {
      const results = await onLabRewrite(selected.coldText, [0, 0.7, 1.3]);
      setRewrites(results);
      setRewriteCount((c) => c + 1);
    } finally {
      setRewriteLoading(false);
    }
  }

  function handleYourTurn() {
    onYourTurn(selected.coldText, selected.topic, "essay");
  }

  const aiColumns = [
    { temp: 0, label: "AI t=0", fallbackExplanation: "Predictable — uses the safest word at every position." },
    { temp: 0.7, label: "AI t=0.7", fallbackExplanation: "Moderate randomness — occasionally picks less obvious words." },
    { temp: 1.3, label: "AI t=1.3", fallbackExplanation: "High randomness — surprising word choices, sometimes incoherent." },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div>
        <h2
          className="text-xl font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          Writing Lab
        </h2>
        <p className="text-sm italic" style={{ color: "var(--muted)" }}>
          &ldquo;Two Kinds of Temperature&rdquo;
        </p>
      </div>

      {/* Example selector grid */}
      <div>
        <p
          className="text-sm font-medium mb-2"
          style={{ color: "var(--foreground)" }}
        >
          Select an example:
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {labExamples.map((ex) => (
            <button
              key={ex.id}
              onClick={() => handleSelectExample(ex)}
              className="text-left rounded-lg p-3 text-xs transition-colors border"
              style={{
                background:
                  selected.id === ex.id ? "var(--accent-light)" : "var(--card)",
                borderColor:
                  selected.id === ex.id
                    ? "var(--accent)"
                    : "var(--card-border)",
                color: "var(--foreground)",
              }}
            >
              <span className="block font-medium leading-tight mb-1">
                {ex.topic}
              </span>
              <span
                className="inline-block rounded-full px-2 py-0.5 text-[10px] font-medium"
                style={{
                  background: "var(--accent-light)",
                  color: "var(--accent)",
                }}
              >
                {TRAIT_LABELS[ex.focusTrait]}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Cold text display */}
      <div
        className="rounded-lg p-4 border"
        style={{
          background: "#f0efed",
          borderColor: "var(--card-border)",
          color: "#6b6966",
        }}
      >
        <p
          className="text-xs font-semibold uppercase tracking-wide mb-2"
          style={{ color: "var(--muted)" }}
        >
          Cold Text (original bland prose)
        </p>
        <p className="text-sm leading-relaxed">{selected.coldText}</p>
      </div>

      {/* See how AI adds temperature button */}
      <div className="flex justify-center">
        {isRateLimited ? (
          <p className="text-sm italic" style={{ color: "var(--muted)" }}>
            You&apos;ve explored enough — try writing your own version!
          </p>
        ) : (
          <button
            onClick={handleSeeTemperature}
            disabled={rewriteLoading}
            className="rounded-lg px-5 py-2.5 text-sm font-medium transition-opacity disabled:opacity-50"
            style={{
              background: "var(--accent)",
              color: "#fff",
            }}
          >
            {rewriteLoading
              ? "Generating rewrites…"
              : "See how AI adds temperature ▶"}
          </button>
        )}
      </div>

      {/* Four-column comparison */}
      {rewrites && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {aiColumns.map((col) => {
            const match = rewrites.find((r) => r.temperature === col.temp);
            return (
              <div
                key={col.temp}
                className="rounded-lg border flex flex-col"
                style={{
                  background: "var(--card)",
                  borderColor: "var(--card-border)",
                }}
              >
                <div
                  className="px-4 py-2 border-b text-xs font-semibold uppercase tracking-wide"
                  style={{
                    borderColor: "var(--card-border)",
                    color: "var(--muted)",
                  }}
                >
                  {col.label}
                </div>
                <div className="px-4 py-3 flex-1">
                  <p
                    className="text-sm leading-relaxed"
                    style={{ color: "var(--foreground)" }}
                  >
                    {match?.text ?? "No result for this temperature."}
                  </p>
                </div>
                <div
                  className="px-4 py-3 border-t text-xs italic"
                  style={{
                    borderColor: "var(--card-border)",
                    color: "var(--muted)",
                    background: "#fafaf8",
                  }}
                >
                  {match?.explanation ?? col.fallbackExplanation}
                </div>
              </div>
            );
          })}

          {/* Human warm column */}
          <div
            className="rounded-lg border flex flex-col"
            style={{
              background: "#fdf6ef",
              borderColor: "#e8d5c4",
            }}
          >
            <div
              className="px-4 py-2 border-b text-xs font-semibold uppercase tracking-wide"
              style={{
                borderColor: "#e8d5c4",
                color: "var(--accent)",
              }}
            >
              Human Warm
            </div>
            <div className="px-4 py-3 flex-1">
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--foreground)" }}
              >
                {selected.humanWarmText}
              </p>
            </div>
            <div
              className="px-4 py-3 border-t text-xs italic"
              style={{
                borderColor: "#e8d5c4",
                color: "var(--accent)",
                background: "#faf0e6",
              }}
            >
              {selected.humanExplanation}
            </div>
          </div>
        </div>
      )}

      {/* Teaching point */}
      <div
        className="rounded-lg p-4 border-l-4"
        style={{
          borderColor: "var(--accent)",
          background: "var(--accent-light)",
          color: "var(--foreground)",
        }}
      >
        <p
          className="text-xs font-semibold uppercase tracking-wide mb-1"
          style={{ color: "var(--accent)" }}
        >
          Teaching Point
        </p>
        <p className="text-base leading-relaxed">{selected.teachingPoint}</p>
      </div>

      {/* Your turn button */}
      <div className="flex justify-center">
        <button
          onClick={handleYourTurn}
          disabled={loading}
          className="rounded-lg px-6 py-3 text-sm font-medium transition-opacity disabled:opacity-50"
          style={{
            background: "var(--foreground)",
            color: "var(--background)",
          }}
        >
          Your turn — try it yourself →
        </button>
      </div>
    </div>
  );
}
