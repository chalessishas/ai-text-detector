"use client";

import type { AnalysisResult, FeatureScore } from "@/lib/analysis";

interface Props {
  result: AnalysisResult;
}

function ScoreBar({ score }: { score: number }) {
  const color =
    score > 0.7 ? "bg-red-400" : score > 0.3 ? "bg-amber-400" : "bg-emerald-400";
  return (
    <div className="h-1.5 rounded-full bg-gray-100 mt-1.5">
      <div
        style={{ width: `${score * 100}%` }}
        className={`h-full rounded-full ${color} transition-all`}
      />
    </div>
  );
}

function FeatureCard({ feature }: { feature: FeatureScore }) {
  const label =
    feature.score > 0.7
      ? "AI-like"
      : feature.score > 0.3
        ? "Mixed"
        : "Human-like";
  const labelColor =
    feature.score > 0.7
      ? "text-red-600"
      : feature.score > 0.3
        ? "text-amber-600"
        : "text-emerald-600";

  return (
    <div className="bg-[var(--background)] rounded-xl p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-[var(--foreground)]">
          {feature.name}
        </span>
        <span className={`text-[10px] font-medium ${labelColor}`}>
          {label}
        </span>
      </div>
      <ScoreBar score={feature.score} />
      <div className="flex justify-between mt-2 text-[10px] text-[var(--muted)]">
        <span>Raw: {feature.raw.toFixed(2)}</span>
        <span>
          H: {feature.humanRef} | AI: {feature.aiRef}
        </span>
      </div>
    </div>
  );
}

export default function FeatureOverview({ result }: Props) {
  const isAI = result.overallScore > 50;
  return (
    <div className="space-y-4">
      {/* AI Similarity Tags */}
      {result.aiSimilarityTags.length > 0 && (
        <div className={`bg-white rounded-2xl p-6 border shadow-sm ${isAI ? "border-red-100" : "border-emerald-100"}`}>
          <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
            {isAI ? "Why this text looks AI-generated" : "Analysis details"}
          </h3>
          <p className="text-xs text-[var(--muted)] mb-4">
            {isAI
              ? "Patterns detected that are characteristic of AI-generated text."
              : "Some patterns were flagged, but overall the text appears human-written."}
          </p>
          <div className="flex flex-wrap gap-2 mb-4">
            {result.aiSimilarityTags.map((tag) => (
              <span
                key={tag.label}
                className="text-[11px] px-2.5 py-1 rounded-full bg-red-50 text-red-700 border border-red-200 font-medium"
              >
                {tag.label}
              </span>
            ))}
          </div>
          <div className="space-y-2">
            {result.aiSimilarityTags.map((tag) => (
              <div key={tag.label} className="bg-[var(--background)] rounded-lg p-3">
                <span className="text-xs font-semibold text-[var(--foreground)]">
                  {tag.label}
                </span>
                <p className="text-[11px] text-[var(--muted)] mt-0.5 leading-relaxed">
                  {tag.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Vocab */}
      {result.aiVocabMatches.length > 0 && (
        <div className="bg-white rounded-2xl p-6 border border-amber-100 shadow-sm">
          <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
            AI Vocabulary Detected
          </h3>
          <p className="text-xs text-[var(--muted)] mb-3">
            Words and phrases that appear frequently in AI-generated text.
          </p>
          <div className="flex flex-wrap gap-2">
            {[...new Set(result.aiVocabMatches.map((m) => m.word.toLowerCase()))].map(
              (word) => {
                const count = result.aiVocabMatches.filter(
                  (m) => m.word.toLowerCase() === word
                ).length;
                return (
                  <span
                    key={word}
                    className="text-[11px] px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium"
                  >
                    {word}
                    {count > 1 && (
                      <span className="ml-1 text-amber-500">×{count}</span>
                    )}
                  </span>
                );
              }
            )}
          </div>
        </div>
      )}

      {/* Feature Breakdown */}
      <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
        <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
          Feature Breakdown
        </h3>
        <p className="text-xs text-[var(--muted)] mb-4">
          Each feature normalized 0-1 (0 = human-like, 1 = AI-like), weighted
          for overall score.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {result.featureScores.map((f) => (
            <FeatureCard key={f.name} feature={f} />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-2xl p-5 border border-[var(--card-border)] shadow-sm">
          <h4 className="text-xs font-semibold text-[var(--foreground)] mb-3">
            Vocabulary
          </h4>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Type-Token Ratio</span>
              <span className="font-medium">
                {result.vocabulary.ttr.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Hapax Ratio</span>
              <span className="font-medium">
                {result.vocabulary.hapaxRatio.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Unique / Total</span>
              <span className="font-medium">
                {result.vocabulary.uniqueWords} / {result.vocabulary.totalWords}
              </span>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-2xl p-5 border border-[var(--card-border)] shadow-sm">
          <h4 className="text-xs font-semibold text-[var(--foreground)] mb-3">
            Sentence Length
          </h4>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Mean</span>
              <span className="font-medium">
                {result.sentenceLength.mean.toFixed(1)} words
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Std Dev</span>
              <span className="font-medium">
                {result.sentenceLength.std.toFixed(1)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--muted)]">Burstiness CV</span>
              <span className="font-medium">
                {result.burstiness.lengthCV.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
