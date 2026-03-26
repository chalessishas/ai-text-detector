"use client";

import { useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import type { AnalysisResult } from "@/lib/analysis";
import type { HumanizeResult } from "@/app/api/humanize/route";

const WritingCenter = dynamic(
  () => import("@/components/writing/WritingCenter"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);

const HumanizeDashboard = dynamic(
  () => import("@/components/HumanizeDashboard"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
const FeatureOverview = dynamic(
  () => import("@/components/FeatureOverview"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
const GLTROverlay = dynamic(() => import("@/components/GLTROverlay"), {
  ssr: false,
  loading: () => <PanelSkeleton />,
});
const PerplexityCurve = dynamic(
  () => import("@/components/PerplexityCurve"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
const EntropyChart = dynamic(() => import("@/components/EntropyChart"), {
  ssr: false,
  loading: () => <PanelSkeleton />,
});
const BurstinessChart = dynamic(
  () => import("@/components/BurstinessChart"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
const SlidingWindowChart = dynamic(
  () => import("@/components/SlidingWindowChart"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
const SentenceAnalysis = dynamic(
  () => import("@/components/SentenceAnalysis"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);

function PanelSkeleton() {
  return (
    <div className="bg-[var(--card)] rounded-xl p-5 border border-[var(--card-border)] h-[360px] animate-pulse">
      <div className="h-4 w-36 bg-[#ece7df] rounded mb-4" />
      <div className="h-3 w-56 bg-[#ece7df] rounded mb-6" />
      <div className="h-[260px] bg-[#f5f1eb] rounded-lg" />
    </div>
  );
}

// ── Sidebar config ──

type Panel = "detect" | "humanize" | "writing";

const panels: { key: Panel; icon: React.ReactNode; label: string }[] = [
  {
    key: "detect",
    label: "AI Detector",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
  },
  {
    key: "humanize",
    label: "Humanizer",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
      </svg>
    ),
  },
  {
    key: "writing",
    label: "Writing Center",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    ),
  },
];

// ── Detect tabs ──

const detectTabsAll = [
  { key: "overview", label: "Overview", icon: "◎", needsTokens: false },
  { key: "sentences", label: "Sentences", icon: "≡", needsTokens: true },
  { key: "gltr", label: "GLTR", icon: "▦", needsTokens: true },
  { key: "perplexity", label: "Perplexity", icon: "〰", needsTokens: true },
  { key: "entropy", label: "Entropy", icon: "◆", needsTokens: true },
  { key: "burstiness", label: "Burstiness", icon: "▥", needsTokens: false },
  { key: "window", label: "Window", icon: "◳", needsTokens: true },
] as const;

type DetectTab = (typeof detectTabsAll)[number]["key"];

// ── Main ──

export default function AppShell() {
  const [activePanel, setActivePanel] = useState<Panel>("detect");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Detect
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<DetectTab>("overview");

  // Humanize
  const [humanizeResult, setHumanizeResult] = useState<HumanizeResult | null>(null);


  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Analysis failed"); return; }
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setLoading(false);
    }
  }

  async function handleHumanize() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setHumanizeResult(null);
    try {
      const res = await fetch("/api/humanize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Humanization failed"); return; }
      setHumanizeResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setLoading(false);
    }
  }

  function scoreColor(s: number) {
    if (s > 70) return "text-[#c44]";
    if (s > 30) return "text-amber-600";
    return "text-emerald-600";
  }
  function scoreBg(s: number) {
    if (s > 70) return "bg-red-50 border-red-200";
    if (s > 30) return "bg-amber-50 border-amber-200";
    return "bg-emerald-50 border-emerald-200";
  }
  function scoreLabel(s: number, fused?: { prediction?: string }) {
    if (fused?.prediction === "uncertain") return "Uncertain — low confidence";
    if (s > 70) return "Likely AI-generated";
    if (s > 30) return "Inconclusive";
    return "Likely Human-written";
  }

  function renderDetectChart() {
    if (!result) return null;
    switch (activeTab) {
      case "overview": return <FeatureOverview result={result} />;
      case "sentences": return <SentenceAnalysis sentenceScores={result.sentenceScores} />;
      case "gltr": return <GLTROverlay tokens={result.tokens} gltr={result.gltr} />;
      case "perplexity": return <PerplexityCurve tokens={result.tokens} />;
      case "entropy": return <EntropyChart tokens={result.tokens} stats={result.entropy} />;
      case "burstiness": return <BurstinessChart sentences={result.sentences} />;
      case "window": return <SlidingWindowChart data={result.slidingWindow} />;
    }
  }

  return (
    <div className="h-screen flex overflow-hidden bg-[var(--background)]">
      {/* ── Activity Bar (VSCode-style) ── */}
      <div className="w-[52px] bg-[#2d2b28] flex flex-col items-center py-3 gap-1 shrink-0">
        {panels.map((p) => (
          <button
            key={p.key}
            onClick={() => setActivePanel(p.key)}
            title={p.label}
            className={`w-10 h-10 flex items-center justify-center rounded-lg transition-all relative group ${
              activePanel === p.key
                ? "text-white bg-white/10"
                : "text-white/40 hover:text-white/70"
            }`}
          >
            {p.icon}
            {activePanel === p.key && (
              <span className="absolute left-0 top-1.5 bottom-1.5 w-[2px] bg-[var(--accent)] rounded-r" />
            )}
            <span className="absolute left-12 px-2 py-1 bg-[#1e1d1b] text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity z-50">
              {p.label}
            </span>
          </button>
        ))}

        <div className="flex-1" />

        <Link
          href="/blog"
          title="Dev Log"
          className="w-10 h-10 flex items-center justify-center rounded-lg text-white/40 hover:text-white/70 transition-all relative group"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
          </svg>
          <span className="absolute left-12 px-2 py-1 bg-[#1e1d1b] text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity z-50">
            Dev Log
          </span>
        </Link>

        <div className="w-6 h-6 rounded-full bg-[var(--accent)] flex items-center justify-center">
          <span className="text-white text-[10px] font-bold">X</span>
        </div>
      </div>

      {/* ── Main Area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Title bar */}
        <div className="h-10 bg-[#3a3836] flex items-center px-4 shrink-0">
          <span className="text-white/60 text-xs font-medium">
            AI Text X-Ray
          </span>
          <span className="text-white/30 text-xs mx-2">/</span>
          <span className="text-white/90 text-xs font-medium">
            {panels.find((p) => p.key === activePanel)?.label}
          </span>
        </div>

        {/* Content */}
        <div className={`flex-1 ${activePanel === "writing" ? "overflow-hidden" : "overflow-y-auto"}`}>
          {error && (
            <div className="mx-6 mt-4 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-red-700 text-sm">
              {error}
            </div>
          )}

          {activePanel === "detect" && (
            <DetectPanel
              text={text}
              setText={setText}
              loading={loading}
              result={result}
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              onAnalyze={handleAnalyze}
              scoreColor={scoreColor}
              scoreBg={scoreBg}
              scoreLabel={scoreLabel}
              renderChart={renderDetectChart}
            />
          )}

          {activePanel === "humanize" && (
            <HumanizePanel
              text={text}
              setText={setText}
              loading={loading}
              humanizeResult={humanizeResult}
              onHumanize={handleHumanize}
            />
          )}

          {activePanel === "writing" && <WritingCenter />}
        </div>

        {/* Status bar (hidden when WritingCenter renders its own) */}
        {activePanel !== "writing" && (
          <div className="h-6 bg-[var(--accent)] flex items-center px-3 shrink-0">
            <span className="text-white/80 text-[10px]">
              {text.trim() ? `${text.trim().split(/\s+/).filter(Boolean).length} words` : "Ready"}
            </span>
            {result && (
              <span className="text-white/80 text-[10px] ml-3">
                {result.fused ? `${result.fused.prediction} (${result.fused.confidence}%)` : `Score: ${result.overallScore.toFixed(0)}%`}
                {result.hasTokenData && ` · ${result.tokens.length} tokens`}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Detect Panel ──

function DetectPanel({
  text, setText, loading, result, activeTab, setActiveTab, onAnalyze,
  scoreColor, scoreBg, scoreLabel, renderChart,
}: {
  text: string;
  setText: (t: string) => void;
  loading: boolean;
  result: AnalysisResult | null;
  activeTab: DetectTab;
  setActiveTab: (t: DetectTab) => void;
  onAnalyze: () => void;
  scoreColor: (s: number) => string;
  scoreBg: (s: number) => string;
  scoreLabel: (s: number, fused?: { prediction?: string }) => string;
  renderChart: () => React.ReactNode;
}) {
  return (
    <div className="p-6 flex flex-col lg:flex-row gap-6 lg:items-start">
      {/* Left: Input + Score */}
      <div className="lg:w-[440px] lg:shrink-0 space-y-4">
        <div className="bg-[var(--card)] rounded-xl border border-[var(--card-border)] shadow-sm">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text here to analyze..."
            rows={16}
            className="w-full bg-transparent text-[var(--foreground)] text-sm leading-relaxed placeholder:text-[#c4bfb7] focus:outline-none resize-y min-h-[300px] p-5"
          />
          <div className="flex items-center justify-between px-5 py-3 border-t border-[var(--card-border)]">
            <span className="text-xs text-[var(--muted)]">
              {text.trim().split(/\s+/).filter(Boolean).length} words
            </span>
            <button
              onClick={onAnalyze}
              disabled={loading || !text.trim()}
              className="bg-[var(--accent)] hover:bg-[#b5583a] disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-sm font-medium px-5 py-2 rounded-lg transition-all"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </span>
              ) : "Analyze"}
            </button>
          </div>
        </div>

        {/* Beta warning */}
        <div className="rounded-lg px-3 py-2 bg-amber-50 border border-amber-200">
          <span className="text-[10px] font-semibold text-amber-700">Beta</span>
          <span className="text-[10px] text-amber-600 ml-1.5">
            Best on English essays (200+ words). May misclassify formal academic writing, business emails, legal, or technical text. Short texts (&lt;50 words) are too brief for reliable detection.
          </span>
        </div>

        {result && result.scoringEligible && (
          <div className={`rounded-xl border overflow-hidden ${scoreBg(result.overallScore)}`}>
            {/* Verdict hero */}
            <div className="p-5 flex items-center gap-5">
              <div className="relative w-16 h-16 shrink-0">
                <svg viewBox="0 0 36 36" className="w-16 h-16 -rotate-90">
                  <circle cx="18" cy="18" r="15.5" fill="none" stroke="currentColor" strokeWidth="2" className="opacity-10" />
                  <circle
                    cx="18" cy="18" r="15.5" fill="none"
                    strokeWidth="3" strokeLinecap="round"
                    stroke={result.overallScore > 70 ? "#c44" : result.overallScore > 30 ? "#d97706" : "#059669"}
                    strokeDasharray={`${result.overallScore * 0.974} 97.4`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className={`text-lg font-bold ${scoreColor(result.overallScore)}`}>
                    {result.overallScore.toFixed(0)}
                  </span>
                </div>
              </div>
              <div className="flex-1">
                <div className={`text-base font-semibold ${scoreColor(result.overallScore)}`}>
                  {scoreLabel(result.overallScore, result.fused)}
                </div>
                <div className="text-xs text-[var(--muted)] mt-1">
                  {result.fused?.prediction === "ai" ? "AI-generated" : result.fused?.prediction === "uncertain" ? "Insufficient evidence" : "Human-written"}
                  {" · "}{result.wordCount} words
                  {result.fused && result.fused.signal_source !== "blended" && (
                    <span className="ml-1 opacity-60">
                      ({result.fused.signal_source === "ppl_override_ai" ? "perplexity confirms" : "perplexity override"})
                    </span>
                  )}
                </div>
                {result.classification && (
                  <div className="flex gap-1.5 mt-2">
                    {Object.entries(result.classification.probabilities).map(([name, prob]) => (
                      <span key={name} className={`text-[10px] px-1.5 py-0.5 rounded ${
                        name === result.classification!.label_name
                          ? "bg-[var(--foreground)]/10 font-semibold"
                          : "text-[var(--muted)]"
                      }`}>
                        {name}: {(prob * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
            {/* AI vocab matches */}
            {result.aiVocabMatches.length > 0 && (
              <div className="px-5 pb-4 pt-0">
                <div className="text-[10px] text-[var(--muted)] mb-1">AI vocabulary detected:</div>
                <div className="flex flex-wrap gap-1">
                  {[...new Set(result.aiVocabMatches.map(m => m.word.toLowerCase()))].map((word) => (
                    <span key={word} className="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-700">
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {result && !result.scoringEligible && (
          <div className="rounded-xl p-4 border bg-amber-50 border-amber-200">
            <div className="text-sm font-semibold text-amber-700">Insufficient text for scoring</div>
            <div className="text-xs text-amber-600 mt-1">
              {result.classification ? "Analysis complete" : `300+ words needed (${result.wordCount} provided)`}. See charts for details.
            </div>
          </div>
        )}
      </div>

      {/* Right: Charts */}
      <div className="flex-1 min-w-0">
        <div className="bg-[var(--card)] rounded-xl border border-[var(--card-border)] shadow-sm overflow-hidden">
          <div className="flex border-b border-[var(--card-border)] overflow-x-auto">
            {detectTabsAll.filter(tab => !tab.needsTokens || (result?.hasTokenData ?? false)).map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex-1 min-w-[72px] px-2.5 py-3 text-xs font-medium transition-colors relative whitespace-nowrap ${
                  activeTab === tab.key
                    ? "text-[var(--accent)]"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
              >
                <span className="flex items-center justify-center gap-1">
                  <span className="text-sm">{tab.icon}</span>
                  {tab.label}
                </span>
                {activeTab === tab.key && (
                  <span className="absolute bottom-0 left-2 right-2 h-[2px] bg-[var(--accent)] rounded-full" />
                )}
              </button>
            ))}
          </div>
          <div className="p-5">
            {result ? renderChart() : (
              <div className="h-[360px] flex items-center justify-center text-[var(--muted)] text-sm">
                {loading ? (
                  <div className="flex flex-col items-center gap-3">
                    <span className="w-7 h-7 border-2 border-[var(--accent)]/30 border-t-[var(--accent)] rounded-full animate-spin" />
                    <span>Analyzing text...</span>
                  </div>
                ) : "Paste text and click Analyze to see results"}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Humanize Panel ──

function HumanizePanel({
  text, setText, loading, humanizeResult, onHumanize,
}: {
  text: string;
  setText: (t: string) => void;
  loading: boolean;
  humanizeResult: HumanizeResult | null;
  onHumanize: () => void;
}) {
  return (
    <div className="p-6 flex flex-col lg:flex-row gap-6 lg:items-start">
      <div className="lg:w-[440px] lg:shrink-0">
        <div className="bg-[var(--card)] rounded-xl border border-[var(--card-border)] shadow-sm">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste AI-generated text to humanize..."
            rows={16}
            className="w-full bg-transparent text-[var(--foreground)] text-sm leading-relaxed placeholder:text-[#c4bfb7] focus:outline-none resize-y min-h-[300px] p-5"
          />
          <div className="flex items-center justify-between px-5 py-3 border-t border-[var(--card-border)]">
            <span className="text-xs text-[var(--muted)]">
              {text.trim().split(/\s+/).filter(Boolean).length} words
            </span>
            <button
              onClick={onHumanize}
              disabled={loading || !text.trim()}
              className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-sm font-medium px-5 py-2 rounded-lg transition-all"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Humanizing...
                </span>
              ) : "Humanize"}
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-w-0">
        {humanizeResult ? (
          <HumanizeDashboard result={humanizeResult} />
        ) : (
          <div className="bg-[var(--card)] rounded-xl border border-[var(--card-border)] shadow-sm">
            <div className="h-[360px] flex items-center justify-center text-[var(--muted)] text-sm">
              {loading ? (
                <div className="flex flex-col items-center gap-3">
                  <span className="w-7 h-7 border-2 border-emerald-300/50 border-t-emerald-500 rounded-full animate-spin" />
                  <span>Running humanization methods...</span>
                </div>
              ) : "Paste AI text and click Humanize to see sentence dashboard"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

