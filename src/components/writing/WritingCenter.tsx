"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type {
  Genre,
  ChatMessage,
  Annotation,
  WriterProfile,
  DailyTip,
  StepCard,
  AnalyzeResponse,
  GuideStepResponse,
  GuideDialogueResponse,
  ExpandResponse,
  DailyTipResponse,
  Trait,
} from "@/lib/writing/types";
import {
  loadState,
  saveState,
  createDefaultProfile,
  updateStreak,
  markDayActive,
  addAnalysisToProfile,
  incrementGenreExperience,
} from "@/lib/writing/storage";
import { selectStaticTip } from "@/lib/writing/daily-tips";
import Editor, { type EditorHandle } from "./Editor";
import ChatPanel from "./ChatPanel";

const GENRE_LABELS: Record<Genre, string> = {
  essay: "Essay",
  article: "Article",
  academic: "Academic",
  creative: "Creative",
  business: "Business",
};

type LayoutPreset = "side" | "top" | "full";
type Tab = "chat" | "dashboard" | "lab";

function countWords(html: string): number {
  const text = html.replace(/<[^>]*>/g, " ").trim();
  if (!text) return 0;
  return text.split(/\s+/).filter(Boolean).length;
}

function latestTraitScores(
  profile: WriterProfile
): Record<Trait, number> | null {
  const last = profile.analysisHistory[profile.analysisHistory.length - 1];
  return last?.traitScores ?? null;
}

async function apiCall<T>(body: Record<string, unknown>): Promise<T> {
  const res = await fetch("/api/writing-assist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Request failed" }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export default function WritingCenter() {
  const [genre, setGenre] = useState<Genre>("essay");
  const [topic, setTopic] = useState("");
  const [document, setDocument] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [layoutPreset, setLayoutPreset] = useState<LayoutPreset>("side");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [profile, setProfile] = useState<WriterProfile>(() =>
    loadState()?.profile ?? createDefaultProfile()
  );
  const [hasIncrementedThisSession, setHasIncrementedThisSession] =
    useState(false);
  const [focusedAnnotationId, setFocusedAnnotationId] = useState<
    string | null
  >(null);
  const [copied, setCopied] = useState(false);

  // Daily tip state
  const [dailyTip, setDailyTip] = useState<DailyTip | null>(null);
  const [showDailyTip, setShowDailyTip] = useState(true);

  // Step cards state
  const [stepCards, setStepCards] = useState<StepCard[]>([]);

  // Expanded annotation state
  const [expandedAnnotation, setExpandedAnnotation] = useState<{
    id: string;
    detail: string;
    suggestion?: string;
    question: string;
  } | null>(null);

  // Conventions suppression tracking
  const [previousConventionsSuppressed, setPreviousConventionsSuppressed] =
    useState(false);

  // Drag state for split pane
  const [splitRatio, setSplitRatio] = useState(0.5);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const editorRef = useRef<EditorHandle>(null);

  // Session word tracking for streak
  const sessionWordsRef = useRef(0);

  // Load saved state on mount
  useEffect(() => {
    const saved = loadState();
    if (saved.draft.document) setDocument(saved.draft.document);
    if (saved.draft.genre) setGenre(saved.draft.genre);
    if (saved.draft.topic) setTopic(saved.draft.topic);
    if (saved.draft.messages.length) setMessages(saved.draft.messages);
    if (saved.draft.annotations.length)
      setAnnotations(saved.draft.annotations);
    if (saved.profile) setProfile(saved.profile);
  }, []);

  // Streak update on mount
  useEffect(() => {
    setProfile((prev) => updateStreak(prev));
  }, []);

  // Daily tip on mount
  useEffect(() => {
    if (!profile.preferences.showDailyTips) {
      setShowDailyTip(false);
      return;
    }

    if (profile.stats.totalAnalyses >= 3) {
      const scores = latestTraitScores(profile);
      apiCall<DailyTipResponse>({
        action: "daily-tip",
        traitScores: scores,
        analysisHistory: profile.analysisHistory.slice(-5),
      })
        .then((data) => setDailyTip(data.tip))
        .catch(() => {
          setDailyTip(selectStaticTip(scores, new Date()));
        });
    } else {
      const scores = latestTraitScores(profile);
      setDailyTip(selectStaticTip(scores, new Date()));
    }
    // Run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Guide step cards when genre + topic set and editor empty
  useEffect(() => {
    if (!genre || !topic || document.trim()) return;
    const experienceLevel = profile.genreExperience[genre];

    apiCall<GuideStepResponse>({
      action: "guide",
      mode: "step",
      genre,
      topic,
      experienceLevel,
    })
      .then((data) => {
        if (data.cards) setStepCards(data.cards);
      })
      .catch(() => {
        // Silently fail — step cards are optional guidance
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [genre, topic]);

  // 60s idle detection
  useEffect(() => {
    if (!document.trim()) return;
    const timer = setTimeout(() => {
      const idleMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "I notice you've paused — need help with anything? Feel free to ask about your writing.",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, idleMsg]);
    }, 60000);
    return () => clearTimeout(timer);
  }, [document]);

  // Track words written for streak
  useEffect(() => {
    const wc = countWords(document);
    const delta = wc - sessionWordsRef.current;
    if (delta > 0) sessionWordsRef.current = wc;
    if (sessionWordsRef.current >= 50) {
      setProfile((prev) => markDayActive(prev));
    }
  }, [document]);

  // Auto-save debounced 2s
  useEffect(() => {
    const timer = setTimeout(() => {
      saveState({
        draft: {
          genre,
          topic,
          document,
          messages,
          annotations,
          lastSaved: Date.now(),
        },
        profile,
      });
    }, 2000);
    return () => clearTimeout(timer);
  }, [document, messages, annotations, genre, topic, profile]);

  // Clear error after 5s
  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(""), 5000);
    return () => clearTimeout(timer);
  }, [error]);

  // ── Handlers ──────────────────────────────────────────────

  async function handleSendMessage(text: string) {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const allMessages = [...messages, userMsg];
      const data = await apiCall<GuideDialogueResponse>({
        action: "guide",
        mode: "dialogue",
        genre,
        topic,
        document,
        messages: allMessages,
      });
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.message,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to get response";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze() {
    setLoading(true);
    setError("");

    try {
      const data = await apiCall<AnalyzeResponse>({
        action: "analyze",
        genre,
        document,
      });

      setAnnotations(data.annotations);
      setExpandedAnnotation(null);

      // Update profile
      const snapshot = {
        date: new Date().toISOString().slice(0, 10),
        genre,
        wordCount: countWords(document),
        traitScores: data.traitScores,
        annotationCounts: {
          good: data.annotations.filter((a) => a.severity === "good").length,
          question: data.annotations.filter((a) => a.severity === "question").length,
          suggestion: data.annotations.filter((a) => a.severity === "suggestion").length,
          issue: data.annotations.filter((a) => a.severity === "issue").length,
        },
      };

      setProfile((prev) => {
        let updated = addAnalysisToProfile(prev, snapshot);
        if (!hasIncrementedThisSession) {
          updated = incrementGenreExperience(updated, genre);
          setHasIncrementedThisSession(true);
        }
        updated = markDayActive(updated);
        return updated;
      });

      // Conventions suppression recovery message
      if (previousConventionsSuppressed && !data.conventionsSuppressed) {
        const recoveryMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "Structure looks solid now — let's fine-tune the grammar.",
          timestamp: Date.now(),
        };
        setMessages((prev) => [recoveryMsg, ...prev]);
      }
      setPreviousConventionsSuppressed(data.conventionsSuppressed);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  async function handleAnnotationExpand(annotationId: string) {
    const annotation = annotations.find((a) => a.id === annotationId);
    if (!annotation) return;

    try {
      const data = await apiCall<ExpandResponse>({
        action: "expand",
        annotationId,
        annotationContext: annotation,
        document,
      });
      setExpandedAnnotation({ id: annotationId, ...data });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to expand annotation";
      setError(msg);
    }
  }

  function handleAnnotationClickFromEditor(id: string) {
    setFocusedAnnotationId(id);
    // Brief pulse then clear
    setTimeout(() => setFocusedAnnotationId(null), 2000);
  }

  function handleAnnotationClickFromPanel(id: string) {
    editorRef.current?.scrollToAnnotation(id);
  }

  function handleStepComplete(cardId: string) {
    setStepCards((prev) =>
      prev.map((c) => (c.id === cardId ? { ...c, completed: true } : c))
    );
  }

  function handleTipTryIt(exercisePrompt: string) {
    setDocument("");
    setTopic(exercisePrompt.slice(0, 60));
    setShowDailyTip(false);
  }

  function handleTipDisable() {
    setShowDailyTip(false);
    setProfile((prev) => ({
      ...prev,
      preferences: { ...prev.preferences, showDailyTips: false },
    }));
  }

  // Drag handlers
  const handleDragStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      isDragging.current = true;

      const onMove = (ev: MouseEvent) => {
        if (!isDragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const ratio =
          layoutPreset === "top"
            ? (ev.clientY - rect.top) / rect.height
            : (ev.clientX - rect.left) / rect.width;
        setSplitRatio(Math.max(0.2, Math.min(0.8, ratio)));
      };

      const onUp = () => {
        isDragging.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [layoutPreset]
  );

  function handleCopy() {
    const text = document
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  }

  function handleLayoutChange(preset: LayoutPreset) {
    setLayoutPreset(preset);
    setSplitRatio(0.5);
  }

  const wordCount = countWords(document);

  const gridStyle: React.CSSProperties =
    layoutPreset === "full"
      ? {}
      : layoutPreset === "side"
        ? { gridTemplateColumns: `${splitRatio}fr 4px ${1 - splitRatio}fr` }
        : { gridTemplateRows: `${splitRatio}fr 4px ${1 - splitRatio}fr` };

  const tabs: { key: Tab; label: string; disabled?: boolean }[] = [
    { key: "chat", label: "Chat" },
    { key: "dashboard", label: "Dashboard", disabled: true },
    { key: "lab", label: "Lab" },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 flex items-center gap-2 shrink-0">
          <span className="text-red-600 text-xs font-medium">{error}</span>
          <button
            onClick={() => setError("")}
            className="ml-auto text-red-400 hover:text-red-600 text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Toolbar */}
      <div className="h-10 bg-[var(--card)] border-b border-[var(--card-border)] flex items-center px-3 gap-2 shrink-0">
        <select
          value={genre}
          onChange={(e) => setGenre(e.target.value as Genre)}
          className="h-7 bg-[var(--background)] border border-[var(--card-border)] rounded-md text-xs text-[var(--foreground)] px-2 focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30"
        >
          {(Object.keys(GENRE_LABELS) as Genre[]).map((g) => (
            <option key={g} value={g}>
              {GENRE_LABELS[g]}
            </option>
          ))}
        </select>

        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Enter your topic..."
          className="flex-1 h-7 bg-[var(--background)] border border-[var(--card-border)] rounded-md text-xs text-[var(--foreground)] px-2.5 placeholder:text-[#c4bfb7] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30 min-w-0"
        />

        <button
          onClick={handleCopy}
          title="Copy plain text"
          className="h-7 px-2.5 bg-[var(--background)] border border-[var(--card-border)] rounded-md text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>

        <div className="flex items-center border border-[var(--card-border)] rounded-md overflow-hidden">
          <button
            onClick={() => handleLayoutChange("side")}
            title="Side by side"
            className={`h-7 w-7 flex items-center justify-center text-xs transition-colors ${
              layoutPreset === "side"
                ? "bg-[var(--accent)] text-white"
                : "bg-[var(--background)] text-[var(--muted)] hover:text-[var(--foreground)]"
            }`}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <rect x="1" y="1" width="12" height="12" rx="1" />
              <line x1="7" y1="1" x2="7" y2="13" />
            </svg>
          </button>
          <button
            onClick={() => handleLayoutChange("top")}
            title="Top and bottom"
            className={`h-7 w-7 flex items-center justify-center text-xs transition-colors ${
              layoutPreset === "top"
                ? "bg-[var(--accent)] text-white"
                : "bg-[var(--background)] text-[var(--muted)] hover:text-[var(--foreground)]"
            }`}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <rect x="1" y="1" width="12" height="12" rx="1" />
              <line x1="1" y1="7" x2="13" y2="7" />
            </svg>
          </button>
          <button
            onClick={() => handleLayoutChange("full")}
            title="Editor only"
            className={`h-7 w-7 flex items-center justify-center text-xs transition-colors ${
              layoutPreset === "full"
                ? "bg-[var(--accent)] text-white"
                : "bg-[var(--background)] text-[var(--muted)] hover:text-[var(--foreground)]"
            }`}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <rect x="1" y="1" width="12" height="12" rx="1" />
            </svg>
          </button>
        </div>
      </div>

      {/* Split pane content */}
      <div
        ref={containerRef}
        className={`flex-1 min-h-0 ${layoutPreset === "full" ? "" : "grid"}`}
        style={gridStyle}
      >
        {/* Editor pane */}
        <div className="min-h-0 min-w-0 flex flex-col bg-[var(--card)] overflow-auto">
          <div className="flex-1">
            <Editor
              ref={editorRef}
              content={document}
              onUpdate={setDocument}
              annotations={annotations}
              onAnnotationClick={handleAnnotationClickFromEditor}
            />
          </div>
          <div className="px-4 py-2 border-t border-[var(--card-border)] flex items-center">
            <button
              onClick={handleAnalyze}
              disabled={!document.trim() || loading}
              className="bg-[var(--accent)] hover:bg-[#b5583a] disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-xs font-medium px-4 py-1.5 rounded-md transition-all flex items-center gap-1.5"
            >
              {loading ? "Analyzing..." : "Analyze"}
              {!loading && (
                <span className="text-[10px] opacity-70">&#9654;</span>
              )}
            </button>
          </div>
        </div>

        {/* Drag divider */}
        {layoutPreset !== "full" && (
          <div
            onMouseDown={handleDragStart}
            className={`bg-[var(--card-border)] hover:bg-[var(--accent)]/40 transition-colors ${
              layoutPreset === "side"
                ? "cursor-col-resize"
                : "cursor-row-resize"
            }`}
          />
        )}

        {/* Collaboration pane */}
        {layoutPreset !== "full" && (
          <div className="min-h-0 min-w-0 flex flex-col bg-[var(--background)] overflow-hidden">
            {/* Tabs */}
            <div className="flex border-b border-[var(--card-border)] bg-[var(--card)] shrink-0">
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => !tab.disabled && setActiveTab(tab.key)}
                  disabled={tab.disabled}
                  className={`px-4 py-2.5 text-xs font-medium transition-colors relative ${
                    tab.disabled
                      ? "text-[var(--muted)]/50 cursor-not-allowed"
                      : activeTab === tab.key
                        ? "text-[var(--accent)]"
                        : "text-[var(--muted)] hover:text-[var(--foreground)]"
                  }`}
                >
                  {tab.label}
                  {activeTab === tab.key && !tab.disabled && (
                    <span className="absolute bottom-0 left-2 right-2 h-[2px] bg-[var(--accent)] rounded-full" />
                  )}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-auto">
              {activeTab === "chat" && (
                <ChatPanel
                  dailyTip={dailyTip}
                  showDailyTip={showDailyTip}
                  onTipTryIt={handleTipTryIt}
                  onTipSkip={() => setShowDailyTip(false)}
                  onTipDisable={handleTipDisable}
                  stepCards={stepCards}
                  onStepComplete={handleStepComplete}
                  messages={messages}
                  annotations={annotations}
                  focusedAnnotationId={focusedAnnotationId}
                  expandedAnnotation={expandedAnnotation}
                  onAnnotationExpand={handleAnnotationExpand}
                  onAnnotationClick={handleAnnotationClickFromPanel}
                  onSendMessage={handleSendMessage}
                  loading={loading}
                />
              )}

              {activeTab === "dashboard" && (
                <div className="flex items-center justify-center h-full p-4 text-[var(--muted)] text-sm text-center">
                  Dashboard — Coming soon. Track your writing progress here.
                </div>
              )}

              {activeTab === "lab" && (
                <div className="flex items-center justify-center h-full p-4 text-[var(--muted)] text-sm text-center">
                  Lab panel — coming soon. Practice exercises and rewriting
                  drills.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="h-6 bg-[var(--accent)] flex items-center px-3 gap-3 shrink-0">
        <span className="text-white/80 text-[10px]">
          {wordCount > 0 ? `${wordCount} words` : "Ready"}
        </span>
        <span className="text-white/50 text-[10px]">
          {GENRE_LABELS[genre]}
        </span>
        {annotations.length > 0 && (
          <span className="text-white/80 text-[10px]">
            {annotations.length} annotation
            {annotations.length !== 1 ? "s" : ""}
          </span>
        )}
        {profile.streak.current > 0 && (
          <span className="text-white/80 text-[10px]">
            {"\uD83D\uDD25"} {profile.streak.current}-day streak
          </span>
        )}
      </div>
    </div>
  );
}
