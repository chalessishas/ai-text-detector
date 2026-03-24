"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
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
import type { BlockInstance, BlockType, BlockCategory } from "@/lib/writing/blocks";
import { getBlockDef, createBlockInstance, PHASE_TRANSITION_PROMPTS } from "@/lib/writing/blocks";
import { apiCall } from "@/lib/writing/api";
import Editor, { type EditorHandle } from "./Editor";
import ChatPanel from "./ChatPanel";
import LabPanel from "./LabPanel";
import Workbench from "./Workbench";
import BlockChat from "./BlockChat";
import ChecklistPanel from "./ChecklistPanel";

// ── Constants ──

const GENRE_LABELS: Record<Genre, string> = {
  essay: "Essay",
  article: "Article",
  academic: "Academic",
  creative: "Creative",
  business: "Business",
};

type LayoutPreset = "side" | "top" | "full";
type Tab = "chat" | "dashboard" | "lab";

// Phase state machine:
//   workbench → block-chat (for chat-mode blocks)
//   workbench → writing    (for editor/lab/analyze blocks)
//   workbench → checklist  (for checklist-mode blocks)
type Phase = "workbench" | "block-chat" | "writing" | "checklist";

const BOARD_STORAGE_KEY = "writing-center-board";

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


// ── Board persistence ──

function loadBoard(): BlockInstance[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(BOARD_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveBoardToStorage(board: BlockInstance[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(BOARD_STORAGE_KEY, JSON.stringify(board));
  } catch {}
}

// ── Main Component ──

export default function WritingCenter() {
  // Phase & block navigation
  const [phase, setPhase] = useState<Phase>("workbench");
  const [board, setBoard] = useState<BlockInstance[]>(() => loadBoard());
  const [activeBlockId, setActiveBlockId] = useState<string | null>(null);

  // Core writing state (shared across blocks)
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

  // Daily tip
  const [dailyTip, setDailyTip] = useState<DailyTip | null>(null);
  const [showDailyTip, setShowDailyTip] = useState(true);

  // Step cards
  const [stepCards, setStepCards] = useState<StepCard[]>([]);

  // Expanded annotation
  const [expandedAnnotation, setExpandedAnnotation] = useState<{
    id: string;
    detail: string;
    suggestion?: string;
    question: string;
  } | null>(null);

  // Conventions suppression tracking
  const [previousConventionsSuppressed, setPreviousConventionsSuppressed] =
    useState(false);

  // Phase-transition micro-reflection (Yancey model)
  const [reflectionPrompt, setReflectionPrompt] = useState<string | null>(null);

  // Layout drag
  const [splitRatio, setSplitRatio] = useState(0.5);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const editorRef = useRef<EditorHandle>(null);
  const sessionWordsRef = useRef(0);

  // ── Derived state ──

  const activeBlock = board.find((b) => b.id === activeBlockId) ?? null;
  const activeBlockDef = activeBlock ? getBlockDef(activeBlock.type) : null;

  // ── Lifecycle ──

  // Check if user has a saved draft → jump to editor with board context
  useEffect(() => {
    const saved = loadState();
    if (saved.draft.document && saved.draft.document.trim()) {
      setDocument(saved.draft.document);
      setGenre(saved.draft.genre);
      setTopic(saved.draft.topic);
      if (saved.draft.messages.length) setMessages(saved.draft.messages);
      if (saved.draft.annotations.length) setAnnotations(saved.draft.annotations);
      // If board has a draft block, activate it; otherwise just go to editor
      const draftBlock = board.find((b) => b.type === "draft");
      if (draftBlock) {
        setActiveBlockId(draftBlock.id);
        updateBlockStatus(draftBlock.id, "active");
      }
      setPhase("writing");
    }
    if (saved.profile) setProfile(saved.profile);
  }, []);

  // Streak on mount
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Guide step cards when entering writing phase with genre + topic
  useEffect(() => {
    if (phase !== "writing" || !genre || !topic || document.trim()) return;
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
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, genre, topic]);

  // Memoize word count (used in render + streak tracking)
  const wordCount = useMemo(() => countWords(document), [document]);

  // Word tracking for streak
  useEffect(() => {
    const delta = wordCount - sessionWordsRef.current;
    if (delta > 0) sessionWordsRef.current = wordCount;
    if (sessionWordsRef.current >= 50) {
      setProfile((prev) => markDayActive(prev));
    }
  }, [wordCount]);

  // Auto-save writing state
  useEffect(() => {
    if (phase === "workbench") return;
    const timer = setTimeout(() => {
      saveState({
        draft: { genre, topic, document, messages, annotations, lastSaved: Date.now() },
        profile,
      });
    }, 2000);
    return () => clearTimeout(timer);
  }, [phase, document, messages, annotations, genre, topic, profile]);

  // Save board (debounced to collapse rapid mutations)
  useEffect(() => {
    const timer = setTimeout(() => saveBoardToStorage(board), 300);
    return () => clearTimeout(timer);
  }, [board]);

  // Clear error
  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(""), 5000);
    return () => clearTimeout(timer);
  }, [error]);

  // ── Board handlers ──

  function updateBlockStatus(blockId: string, status: BlockInstance["status"]) {
    setBoard((prev) =>
      prev.map((b) => (b.id === blockId ? { ...b, status } : b))
    );
  }

  function handleBlockClick(block: BlockInstance) {
    const def = getBlockDef(block.type);

    // Check for phase-transition micro-reflection
    if (activeBlock) {
      const prevCategory = getBlockDef(activeBlock.type).category;
      const nextCategory = def.category;
      if (prevCategory !== nextCategory) {
        const key = `${prevCategory}->${nextCategory}` as `${BlockCategory}->${BlockCategory}`;
        const prompt = PHASE_TRANSITION_PROMPTS[key];
        if (prompt) setReflectionPrompt(prompt);
      }
    }

    setActiveBlockId(block.id);
    updateBlockStatus(block.id, "active");

    switch (def.mode) {
      case "chat":
        setPhase("block-chat");
        break;
      case "checklist":
        setPhase("checklist");
        break;
      case "editor":
        setPhase("writing");
        break;
      case "lab":
        setActiveTab("lab");
        setPhase("writing");
        break;
    }
  }

  function handleStartWriting() {
    // Start with the first undone block
    const first = board.find((b) => b.status !== "done") ?? board[0];
    if (first) {
      handleBlockClick(first);
    }
  }

  function handleBlockDone(blockId: string, output?: string) {
    setBoard((prev) => {
      const updated = prev.map((b) =>
        b.id === blockId ? { ...b, status: "done" as const, output: output ?? b.output } : b
      );
      // Auto-advance to next undone block, or return to workbench
      const currentIdx = updated.findIndex((b) => b.id === blockId);
      const next = updated.find((b, i) => i > currentIdx && b.status !== "done");
      if (next) {
        // Schedule navigation outside the updater to avoid nested setState
        queueMicrotask(() => handleBlockClick(next));
      } else {
        queueMicrotask(() => {
          setActiveBlockId(null);
          setPhase("workbench");
        });
      }
      return updated;
    });
  }

  function handleBackToBoard() {
    if (activeBlockId) {
      updateBlockStatus(activeBlockId, board.find(b => b.id === activeBlockId)?.output ? "done" : "todo");
    }
    setActiveBlockId(null);
    setPhase("workbench");
  }

  // ── Writing phase handlers ──

  async function handleSendMessage(text: string) {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    if (!topic && messages.length <= 2) {
      setTopic(text.slice(0, 80));
    }

    try {
      const allMessages = [...messages, userMsg];
      const data = await apiCall<GuideDialogueResponse>({
        action: "guide",
        mode: "dialogue",
        genre,
        topic: topic || text,
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

      if (previousConventionsSuppressed && !data.conventionsSuppressed) {
        const recoveryMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Structure looks solid now — let's fine-tune the grammar.",
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
      const msg = err instanceof Error ? err.message : "Failed to expand";
      setError(msg);
    }
  }

  function handleAnnotationClickFromEditor(id: string) {
    setFocusedAnnotationId(id);
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
    setTopic(exercisePrompt.slice(0, 60));
    setGenre("essay");
    setShowDailyTip(false);
    const existing = board.find((b) => b.type === "draft");
    if (existing) {
      handleBlockClick(existing);
    } else {
      const newBlock = createBlockInstance("draft");
      setBoard((prev) => [...prev, newBlock]);
      setActiveBlockId(newBlock.id);
      setPhase("writing");
    }
  }

  function handleTipDisable() {
    setShowDailyTip(false);
    setProfile((prev) => ({
      ...prev,
      preferences: { ...prev.preferences, showDailyTips: false },
    }));
  }

  async function handleLabRewrite(
    text: string,
    temperatures: number[]
  ): Promise<{ temperature: number; text: string; explanation: string }[]> {
    const res = await apiCall<{
      rewrites: { temperature: number; text: string; explanation: string }[];
    }>({ action: "lab-rewrite", text, temperatures });
    return res.rewrites;
  }

  function handleLabYourTurn(coldText: string, labTopic: string) {
    setDocument(`<p>${coldText}</p>`);
    setTopic(labTopic);
    setAnnotations([]);
    setStepCards([]);
    setHasIncrementedThisSession(false);
    setActiveTab("chat");
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
    const text = document.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
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

  // ── Render ──

  // Phase: Workbench (block selection & arrangement)
  if (phase === "workbench") {
    return (
      <Workbench
        board={board}
        onBoardChange={setBoard}
        onBlockClick={handleBlockClick}
        onStartWriting={handleStartWriting}
        streak={profile.streak.current}
      />
    );
  }

  // Micro-reflection banner (shown during phase transitions)
  const reflectionBanner = reflectionPrompt ? (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-3 flex items-center gap-3 shrink-0">
      <span className="text-amber-700 text-xs leading-relaxed flex-1">{reflectionPrompt}</span>
      <button
        onClick={() => setReflectionPrompt(null)}
        className="text-amber-500 hover:text-amber-700 text-xs font-medium px-2 py-1 rounded shrink-0"
      >
        Got it
      </button>
    </div>
  ) : null;

  // Phase: Block chat (Socratic chat for a specific block)
  if (phase === "block-chat" && activeBlock) {
    return (
      <>
        {reflectionBanner}
        <BlockChat
          block={activeBlock}
          sharedContext={{ topic, document, genre }}
          onDone={(output) => handleBlockDone(activeBlock.id, output)}
          onBack={handleBackToBoard}
        />
      </>
    );
  }

  // Phase: Checklist (self-review, peer-review, submit-ready)
  if (phase === "checklist" && activeBlock) {
    return (
      <>
        {reflectionBanner}
        <ChecklistPanel
          block={activeBlock}
          document={document}
          profile={profile}
          onDone={(output) => handleBlockDone(activeBlock.id, output)}
          onBack={handleBackToBoard}
        />
      </>
    );
  }

  // Phase: Writing (editor + analysis + lab)
  // This is the full writing environment, now with block navigation
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
      {/* Micro-reflection banner */}
      {reflectionBanner}

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

      {/* Block navigation bar (shows when board has blocks) */}
      {board.length > 0 && (
        <div className="h-9 bg-[var(--card)] border-b border-[var(--card-border)] flex items-center px-3 gap-1 shrink-0 overflow-x-auto">
          <button
            onClick={handleBackToBoard}
            className="text-[10px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors mr-2 shrink-0"
          >
            ← Board
          </button>
          {board.map((block, idx) => {
            const def = getBlockDef(block.type);
            const isActive = block.id === activeBlockId;
            return (
              <button
                key={block.id}
                onClick={() => handleBlockClick(block)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium transition-all shrink-0 ${
                  isActive
                    ? "text-white"
                    : block.status === "done"
                      ? "bg-green-50 text-green-700"
                      : "text-[var(--muted)] hover:bg-[var(--background)]"
                }`}
                style={isActive ? { background: def.color } : undefined}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: isActive ? "white" : def.color }}
                />
                {def.nameZh}
                {block.status === "done" && !isActive && (
                  <span className="text-[8px] opacity-60">&#10003;</span>
                )}
              </button>
            );
          })}
          {/* Done button for current block */}
          {activeBlock && (
            <button
              onClick={() => handleBlockDone(activeBlock.id)}
              className="ml-auto text-[10px] font-medium px-2.5 py-1 rounded-md transition-all shrink-0"
              style={{
                background: activeBlockDef ? `${activeBlockDef.color}15` : undefined,
                color: activeBlockDef?.color,
                border: activeBlockDef ? `1px solid ${activeBlockDef.color}30` : undefined,
              }}
            >
              Done →
            </button>
          )}
        </div>
      )}

      {/* Toolbar */}
      <div className="h-10 bg-[var(--card)] border-b border-[var(--card-border)] flex items-center px-3 gap-2 shrink-0">
        {board.length === 0 && (
          <button
            onClick={handleBackToBoard}
            className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors mr-1"
            title="Back to workbench"
          >
            ← Back
          </button>
        )}

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
          placeholder="Topic..."
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
          {(["side", "top", "full"] as LayoutPreset[]).map((preset) => (
            <button
              key={preset}
              onClick={() => handleLayoutChange(preset)}
              title={preset === "side" ? "Side by side" : preset === "top" ? "Top and bottom" : "Editor only"}
              className={`h-7 w-7 flex items-center justify-center text-xs transition-colors ${
                layoutPreset === preset
                  ? "bg-[var(--accent)] text-white"
                  : "bg-[var(--background)] text-[var(--muted)] hover:text-[var(--foreground)]"
              }`}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="1" y="1" width="12" height="12" rx="1" />
                {preset === "side" && <line x1="7" y1="1" x2="7" y2="13" />}
                {preset === "top" && <line x1="1" y1="7" x2="13" y2="7" />}
              </svg>
            </button>
          ))}
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
          <div className="px-4 py-2 border-t border-[var(--card-border)] flex items-center gap-3">
            <button
              onClick={handleAnalyze}
              disabled={wordCount < 100 || loading}
              title={wordCount < 100 ? `Need ${100 - wordCount} more words` : "Analyze your writing"}
              className="bg-[var(--accent)] hover:bg-[#b5583a] disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-xs font-medium px-4 py-1.5 rounded-md transition-all flex items-center gap-1.5"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
            {wordCount < 100 && wordCount > 0 && (
              <span className="text-[10px] text-[var(--muted)]">
                {100 - wordCount} words to go
              </span>
            )}
          </div>
        </div>

        {/* Drag divider */}
        {layoutPreset !== "full" && (
          <div
            onMouseDown={handleDragStart}
            className={`bg-[var(--card-border)] hover:bg-[var(--accent)]/40 transition-colors ${
              layoutPreset === "side" ? "cursor-col-resize" : "cursor-row-resize"
            }`}
          />
        )}

        {/* Collaboration pane */}
        {layoutPreset !== "full" && (
          <div className="min-h-0 min-w-0 flex flex-col bg-[var(--background)] overflow-hidden">
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
                  Dashboard — Coming soon.
                </div>
              )}
              {activeTab === "lab" && (
                <LabPanel
                  onYourTurn={handleLabYourTurn}
                  loading={loading}
                  onLabRewrite={handleLabRewrite}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div
        className="h-6 flex items-center px-3 gap-3 shrink-0"
        style={{ background: activeBlockDef?.color ?? "var(--accent)" }}
      >
        <span className="text-white/80 text-[10px]">
          {wordCount > 0 ? `${wordCount} words` : "Ready"}
        </span>
        {activeBlockDef && (
          <span className="text-white/80 text-[10px]">
            {activeBlockDef.nameZh}
          </span>
        )}
        <span className="text-white/50 text-[10px]">{GENRE_LABELS[genre]}</span>
        {annotations.length > 0 && (
          <span className="text-white/80 text-[10px]">
            {annotations.length} annotation{annotations.length !== 1 ? "s" : ""}
          </span>
        )}
        {profile.streak.current > 0 && (
          <span className="text-white/80 text-[10px]">
            {profile.streak.current}-day streak
          </span>
        )}
      </div>
    </div>
  );
}
