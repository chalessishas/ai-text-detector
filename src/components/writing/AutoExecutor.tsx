"use client";

import { useState, useRef, useCallback, useMemo } from "react";
import type { BlockType } from "@/lib/writing/blocks";
import { getBlockDef } from "@/lib/writing/blocks";
import type {
  Genre,
  Annotation,
  PipelineOutputs,
  PipelineStatus,
  AutoExecuteResponse,
} from "@/lib/writing/types";
import { CHECKPOINT_DESCRIPTIONS } from "@/lib/prompts/writing/auto-prompts";
import { apiCall } from "@/lib/writing/api";
import Editor, { type EditorHandle } from "./Editor";

import type { BlockInstance } from "@/lib/writing/blocks";

interface AutoExecutorProps {
  board: BlockInstance[];
  genre: Genre;
  topic: string;
  language: "en" | "zh";
  onExit: () => void;
}

function textToHtml(text: string): string {
  return text
    .split("\n\n")
    .filter(Boolean)
    .map((p) => `<p>${p.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>`)
    .join("");
}

export default function AutoExecutor({ board, genre, topic, language, onExit }: AutoExecutorProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [status, setStatus] = useState<PipelineStatus>("checkpoint");
  const [outputs, setOutputs] = useState<PipelineOutputs>({});
  const [document, setDocument] = useState("");
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [checkpointInput, setCheckpointInput] = useState("");
  // AI-generated options for checkpoint-with-options blocks
  const [aiOptions, setAiOptions] = useState("");
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [executionLog, setExecutionLog] = useState<Array<{ blockType: string; dialogue: string; timestamp: number }>>([]);
  // Text results from analysis blocks (shown in execution area)
  const [textResults, setTextResults] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const editorRef = useRef<EditorHandle>(null);
  const [autoDocVersion, setAutoDocVersion] = useState(0);

  const currentBlock = board[currentIndex];
  const currentDef = currentBlock ? getBlockDef(currentBlock.type) : null;
  const isFinished = status === "done" || status === "cancelled";
  const editorDisabled = !isFinished && !isEditing;

  // Determine if current block is a checkpoint (user provides input)
  const isCheckpoint = currentDef?.auto?.checkpoint === true;
  // Pure checkpoint = thesis (no AI options)
  const isPureCheckpoint = isCheckpoint && currentBlock?.type === "thesis";
  // Options checkpoint = brainstorm, audience, outline, hook (AI generates options first)
  const isOptionsCheckpoint = isCheckpoint && !isPureCheckpoint;

  // Call API to generate AI options for checkpoint-with-options blocks
  const fetchOptions = useCallback(async (blockIndex: number, currentOutputs: PipelineOutputs, currentDoc: string) => {
    const block = board[blockIndex];
    setLoadingOptions(true);
    setAiOptions("");
    try {
      const res = await apiCall<AutoExecuteResponse>({
        action: "auto-execute",
        blockType: block.type,
        genre,
        topic,
        language,
        document: currentDoc,
        previousOutputs: currentOutputs,
      });
      setAiOptions(res.optionsResult || res.textResult || "");
      setLoadingOptions(false);
    } catch (err) {
      setLoadingOptions(false);
      // Options generation failed — let user provide input manually
      setAiOptions(language === "zh" ? "(AI 生成选项失败，请手动输入)" : "(Failed to generate options — please type manually)");
    }
  }, [board, genre, topic, language]);

  // Execute a non-checkpoint block (AI does the work)
  const executeBlock = useCallback(async (blockIndex: number, currentOutputs: PipelineOutputs, currentDoc: string) => {
    const block = board[blockIndex];

    setStatus("executing");
    setCurrentIndex(blockIndex);

    const res = await apiCall<AutoExecuteResponse>({
      action: "auto-execute",
      blockType: block.type,
      genre,
      topic,
      language,
      document: currentDoc,
      previousOutputs: currentOutputs,
    });

    const newOutputs = { ...currentOutputs };
    let newDoc = currentDoc;

    // Handle response based on block type
    if (res.documentUpdate) {
      // Doc-producing or doc-modifying blocks
      newDoc = res.documentUpdate;
      if (block.type === "draft") {
        newOutputs.draft = { document: newDoc };
      } else {
        // Doc-modifying blocks (evidence, counterargument, conclusion, transitions, style-edit)
        (newOutputs as Record<string, unknown>)[block.type] = { document: newDoc, changes: res.dialogue };
      }
    } else if (block.type === "analyze" && res.analysisResult) {
      newOutputs.analyze = res.analysisResult;
      setAnnotations(res.analysisResult.annotations ?? []);
    } else if (block.type === "grammar" && res.grammarResult) {
      newDoc = res.grammarResult.document;
      newOutputs.grammar = res.grammarResult;
    } else if (res.textResult) {
      // Text-output blocks (research, evidence-eval, peer-review, reader-sim, logic-check, voice-lab, submit-ready, self-review)
      (newOutputs as Record<string, unknown>)[block.type] = { text: res.textResult };
      setTextResults((prev) => ({ ...prev, [block.type]: res.textResult! }));
    }

    setExecutionLog((prev) => [...prev, {
      blockType: block.type,
      dialogue: res.dialogue,
      timestamp: Date.now(),
    }]);

    return { nextIndex: blockIndex + 1, outputs: newOutputs, doc: newDoc };
  }, [board, genre, topic, language]);

  // Run blocks until we hit a checkpoint or finish
  const runUntilCheckpoint = useCallback(async (startIndex: number, currentOutputs: PipelineOutputs, currentDoc: string) => {
    let idx = startIndex;
    let outs = currentOutputs;
    let doc = currentDoc;

    while (idx < board.length) {
      const block = board[idx];
      const def = getBlockDef(block.type);

      // Checkpoint block — pause for user input
      if (def.auto?.checkpoint) {
        setCurrentIndex(idx);
        setOutputs(outs);
        setDocument(doc);
        setAutoDocVersion((v) => v + 1);
        setStatus("checkpoint");
        setCheckpointInput("");
        setAiOptions("");
        // For options checkpoints, fetch AI options
        if (block.type !== "thesis") {
          fetchOptions(idx, outs, doc);
        }
        return;
      }

      // Block without auto config — skip (shouldn't happen now, but safety)
      if (!def.auto) {
        idx++;
        continue;
      }

      // Auto-executable block — AI does the work
      try {
        const result = await executeBlock(idx, outs, doc);
        outs = result.outputs;
        doc = result.doc;
        idx = result.nextIndex;
        setOutputs(outs);
        setDocument(doc);
        setAutoDocVersion((v) => v + 1);
      } catch (err) {
        setCurrentIndex(idx);
        setError(err instanceof Error ? err.message : "Execution failed");
        setStatus("error");
        return;
      }
    }

    // All blocks done
    setOutputs(outs);
    setDocument(doc);
    setAutoDocVersion((v) => v + 1);
    setStatus("done");
  }, [board, executeBlock, fetchOptions]);

  function handleCheckpointSubmit() {
    if (!checkpointInput.trim()) return;

    const newOutputs = { ...outputs };
    const key = currentBlock.type as string;
    if (isOptionsCheckpoint) {
      (newOutputs as Record<string, unknown>)[key] = { userInput: checkpointInput.trim(), aiOptions };
    } else {
      (newOutputs as Record<string, unknown>)[key] = { userInput: checkpointInput.trim() };
    }

    setOutputs(newOutputs);
    runUntilCheckpoint(currentIndex + 1, newOutputs, document);
  }

  function handleContinue() {
    setIsEditing(false);
    runUntilCheckpoint(currentIndex + 1, outputs, document);
  }

  function handleStop() {
    setStatus("cancelled");
  }

  function handleRetry() {
    setError("");
    runUntilCheckpoint(currentIndex, outputs, document);
  }

  // ── Progress bar ──
  const progressBar = (
    <div className="h-10 bg-[var(--card)] border-b border-[var(--card-border)] flex items-center px-3 gap-1 overflow-x-auto shrink-0">
      <button
        onClick={onExit}
        className="text-[10px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors mr-2 shrink-0"
      >
        ← Board
      </button>
      {board.map((block, idx) => {
        const def = getBlockDef(block.type);
        const isCurrent = idx === currentIndex;
        const isDone = idx < currentIndex;
        return (
          <div
            key={block.id}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium shrink-0 ${
              isCurrent
                ? "text-white"
                : isDone
                  ? "bg-green-50 text-green-700"
                  : "text-[var(--muted)]"
            }`}
            style={isCurrent ? { background: def.color } : undefined}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: isCurrent ? "white" : isDone ? "#22c55e" : def.color }}
            />
            {def.nameZh}
            {isDone && <span className="text-[8px] opacity-60">&#10003;</span>}
          </div>
        );
      })}
    </div>
  );

  const checkpointDesc = currentDef?.auto?.promptKey
    ? CHECKPOINT_DESCRIPTIONS[currentDef.auto.promptKey]
    : null;

  // ── Execution area (right panel) ──
  const executionArea = (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* Executing spinner */}
        {status === "executing" && (
          <div className="text-center py-8 space-y-3">
            <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin mx-auto" />
            <div className="text-sm text-[var(--foreground)]">
              {language === "zh" ? "AI 正在执行" : "AI is working on"} {currentDef?.nameZh}...
            </div>
          </div>
        )}

        {/* Checkpoint: pure input (thesis only) */}
        {status === "checkpoint" && isPureCheckpoint && checkpointDesc && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-[var(--foreground)]">
              {language === "zh" ? checkpointDesc.titleZh : checkpointDesc.title}
            </h3>
            <textarea
              value={checkpointInput}
              onChange={(e) => setCheckpointInput(e.target.value)}
              placeholder={language === "zh" ? checkpointDesc.placeholderZh : checkpointDesc.placeholder}
              className="w-full h-32 bg-[var(--background)] border border-[var(--card-border)] rounded-lg text-sm text-[var(--foreground)] p-3 resize-none focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30 placeholder:text-[#c4bfb7]"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={handleCheckpointSubmit}
                disabled={!checkpointInput.trim()}
                className="bg-[var(--accent)] hover:bg-[#b5583a] disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-xs font-medium px-4 py-2 rounded-md transition-all"
              >
                {language === "zh" ? "继续" : "Continue"} →
              </button>
              <button
                onClick={handleStop}
                className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-2 transition-colors"
              >
                {language === "zh" ? "停止" : "Stop"}
              </button>
            </div>
          </div>
        )}

        {/* Checkpoint with AI options (brainstorm, audience, outline, hook) */}
        {status === "checkpoint" && isOptionsCheckpoint && checkpointDesc && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-[var(--foreground)]">
              {language === "zh" ? checkpointDesc.titleZh : checkpointDesc.title}
            </h3>

            {/* AI options */}
            {loadingOptions ? (
              <div className="flex items-center gap-2 py-4">
                <div className="w-4 h-4 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-[var(--muted)]">
                  {language === "zh" ? "AI 正在生成选项..." : "AI generating options..."}
                </span>
              </div>
            ) : aiOptions ? (
              <div className="bg-[var(--background)] border border-[var(--card-border)] rounded-lg p-3 max-h-60 overflow-auto">
                <p className="text-[10px] text-[var(--accent)] font-medium mb-2">
                  {language === "zh" ? "AI 建议：" : "AI suggests:"}
                </p>
                <pre className="text-xs text-[var(--foreground)] whitespace-pre-wrap leading-relaxed font-[inherit]">
                  {aiOptions}
                </pre>
              </div>
            ) : null}

            {/* User input */}
            <textarea
              value={checkpointInput}
              onChange={(e) => setCheckpointInput(e.target.value)}
              placeholder={language === "zh" ? checkpointDesc.placeholderZh : checkpointDesc.placeholder}
              className="w-full h-24 bg-[var(--background)] border border-[var(--card-border)] rounded-lg text-sm text-[var(--foreground)] p-3 resize-none focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30 placeholder:text-[#c4bfb7]"
            />
            <div className="flex gap-2">
              <button
                onClick={handleCheckpointSubmit}
                disabled={!checkpointInput.trim()}
                className="bg-[var(--accent)] hover:bg-[#b5583a] disabled:bg-[#d4cfc7] disabled:text-[#a09a92] text-white text-xs font-medium px-4 py-2 rounded-md transition-all"
              >
                {language === "zh" ? "继续" : "Continue"} →
              </button>
              <button
                onClick={handleStop}
                className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-2 transition-colors"
              >
                {language === "zh" ? "停止" : "Stop"}
              </button>
            </div>
          </div>
        )}

        {/* Review checkpoint (after auto blocks complete) */}
        {status === "checkpoint" && !isCheckpoint && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              <h3 className="text-sm font-medium text-[var(--foreground)]">
                {board[currentIndex - 1] ? getBlockDef(board[currentIndex - 1].type).nameZh : ""} {language === "zh" ? "完成" : "complete"}
              </h3>
            </div>
            <p className="text-xs text-[var(--muted)] leading-relaxed">
              {executionLog[executionLog.length - 1]?.dialogue}
            </p>

            {/* Show text results for analysis blocks */}
            {(() => {
              const lastBlock = board[currentIndex - 1];
              const lastResult = lastBlock ? textResults[lastBlock.type] : null;
              if (!lastResult) return null;
              return (
                <div className="bg-[var(--background)] border border-[var(--card-border)] rounded-lg p-3 max-h-48 overflow-auto">
                  <pre className="text-xs text-[var(--foreground)] whitespace-pre-wrap leading-relaxed font-[inherit]">
                    {lastResult}
                  </pre>
                </div>
              );
            })()}

            {isEditing ? (
              <div className="space-y-2">
                <p className="text-xs text-[var(--accent)]">
                  {language === "zh" ? "编辑器已解锁，完成后点击继续。" : "Editor unlocked. Click Resume when done."}
                </p>
                <button
                  onClick={() => { setIsEditing(false); handleContinue(); }}
                  className="bg-[var(--accent)] hover:bg-[#b5583a] text-white text-xs font-medium px-4 py-2 rounded-md transition-all"
                >
                  {language === "zh" ? "继续" : "Resume"} →
                </button>
              </div>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={handleContinue}
                  className="bg-[var(--accent)] hover:bg-[#b5583a] text-white text-xs font-medium px-4 py-2 rounded-md transition-all"
                >
                  {language === "zh" ? "继续到" : "Continue to"} {currentDef?.nameZh} →
                </button>
                <button
                  onClick={() => setIsEditing(true)}
                  className="border border-[var(--card-border)] text-xs text-[var(--foreground)] px-3 py-2 rounded-md hover:bg-[var(--background)] transition-colors"
                >
                  {language === "zh" ? "编辑文章" : "Edit document"}
                </button>
                <button
                  onClick={handleStop}
                  className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-2 transition-colors"
                >
                  {language === "zh" ? "到此为止" : "Stop here"}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="space-y-3">
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-xs text-red-600">{error}</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleRetry}
                className="bg-[var(--accent)] hover:bg-[#b5583a] text-white text-xs font-medium px-4 py-2 rounded-md transition-all"
              >
                Retry
              </button>
              <button
                onClick={handleStop}
                className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] px-3 py-2 transition-colors"
              >
                Stop
              </button>
            </div>
          </div>
        )}

        {/* Finished */}
        {isFinished && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              <h3 className="text-sm font-medium text-[var(--foreground)]">
                {status === "done"
                  ? (language === "zh" ? "全部完成" : "All done")
                  : (language === "zh" ? "已停止" : "Stopped")}
              </h3>
            </div>
            <p className="text-xs text-[var(--muted)]">
              {language === "zh" ? "你现在可以自由编辑文章。" : "You can now freely edit the document."}
            </p>
            <button
              onClick={onExit}
              className="text-xs text-[var(--accent)] hover:underline"
            >
              ← {language === "zh" ? "返回积木板" : "Back to board"}
            </button>
          </div>
        )}

        {/* Execution log */}
        {executionLog.length > 0 && (
          <div className="border-t border-[var(--card-border)] pt-3">
            <button
              onClick={() => setShowLog(!showLog)}
              className="text-[10px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              {showLog ? "▾" : "▸"} {language === "zh" ? "执行记录" : "Execution log"} ({executionLog.length})
            </button>
            {showLog && (
              <div className="mt-2 space-y-2">
                {executionLog.map((entry, i) => (
                  <div key={i} className="text-[10px] text-[var(--muted)] leading-relaxed">
                    <span className="font-medium">{getBlockDef(entry.blockType as BlockType).nameZh}:</span>{" "}
                    {entry.dialogue}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const editorContent = useMemo(() => textToHtml(document), [autoDocVersion]);

  return (
    <div className="flex flex-col h-full">
      {progressBar}

      <div className="flex-1 min-h-0 grid" style={{ gridTemplateColumns: "1fr 4px 1fr" }}>
        <div className="min-h-0 min-w-0 flex flex-col bg-[var(--card)] overflow-auto">
          <div className="flex-1">
            <Editor
              ref={editorRef}
              content={editorContent}
              onUpdate={(html) => {
                if (!editorDisabled) {
                  const text = html.replace(/<\/p>\s*<p[^>]*>/g, "\n\n").replace(/<[^>]*>/g, "").trim();
                  setDocument(text);
                }
              }}
              annotations={annotations}
              onAnnotationClick={() => {}}
              disabled={editorDisabled}
            />
          </div>
        </div>

        <div className="bg-[var(--card-border)]" />

        <div className="min-h-0 min-w-0 bg-[var(--background)] overflow-hidden">
          {executionArea}
        </div>
      </div>

      <div
        className="h-6 flex items-center px-3 gap-3 shrink-0"
        style={{ background: currentDef?.color ?? "var(--accent)" }}
      >
        <span className="text-white/80 text-[10px]">
          {status === "executing"
            ? `${language === "zh" ? "执行中" : "Running"}: ${currentDef?.nameZh} (${currentIndex + 1}/${board.length})`
            : status === "checkpoint"
              ? `${language === "zh" ? "等待输入" : "Waiting"}: ${currentDef?.nameZh} (${currentIndex + 1}/${board.length})`
              : status === "done"
                ? (language === "zh" ? "完成" : "Done")
                : status === "cancelled"
                  ? (language === "zh" ? "已停止" : "Stopped")
                  : (language === "zh" ? "错误" : "Error")}
        </span>
        {document && (
          <span className="text-white/80 text-[10px]">
            {language === "zh"
              ? `${document.replace(/\s/g, "").length} 字`
              : `${document.split(/\s+/).filter(Boolean).length} words`}
          </span>
        )}
        {!isFinished && (
          <button
            onClick={handleStop}
            className="ml-auto text-white/60 hover:text-white text-[10px] transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
