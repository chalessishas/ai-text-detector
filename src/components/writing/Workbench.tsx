"use client";

import { useState } from "react";
import type { BlockInstance, BlockType, BlockCategory, BlockPreset } from "@/lib/writing/blocks";
import {
  BLOCK_CATALOG,
  BLOCK_PRESETS,
  CATEGORY_ORDER,
  CATEGORY_LABELS,
  getBlockDef,
  createBlockInstance,
} from "@/lib/writing/blocks";

interface WorkbenchProps {
  board: BlockInstance[];
  onBoardChange: (board: BlockInstance[]) => void;
  onBlockClick: (block: BlockInstance) => void;
  onStartWriting: () => void;
  onAutoStart?: () => void;
  language: "en" | "zh";
  onLanguageChange: (lang: "en" | "zh") => void;
  streak: number;
}

export default function Workbench({
  board,
  onBoardChange,
  onBlockClick,
  onStartWriting,
  onAutoStart,
  language,
  onLanguageChange,
  streak,
}: WorkbenchProps) {
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [dragOverIdx, setDragOverIdx] = useState<number | null>(null);

  // Auto-start enabled when board has blocks with auto configs
  const isAutoEnabled = board.length > 0
    && board.some((b) => getBlockDef(b.type).auto != null);

  // Group catalog by category
  const grouped = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    ...CATEGORY_LABELS[cat],
    blocks: BLOCK_CATALOG.filter((b) => b.category === cat),
  }));

  function addBlock(type: BlockType) {
    // Don't add duplicates
    if (board.some((b) => b.type === type)) return;
    onBoardChange([...board, createBlockInstance(type)]);
  }

  function removeBlock(id: string) {
    onBoardChange(board.filter((b) => b.id !== id));
  }

  function applyPreset(preset: BlockPreset) {
    const newBoard = preset.blocks.map((type) => {
      // Preserve existing block instance if already on board
      const existing = board.find((b) => b.type === type);
      return existing ?? createBlockInstance(type);
    });
    onBoardChange(newBoard);
  }

  function handleDragStart(idx: number) {
    setDragIdx(idx);
  }

  function handleDragOver(e: React.DragEvent, idx: number) {
    e.preventDefault();
    setDragOverIdx(idx);
  }

  function handleDrop(idx: number) {
    if (dragIdx === null || dragIdx === idx) {
      setDragIdx(null);
      setDragOverIdx(null);
      return;
    }
    const next = [...board];
    const [moved] = next.splice(dragIdx, 1);
    next.splice(idx, 0, moved);
    onBoardChange(next);
    setDragIdx(null);
    setDragOverIdx(null);
  }

  const isOnBoard = (type: BlockType) => board.some((b) => b.type === type);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto p-6 pb-24 space-y-8">
          {/* Header */}
          <div className="text-center space-y-2">
            <h2 className="text-2xl font-semibold text-[var(--foreground)]">
              Writing Center
            </h2>
            <p className="text-[var(--muted)] text-sm max-w-lg mx-auto">
              Pick your blocks, arrange them, start writing.
              Every piece of writing is different — build the process that fits yours.
            </p>
          </div>

          {/* Presets */}
          <div>
            <div className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider mb-3">
              Quick Start
            </div>
            <div className="flex gap-2 flex-wrap">
              {BLOCK_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => applyPreset(preset)}
                  className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg px-4 py-2.5 text-left hover:border-[var(--accent)]/40 hover:shadow-sm transition-all group"
                >
                  <div className="text-sm font-medium text-[var(--foreground)] group-hover:text-[var(--accent)] transition-colors">
                    {preset.nameZh}
                  </div>
                  <div className="text-[10px] text-[var(--muted)] mt-0.5">
                    {preset.description}
                  </div>
                  <div className="flex gap-1 mt-2">
                    {preset.blocks.map((type) => {
                      const def = getBlockDef(type);
                      return (
                        <span
                          key={type}
                          className="w-2 h-2 rounded-full"
                          style={{ background: def.color }}
                          title={def.nameZh}
                        />
                      );
                    })}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Board */}
          <div>
            <div className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider mb-3">
              Your Board
              {board.length > 0 && (
                <span className="ml-2 normal-case tracking-normal font-normal">
                  — drag to reorder, click to start
                </span>
              )}
            </div>

            {board.length === 0 ? (
              <div className="bg-[var(--card)] border-2 border-dashed border-[var(--card-border)] rounded-xl p-8 text-center">
                <p className="text-sm text-[var(--muted)]">
                  Pick blocks below or choose a Quick Start above
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {board.map((block, idx) => {
                  const def = getBlockDef(block.type);
                  const isOver = dragOverIdx === idx;
                  return (
                    <div
                      key={block.id}
                      draggable
                      onDragStart={() => handleDragStart(idx)}
                      onDragOver={(e) => handleDragOver(e, idx)}
                      onDragEnd={() => { setDragIdx(null); setDragOverIdx(null); }}
                      onDrop={() => handleDrop(idx)}
                      onClick={() => onBlockClick(block)}
                      className={`flex items-center gap-3 bg-[var(--card)] border rounded-lg px-4 py-3 cursor-pointer transition-all group ${
                        isOver
                          ? "border-[var(--accent)] shadow-md"
                          : "border-[var(--card-border)] hover:border-[var(--accent)]/40 hover:shadow-sm"
                      }`}
                    >
                      {/* Drag handle */}
                      <span className="text-[var(--muted)]/40 group-hover:text-[var(--muted)] transition-colors cursor-grab text-sm select-none">
                        ⠿
                      </span>

                      {/* Step number */}
                      <span className="text-[10px] font-mono text-[var(--muted)] w-4 text-center shrink-0">
                        {idx + 1}
                      </span>

                      {/* Color dot */}
                      <span
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{ background: def.color }}
                      />

                      {/* Name + description */}
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-[var(--foreground)]">
                          {def.nameZh}
                          <span className="text-[10px] text-[var(--muted)] font-normal ml-2">
                            {def.name}
                          </span>
                        </div>
                        <div className="text-[11px] text-[var(--muted)] truncate">
                          {def.aiRole}
                        </div>
                      </div>

                      {/* Status badge */}
                      <span
                        className={`text-[10px] font-medium px-2 py-0.5 rounded-full shrink-0 ${
                          block.status === "done"
                            ? "bg-green-50 text-green-700"
                            : block.status === "active"
                              ? "bg-amber-50 text-amber-700"
                              : "bg-[var(--background)] text-[var(--muted)]"
                        }`}
                      >
                        {block.status === "done" ? "Done" : block.status === "active" ? "In progress" : "To do"}
                      </span>

                      {/* Remove button */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeBlock(block.id);
                        }}
                        className="text-[var(--muted)]/40 hover:text-red-400 transition-colors text-xs shrink-0"
                        title="Remove from board"
                      >
                        x
                      </button>
                    </div>
                  );
                })}

                {/* Start buttons */}
                <div className="space-y-2 mt-4">
                  <button
                    onClick={onStartWriting}
                    className="w-full bg-[var(--foreground)] hover:bg-[var(--foreground)]/90 text-white text-sm font-medium py-3 rounded-lg transition-all"
                  >
                    Start with {getBlockDef(board[0].type).nameZh} →
                  </button>
                  {onAutoStart && (
                    <>
                      {isAutoEnabled && (
                        <div className="flex items-center justify-center gap-2">
                          <span className="text-[10px] text-[var(--muted)]">Language:</span>
                          <button
                            onClick={() => onLanguageChange(language === "en" ? "zh" : "en")}
                            className="text-[10px] font-medium px-2 py-0.5 rounded border border-[var(--card-border)] hover:border-[var(--accent)]/40 transition-colors"
                          >
                            {language === "en" ? "English" : "中文"}
                          </button>
                        </div>
                      )}
                      <button
                        onClick={onAutoStart}
                        disabled={!isAutoEnabled}
                        title={!isAutoEnabled ? "Add blocks to enable auto mode" : "AI executes blocks, you provide ideas at checkpoints"}
                        className={`w-full text-sm font-medium py-3 rounded-lg transition-all border ${
                          isAutoEnabled
                            ? "bg-[var(--accent)] hover:bg-[#b5583a] text-white border-transparent"
                            : "bg-[var(--background)] text-[var(--muted)] border-[var(--card-border)] cursor-not-allowed"
                        }`}
                      >
                        Auto-start →
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Block palette */}
          <div>
            <div className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider mb-4">
              All Blocks
            </div>
            <div className="space-y-5">
              {grouped.map(({ category, nameZh, name, blocks }) => (
                <div key={category}>
                  <div className="text-[11px] text-[var(--muted)] mb-2">
                    {nameZh}
                    <span className="text-[10px] ml-1.5 opacity-60">{name}</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {blocks.map((def) => {
                      const onBoard = isOnBoard(def.type);
                      return (
                        <button
                          key={def.type}
                          onClick={() => addBlock(def.type)}
                          disabled={onBoard}
                          className={`text-left rounded-lg px-3 py-2.5 border transition-all ${
                            onBoard
                              ? "bg-[var(--background)] border-[var(--card-border)] opacity-50 cursor-default"
                              : "bg-[var(--card)] border-[var(--card-border)] hover:border-[var(--accent)]/40 hover:shadow-sm cursor-pointer"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2.5 h-2.5 rounded-full shrink-0"
                              style={{ background: def.color }}
                            />
                            <span className="text-xs font-medium text-[var(--foreground)]">
                              {def.nameZh}
                            </span>
                            {onBoard && (
                              <span className="text-[9px] text-[var(--muted)] ml-auto">
                                on board
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-[var(--muted)] mt-1 leading-relaxed line-clamp-2">
                            {def.description}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Streak */}
          {streak > 0 && (
            <div className="text-center text-xs text-[var(--muted)]">
              {streak}-day writing streak
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
