"use client";

import { useState, useEffect } from "react";
import type { BlockInstance } from "@/lib/writing/blocks";
import { getBlockDef } from "@/lib/writing/blocks";
import { apiCall } from "@/lib/writing/api";
import type { WriterProfile, Trait } from "@/lib/writing/types";

interface ChecklistItem {
  id: string;
  text: string;
  checked: boolean;
  category?: string;
}

interface ChecklistPanelProps {
  block: BlockInstance;
  document: string;
  profile: WriterProfile;
  onDone: (output: string) => void;
  onBack: () => void;
}

function generateSelfReviewChecklist(
  document: string,
  profile: WriterProfile
): ChecklistItem[] {
  const items: ChecklistItem[] = [];
  let id = 0;

  // Core structure checks (always present)
  items.push(
    { id: `c${id++}`, text: "My thesis is clearly stated and arguable", checked: false, category: "Ideas" },
    { id: `c${id++}`, text: "Every paragraph supports or develops my thesis", checked: false, category: "Organization" },
    { id: `c${id++}`, text: "I have an introduction that hooks the reader", checked: false, category: "Organization" },
    { id: `c${id++}`, text: "My conclusion does more than repeat the introduction", checked: false, category: "Organization" },
  );

  // Evidence checks
  items.push(
    { id: `c${id++}`, text: "Each major claim is supported by evidence", checked: false, category: "Ideas" },
    { id: `c${id++}`, text: "I introduce and explain my evidence, not just drop quotes", checked: false, category: "Ideas" },
  );

  // Voice and style
  items.push(
    { id: `c${id++}`, text: "I can hear my own voice in this writing, not a generic AI tone", checked: false, category: "Voice" },
    { id: `c${id++}`, text: "Sentence lengths vary — not all the same pattern", checked: false, category: "Fluency" },
  );

  // Adaptive items based on weak traits from analysis history
  const lastAnalysis = profile.analysisHistory[profile.analysisHistory.length - 1];
  if (lastAnalysis?.traitScores) {
    const weakTraits = (Object.entries(lastAnalysis.traitScores) as [Trait, number][])
      .filter(([, score]) => score < 60)
      .sort(([, a], [, b]) => a - b);

    for (const [trait] of weakTraits.slice(0, 3)) {
      switch (trait) {
        case "wordChoice":
          items.push({ id: `c${id++}`, text: "I've replaced vague words (things, stuff, very) with precise alternatives", checked: false, category: "Word Choice" });
          break;
        case "conventions":
          items.push({ id: `c${id++}`, text: "I've read the piece aloud to catch grammar and punctuation errors", checked: false, category: "Conventions" });
          break;
        case "fluency":
          items.push({ id: `c${id++}`, text: "I've checked that transitions connect my paragraphs logically", checked: false, category: "Fluency" });
          break;
        case "voice":
          items.push({ id: `c${id++}`, text: "I've removed phrases that sound like they came from a template or AI", checked: false, category: "Voice" });
          break;
        case "ideas":
          items.push({ id: `c${id++}`, text: "I've asked: would a skeptic find my argument convincing?", checked: false, category: "Ideas" });
          break;
        case "organization":
          items.push({ id: `c${id++}`, text: "I can explain why my paragraphs are in this specific order", checked: false, category: "Organization" });
          break;
        case "presentation":
          items.push({ id: `c${id++}`, text: "Formatting is consistent: headings, spacing, citations", checked: false, category: "Presentation" });
          break;
      }
    }
  }

  // Final metacognitive check
  items.push(
    { id: `c${id++}`, text: "I'm submitting my best work, not my first draft", checked: false, category: "Reflection" },
  );

  return items;
}

function generateSubmitReadyChecklist(): ChecklistItem[] {
  let id = 0;
  return [
    { id: `s${id++}`, text: "Word count meets the assignment requirements", checked: false, category: "Format" },
    { id: `s${id++}`, text: "Title page / header follows the required format", checked: false, category: "Format" },
    { id: `s${id++}`, text: "Font, spacing, and margins match the style guide", checked: false, category: "Format" },
    { id: `s${id++}`, text: "All in-text citations have matching references", checked: false, category: "Citations" },
    { id: `s${id++}`, text: "Reference list / bibliography is complete and formatted correctly", checked: false, category: "Citations" },
    { id: `s${id++}`, text: "Page numbers are present and correct", checked: false, category: "Format" },
    { id: `s${id++}`, text: "All figures/tables are numbered and referenced in the text", checked: false, category: "Format" },
    { id: `s${id++}`, text: "File is saved in the required format (PDF, DOCX, etc.)", checked: false, category: "Submission" },
    { id: `s${id++}`, text: "I've read the assignment prompt one final time to check I addressed everything", checked: false, category: "Submission" },
  ];
}

function generatePeerReviewChecklist(): ChecklistItem[] {
  let id = 0;
  return [
    { id: `p${id++}`, text: "I can identify the writer's thesis in the first few paragraphs", checked: false, category: "Ideas" },
    { id: `p${id++}`, text: "The argument is convincing — I can follow the logic", checked: false, category: "Ideas" },
    { id: `p${id++}`, text: "Each paragraph has a clear purpose", checked: false, category: "Organization" },
    { id: `p${id++}`, text: "Evidence is introduced and analyzed, not just dropped in", checked: false, category: "Ideas" },
    { id: `p${id++}`, text: "I noted where I got confused or lost interest", checked: false, category: "Engagement" },
    { id: `p${id++}`, text: "I identified at least 2 things that work well", checked: false, category: "Strengths" },
    { id: `p${id++}`, text: "I identified at least 2 specific areas for improvement", checked: false, category: "Growth" },
    { id: `p${id++}`, text: "My feedback focuses on IDEAS and STRUCTURE, not just grammar", checked: false, category: "Focus" },
  ];
}

export default function ChecklistPanel({
  block,
  document,
  profile,
  onDone,
  onBack,
}: ChecklistPanelProps) {
  const def = getBlockDef(block.type);

  const [items, setItems] = useState<ChecklistItem[]>(() => {
    switch (block.type) {
      case "self-review": return generateSelfReviewChecklist(document, profile);
      case "submit-ready": return generateSubmitReadyChecklist();
      case "peer-review": return generatePeerReviewChecklist();
      default: return [];
    }
  });

  function toggleItem(id: string) {
    setItems((prev) => prev.map((item) =>
      item.id === id ? { ...item, checked: !item.checked } : item
    ));
  }

  const checkedCount = items.filter((i) => i.checked).length;
  const progress = items.length > 0 ? Math.round((checkedCount / items.length) * 100) : 0;
  const allDone = checkedCount === items.length;

  function handleDone() {
    const summary = items
      .map((i) => `${i.checked ? "[x]" : "[ ]"} ${i.text}`)
      .join("\n");
    onDone(`Checklist: ${checkedCount}/${items.length} completed\n\n${summary}`);
  }

  // Group by category
  const grouped = items.reduce<Record<string, ChecklistItem[]>>((acc, item) => {
    const cat = item.category || "General";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(item);
    return acc;
  }, {});

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="h-11 bg-[var(--card)] border-b border-[var(--card-border)] flex items-center px-4 shrink-0 gap-3">
        <button
          onClick={onBack}
          className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        >
          ← Board
        </button>
        <span
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ background: def.color }}
        />
        <span className="text-sm font-medium text-[var(--foreground)]">
          {def.nameZh}
        </span>
        <span className="text-[10px] text-[var(--muted)]">{def.name}</span>
        <div className="flex-1" />
        <span className="text-[10px] text-[var(--muted)]">
          {checkedCount}/{items.length}
        </span>
        <button
          onClick={handleDone}
          className="text-xs font-medium px-3 py-1 rounded-md transition-all"
          style={{
            background: allDone ? def.color : `${def.color}15`,
            color: allDone ? "white" : def.color,
            border: `1px solid ${def.color}30`,
          }}
        >
          {allDone ? "Complete" : "Done"}
        </button>
      </div>

      {/* AI role hint */}
      <div className="px-4 py-2 bg-[var(--background)] border-b border-[var(--card-border)] shrink-0">
        <p className="text-[10px] text-[var(--muted)] leading-relaxed max-w-2xl mx-auto">
          {def.aiRole}
        </p>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-[var(--card-border)] shrink-0">
        <div
          className="h-full transition-all duration-300"
          style={{ width: `${progress}%`, background: def.color }}
        />
      </div>

      {/* Checklist */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-2xl mx-auto p-6 space-y-6">
          {Object.entries(grouped).map(([category, categoryItems]) => (
            <div key={category}>
              <div className="text-[10px] font-medium text-[var(--muted)] uppercase tracking-wider mb-2">
                {category}
              </div>
              <div className="space-y-1">
                {categoryItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => toggleItem(item.id)}
                    className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                      item.checked
                        ? "bg-green-50 border border-green-200"
                        : "bg-[var(--card)] border border-[var(--card-border)] hover:border-[var(--accent)]/30"
                    }`}
                  >
                    <span className={`mt-0.5 w-4 h-4 rounded border-2 flex items-center justify-center shrink-0 transition-all ${
                      item.checked
                        ? "border-green-500 bg-green-500 text-white"
                        : "border-[var(--card-border)]"
                    }`}>
                      {item.checked && (
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M2 5l2 2 4-4" />
                        </svg>
                      )}
                    </span>
                    <span className={`text-sm leading-relaxed ${
                      item.checked ? "text-green-700 line-through opacity-70" : "text-[var(--foreground)]"
                    }`}>
                      {item.text}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
