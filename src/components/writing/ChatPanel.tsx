"use client";

import { useState, useRef, useEffect } from "react";
import type {
  DailyTip,
  StepCard,
  ChatMessage,
  Annotation,
  Severity,
} from "@/lib/writing/types";
import DailyTipCard from "./DailyTipCard";
import StepCardComponent from "./StepCard";
import AnnotationCard from "./AnnotationCard";

const SEVERITY_ORDER: Severity[] = ["good", "question", "suggestion", "issue"];

interface ChatPanelProps {
  dailyTip: DailyTip | null;
  showDailyTip: boolean;
  onTipTryIt: (prompt: string) => void;
  onTipSkip: () => void;
  onTipDisable: () => void;
  stepCards: StepCard[];
  onStepComplete: (cardId: string) => void;
  messages: ChatMessage[];
  annotations: Annotation[];
  focusedAnnotationId: string | null;
  expandedAnnotation: {
    id: string;
    detail: string;
    suggestion?: string;
    question: string;
  } | null;
  onAnnotationExpand: (annotationId: string) => void;
  onAnnotationClick: (annotationId: string) => void;
  onSendMessage: (text: string) => void;
  loading: boolean;
}

export default function ChatPanel({
  dailyTip,
  showDailyTip,
  onTipTryIt,
  onTipSkip,
  onTipDisable,
  stepCards,
  onStepComplete,
  messages,
  annotations,
  focusedAnnotationId,
  expandedAnnotation,
  onAnnotationExpand,
  onAnnotationClick,
  onSendMessage,
  loading,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, annotations, stepCards]);

  // Sort annotations in Liz Lerman order
  const sortedAnnotations = [...annotations].sort(
    (a, b) =>
      SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity)
  );

  function handleSend() {
    const text = draft.trim();
    if (!text || loading) return;
    onSendMessage(text);
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setDraft(e.target.value);
    // Auto-resize
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }

  return (
    <div className="flex flex-col h-full">
      {/* Scrollable content */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Daily tip */}
        {showDailyTip && dailyTip && (
          <DailyTipCard
            tip={dailyTip}
            onTryIt={onTipTryIt}
            onSkip={onTipSkip}
            onDisable={onTipDisable}
          />
        )}

        {/* Step cards */}
        {stepCards.map((card) => (
          <StepCardComponent
            key={card.id}
            card={card}
            onComplete={onStepComplete}
          />
        ))}

        {/* Chat messages */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed ${
                msg.role === "user"
                  ? "bg-[var(--accent)] text-white rounded-br-sm"
                  : "bg-[var(--background)] text-[var(--foreground)] border border-[var(--card-border)] rounded-bl-sm"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {/* Annotation cards */}
        {sortedAnnotations.map((ann) => (
          <AnnotationCard
            key={ann.id}
            annotation={ann}
            expanded={expandedAnnotation?.id === ann.id}
            expandedDetail={
              expandedAnnotation?.id === ann.id ? expandedAnnotation : null
            }
            onExpand={onAnnotationExpand}
            onClick={onAnnotationClick}
            isFocused={focusedAnnotationId === ann.id}
          />
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-[var(--background)] border border-[var(--card-border)] rounded-xl px-4 py-2 rounded-bl-sm">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-[var(--muted)] rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-1.5 h-1.5 bg-[var(--muted)] rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 bg-[var(--muted)] rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sticky input */}
      <div className="border-t border-[var(--card-border)] p-3 bg-[var(--card)]">
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your writing..."
            rows={1}
            className="flex-1 resize-none bg-[var(--background)] border border-[var(--card-border)] rounded-lg px-3 py-2 text-xs text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
          <button
            onClick={handleSend}
            disabled={!draft.trim() || loading}
            className="px-3 py-2 bg-[var(--accent)] text-white rounded-lg text-xs font-medium hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
