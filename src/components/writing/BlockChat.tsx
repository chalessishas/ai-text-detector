"use client";

import { useState, useRef, useEffect } from "react";
import type { BlockDef, BlockInstance } from "@/lib/writing/blocks";
import { getBlockDef, BLOCK_SYSTEM_PROMPTS } from "@/lib/writing/blocks";
import { apiCall } from "@/lib/writing/api";
import type { ChatMessage } from "@/lib/writing/types";

interface BlockChatProps {
  block: BlockInstance;
  sharedContext: {
    topic: string;
    document: string;
    genre: string;
  };
  onDone: (output: string) => void;
  onBack: () => void;
}

export default function BlockChat({
  block,
  sharedContext,
  onDone,
  onBack,
}: BlockChatProps) {
  const def = getBlockDef(block.type);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Opening message on mount
  useEffect(() => {
    const opener = getOpenerMessage(def, sharedContext);
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: opener,
        timestamp: Date.now(),
      },
    ]);
  }, []);

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      const systemPrompt = BLOCK_SYSTEM_PROMPTS[block.type] || "";
      const contextNote = sharedContext.topic
        ? `\n\nContext: The writer is working on "${sharedContext.topic}" (${sharedContext.genre}).`
        : "";

      const data = await apiCall<{ message: string }>({
        action: "guide",
        mode: "dialogue",
        genre: sharedContext.genre || "essay",
        topic: sharedContext.topic || text,
        document: sharedContext.document || "",
        messages: updated,
        blockSystemPrompt: systemPrompt + contextNote,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.message,
          timestamp: Date.now(),
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Something went wrong. Try again?",
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Collect all user messages as the block's output
  function handleDone() {
    const userMessages = messages
      .filter((m) => m.role === "user")
      .map((m) => m.content)
      .join("\n\n");
    onDone(userMessages);
  }

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
        <button
          onClick={handleDone}
          className="text-xs font-medium px-3 py-1 rounded-md transition-all"
          style={{
            background: `${def.color}15`,
            color: def.color,
            border: `1px solid ${def.color}30`,
          }}
        >
          Done
        </button>
      </div>

      {/* AI role hint */}
      <div className="px-4 py-2 bg-[var(--background)] border-b border-[var(--card-border)] shrink-0">
        <p className="text-[10px] text-[var(--muted)] leading-relaxed max-w-2xl mx-auto">
          {def.aiRole}
        </p>
      </div>

      {/* Chat area */}
      <div ref={scrollRef} className="flex-1 overflow-auto">
        <div className="max-w-2xl mx-auto p-6 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "text-white"
                    : "bg-[var(--card)] border border-[var(--card-border)] text-[var(--foreground)]"
                }`}
                style={
                  msg.role === "user" ? { background: def.color } : undefined
                }
              >
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-xl px-4 py-3">
                <span className="flex gap-1">
                  <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-[var(--muted)] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-[var(--card-border)] bg-[var(--card)] p-3 shrink-0">
        <div className="max-w-2xl mx-auto flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your thoughts..."
            rows={1}
            className="flex-1 bg-[var(--background)] border border-[var(--card-border)] rounded-lg px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[#c4bfb7] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30 resize-none"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 py-2 text-white text-sm font-medium rounded-lg transition-all disabled:opacity-40"
            style={{ background: def.color }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

// Generate the opening message based on block type and context
function getOpenerMessage(
  def: BlockDef,
  ctx: { topic: string; document: string; genre: string }
): string {
  const topicNote = ctx.topic ? ` about "${ctx.topic}"` : "";

  switch (def.type) {
    case "brainstorm":
      return ctx.topic
        ? `Let's explore your ideas${topicNote}. What's the one thing you most want your reader to take away?`
        : "What's on your mind? Tell me what you're thinking about writing, and I'll help you find your direction.";

    case "audience":
      return `Who are you writing this for${topicNote}? Describe your ideal reader — what do they already know, and what do they care about?`;

    case "thesis":
      return ctx.topic
        ? `Time to sharpen your argument${topicNote}. In one sentence, what's your position? Don't worry about making it perfect — we'll refine it together.`
        : "What do you believe that others might disagree with? Start there, and we'll craft it into a thesis.";

    case "outline":
      return ctx.topic
        ? `Let's build the structure${topicNote}. What are the 2-4 main points you want to make? List them in any order — we'll figure out the best sequence together.`
        : "What's the skeleton of your piece? Give me your main ideas, even if they're rough.";

    case "research":
      return `What claims in your writing need evidence? Let's figure out what types of sources would be most convincing for your argument${topicNote}.`;

    case "counterargument":
      return ctx.document
        ? "I've read your draft. Let me challenge your argument — not to tear it down, but to make it stronger. Ready?"
        : `Tell me your main argument${topicNote}, and I'll play devil's advocate. The goal: find the weak spots before your reader does.`;

    case "logic-check":
      return ctx.document
        ? "Let me trace the logic of your argument step by step. I'll flag anything that feels like a leap or an unsupported claim."
        : `Walk me through your reasoning${topicNote}. I'll check for gaps and fallacies.`;

    case "style-edit":
      return ctx.document
        ? "Let's look at your writing sentence by sentence. Share a paragraph you'd like to polish, and I'll point out what could be tighter."
        : "Share a paragraph of your writing, and I'll help you see where the style can be sharpened — with explanations of why.";

    default:
      return `Let's work on ${def.nameZh}${topicNote}. What would you like to start with?`;
  }
}
