export const maxDuration = 120;

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const HUMANIZER_SERVER =
  process.env.HUMANIZER_SERVER_URL || "http://127.0.0.1:5002";
const PERPLEXITY_SERVER =
  process.env.PERPLEXITY_SERVER_URL || "http://127.0.0.1:5001";
const MAX_TEXT_LENGTH = 10000;

export interface MethodResult {
  text: string;
  score?: number;
  template?: string;
  swaps?: { from: string; to: string }[];
  splitPoint?: number;
}

export type MethodKey =
  | "corpus"
  | "structure"
  | "transplant"
  | "inject"
  | "harvest"
  | "remix"
  | "anchor"
  | "adversarial";

export interface HumanizeSentenceDetail {
  original: string;
  methods: Partial<Record<MethodKey, MethodResult>>;
}

export interface HumanizeResult {
  humanized: string;
  sentenceCount: number;
  details: HumanizeSentenceDetail[];
  adversarial?: AdversarialResult;
}

// Adversarial Paraphrasing: iteratively rewrite until detector is fooled
interface AdversarialResult {
  text: string;
  iterations: number;
  initialScore: number;
  finalScore: number;
  history: { iteration: number; score: number; preview: string }[];
}

async function getDetectorScore(text: string): Promise<number> {
  const res = await fetch(PERPLEXITY_SERVER, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) return 50;
  const data = await res.json();
  return data.fused?.ai_score ?? data.classification?.ai_score ?? 50;
}

const ADVERSARIAL_SYSTEM_PROMPT = `You are a writing style editor. Rewrite the given text to sound more naturally human-written.

Rules:
- Preserve ALL factual content and meaning
- Add natural imperfections: contractions, varied sentence length, informal transitions
- Use first-person perspective where appropriate
- Vary sentence starters (avoid starting consecutive sentences the same way)
- Mix short punchy sentences with longer flowing ones
- Replace formal transition words (furthermore, moreover, additionally) with casual ones (also, plus, and)
- Add occasional hedging language (I think, probably, seems like)
- Do NOT add new information or change the meaning
- Output ONLY the rewritten text, no explanations`;

async function adversarialParaphrase(text: string): Promise<AdversarialResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DEEPSEEK_API_KEY not configured");

  const client = new OpenAI({ apiKey, baseURL: "https://api.deepseek.com" });
  const MAX_ITER = 4;
  const TARGET_SCORE = 40; // below this = considered human

  const initialScore = await getDetectorScore(text);
  const history: AdversarialResult["history"] = [
    { iteration: 0, score: initialScore, preview: text.slice(0, 80) },
  ];

  // Already passes — no rewrite needed
  if (initialScore <= TARGET_SCORE) {
    return { text, iterations: 0, initialScore, finalScore: initialScore, history };
  }

  let current = text;
  let currentScore = initialScore;

  for (let i = 1; i <= MAX_ITER; i++) {
    const emphasis = currentScore > 70
      ? "The text still reads very AI-like. Make more dramatic changes to sentence structure and word choice."
      : "The text is close to human-like. Make subtle adjustments to sentence rhythm and transitions.";

    const completion = await client.chat.completions.create({
      model: "deepseek-chat",
      messages: [
        { role: "system", content: ADVERSARIAL_SYSTEM_PROMPT },
        { role: "user", content: `${emphasis}\n\nRewrite this text:\n\n${current}` },
      ],
      temperature: 0.8 + i * 0.1, // increase randomness each iteration
      max_tokens: Math.max(text.length * 2, 2000),
    });

    const rewritten = completion.choices[0]?.message?.content?.trim();
    if (!rewritten) break;

    const newScore = await getDetectorScore(rewritten);
    history.push({ iteration: i, score: newScore, preview: rewritten.slice(0, 80) });

    // Only accept if score improved
    if (newScore < currentScore) {
      current = rewritten;
      currentScore = newScore;
    }

    if (currentScore <= TARGET_SCORE) break;
  }

  return {
    text: current,
    iterations: history.length - 1,
    initialScore,
    finalScore: currentScore,
    history,
  };
}

export async function POST(req: NextRequest) {
  try {
    const { text, method } = await req.json();

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json({ error: "Text is required" }, { status: 400 });
    }

    const trimmed = text.trim();

    if (trimmed.length > MAX_TEXT_LENGTH) {
      return NextResponse.json(
        { error: `Text too long (max ${MAX_TEXT_LENGTH} characters)` },
        { status: 400 }
      );
    }

    // Adversarial mode: iterative LLM rewrite guided by detector
    if (method === "adversarial") {
      const result = await adversarialParaphrase(trimmed);
      return NextResponse.json(result);
    }

    // Default: FAISS-based humanizer (7 methods)
    const pyRes = await fetch(HUMANIZER_SERVER, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed, topK: 20 }),
      signal: AbortSignal.timeout(60000),
    }).catch(() => {
      throw new Error(
        "Cannot reach humanizer server. Is 'npm run humanizer' running?"
      );
    });

    if (!pyRes.ok) {
      throw new Error(`Humanizer server returned HTTP ${pyRes.status}`);
    }

    const result = await pyRes.json();

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 500 });
    }

    return NextResponse.json(result);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
