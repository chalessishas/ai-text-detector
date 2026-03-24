import { NextRequest, NextResponse } from "next/server";

const HUMANIZER_SERVER =
  process.env.HUMANIZER_SERVER_URL || "http://127.0.0.1:5002";
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
  | "anchor";
// ⚠️ 以下四种方法经验证无效，已从后端禁用：phrase, collocation, noise, splice

export interface HumanizeSentenceDetail {
  original: string;
  methods: Partial<Record<MethodKey, MethodResult>>;
}

export interface HumanizeResult {
  humanized: string;
  sentenceCount: number;
  details: HumanizeSentenceDetail[];
}

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json();

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
