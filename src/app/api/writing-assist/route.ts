export const maxDuration = 60; // Vercel Fluid Compute: extend timeout for DeepSeek API calls

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import type {
  WritingAssistRequest,
  GuideStepResponse,
  GuideDialogueResponse,
  AnalyzeResponse,
  ExpandResponse,
  DailyTipResponse,
  LabRewriteResponse,
  Trait,
} from "@/lib/writing/types";
import { ANALYZE_SYSTEM_PROMPT } from "@/lib/prompts/writing/analyze";
import { GUIDE_STEP_SYSTEM_PROMPT } from "@/lib/prompts/writing/guide-step";
import { GUIDE_DIALOGUE_SYSTEM_PROMPT } from "@/lib/prompts/writing/guide-dialogue";
import { EXPAND_SYSTEM_PROMPT } from "@/lib/prompts/writing/expand";
import { DAILY_TIP_SYSTEM_PROMPT } from "@/lib/prompts/writing/daily-tip";
import { LAB_REWRITE_SYSTEM_PROMPT } from "@/lib/prompts/writing/lab-rewrite";
import { selectStaticTip } from "@/lib/writing/daily-tips";
import { AUTO_PROMPTS } from "@/lib/prompts/writing/auto-prompts";
import type { AutoExecuteResponse, GrammarOutput, PipelineOutputs } from "@/lib/writing/types";

const MODEL = "deepseek-chat";

function getClient(): OpenAI {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("API key not configured");
  return new OpenAI({ apiKey, baseURL: "https://api.deepseek.com" });
}

function stripMarkdownFencing(raw: string): string {
  return raw.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();
}

function parseJSON<T>(raw: string): T {
  try {
    return JSON.parse(stripMarkdownFencing(raw));
  } catch {
    throw new Error("Failed to parse AI response");
  }
}

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter((w) => w.length > 0).length;
}

async function handleGuideStep(body: WritingAssistRequest): Promise<NextResponse> {
  const client = getClient();
  const { genre, topic, experienceLevel } = body;

  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.3,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: GUIDE_STEP_SYSTEM_PROMPT },
      {
        role: "user",
        content: JSON.stringify({ genre, topic, experienceLevel: experienceLevel ?? 0 }),
      },
    ],
  });

  const data = parseJSON<GuideStepResponse>(res.choices[0].message.content ?? "");
  return NextResponse.json(data);
}

async function handleGuideDialogue(body: WritingAssistRequest): Promise<NextResponse> {
  const client = getClient();
  const { messages = [], document, blockSystemPrompt } = body;

  const systemPrompt = blockSystemPrompt || GUIDE_DIALOGUE_SYSTEM_PROMPT;

  const llmMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
  ];

  if (document) {
    llmMessages.push({
      role: "user",
      content: `[Current document]\n${document}`,
    });
  }

  for (const msg of messages) {
    llmMessages.push({ role: msg.role, content: msg.content });
  }

  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.7,
    messages: llmMessages,
  });

  const reply = res.choices[0].message.content ?? "";
  const data: GuideDialogueResponse = { type: "dialogue", message: reply };
  return NextResponse.json(data);
}

async function handleAnalyze(body: WritingAssistRequest): Promise<NextResponse> {
  const { document } = body;
  if (!document) {
    return NextResponse.json({ error: "Document is required" }, { status: 400 });
  }
  if (wordCount(document) < 100) {
    return NextResponse.json({ error: "Text must be at least 100 words" }, { status: 400 });
  }
  if (document.length > 15000) {
    return NextResponse.json({ error: "Text must be under 15,000 characters" }, { status: 400 });
  }

  const client = getClient();
  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: ANALYZE_SYSTEM_PROMPT },
      { role: "user", content: document },
    ],
  });

  const data = parseJSON<AnalyzeResponse>(res.choices[0].message.content ?? "");
  return NextResponse.json(data);
}

async function handleExpand(body: WritingAssistRequest): Promise<NextResponse> {
  const client = getClient();
  const { annotationContext, document } = body;

  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.3,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: EXPAND_SYSTEM_PROMPT },
      {
        role: "user",
        content: JSON.stringify({ annotation: annotationContext, document }),
      },
    ],
  });

  const data = parseJSON<ExpandResponse>(res.choices[0].message.content ?? "");
  return NextResponse.json(data);
}

async function handleDailyTip(body: WritingAssistRequest): Promise<NextResponse> {
  const { traitScores, analysisHistory } = body;
  const today = new Date().toISOString().slice(0, 10);

  try {
    const client = getClient();
    const res = await client.chat.completions.create({
      model: MODEL,
      temperature: 0.7,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: DAILY_TIP_SYSTEM_PROMPT },
        {
          role: "user",
          content: JSON.stringify({ traitScores, analysisHistory, date: today }),
        },
      ],
    });

    const data = parseJSON<DailyTipResponse>(res.choices[0].message.content ?? "");
    return NextResponse.json(data);
  } catch {
    const tip = selectStaticTip(
      (traitScores as Record<Trait, number>) ?? null,
      new Date(),
    );
    return NextResponse.json({ tip } satisfies DailyTipResponse);
  }
}

async function handleLabRewrite(body: WritingAssistRequest): Promise<NextResponse> {
  const { temperatures, text } = body;

  if (!text) {
    return NextResponse.json({ error: "Text is required" }, { status: 400 });
  }
  if (
    !temperatures ||
    !Array.isArray(temperatures) ||
    temperatures.length < 1 ||
    temperatures.length > 3 ||
    temperatures.some((t) => typeof t !== "number" || t < 0 || t > 2)
  ) {
    return NextResponse.json(
      { error: "temperatures must be an array of 1-3 numbers, each 0.0-2.0" },
      { status: 400 },
    );
  }

  const client = getClient();
  const rewrites = await Promise.all(
    temperatures.map(async (temp) => {
      const res = await client.chat.completions.create({
        model: MODEL,
        temperature: temp,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: LAB_REWRITE_SYSTEM_PROMPT },
          { role: "user", content: text },
        ],
      });
      const raw = res.choices[0].message.content ?? "{}";
      const parsed = parseJSON<{ text?: string; explanation?: string }>(raw);
      return {
        temperature: temp,
        text: parsed.text ?? raw,
        explanation: parsed.explanation ?? "",
      };
    }),
  );

  const data: LabRewriteResponse = { rewrites };
  return NextResponse.json(data);
}

async function handleAutoExecute(body: WritingAssistRequest): Promise<NextResponse> {
  const { blockType, genre, topic, language, document, previousOutputs } = body;

  if (!blockType || !["draft", "analyze", "grammar"].includes(blockType)) {
    return NextResponse.json({ error: "Invalid blockType" }, { status: 400 });
  }
  if (!topic) {
    return NextResponse.json({ error: "Topic is required" }, { status: 400 });
  }

  const client = getClient();
  const systemPrompt = AUTO_PROMPTS[`${blockType}-auto`];
  if (!systemPrompt) {
    return NextResponse.json({ error: `No auto prompt for ${blockType}` }, { status: 400 });
  }

  const lang = language === "zh" ? "Chinese" : "English";

  if (blockType === "draft") {
    const userContext = buildDraftContext(genre!, topic, lang, previousOutputs);
    const res = await client.chat.completions.create({
      model: MODEL,
      temperature: 0.7,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContext },
      ],
    });
    const text = res.choices[0].message.content ?? "";
    const result: AutoExecuteResponse = {
      dialogue: `Wrote a ${wordCount(text)}-word ${genre} essay based on your thesis and outline.`,
      documentUpdate: text.trim(),
    };
    return NextResponse.json(result);
  }

  if (blockType === "analyze") {
    if (!document) {
      return NextResponse.json({ error: "Document required for analyze" }, { status: 400 });
    }
    const res = await client.chat.completions.create({
      model: MODEL,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: document },
      ],
    });
    const data = parseJSON<{ annotations: unknown[]; traitScores: Record<string, number>; summary: string; conventionsSuppressed: boolean }>(
      res.choices[0].message.content ?? ""
    );
    const result: AutoExecuteResponse = {
      dialogue: `Analyzed the essay across 7 traits. ${data.summary}`,
      analysisResult: data as AnalyzeResponse,
    };
    return NextResponse.json(result);
  }

  // grammar
  if (!document) {
    return NextResponse.json({ error: "Document required for grammar" }, { status: 400 });
  }
  const res = await client.chat.completions.create({
    model: MODEL,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: document },
    ],
  });
  const data = parseJSON<GrammarOutput>(res.choices[0].message.content ?? "");
  const result: AutoExecuteResponse = {
    dialogue: `Fixed ${data.corrections.length} mechanical error${data.corrections.length !== 1 ? "s" : ""}.`,
    grammarResult: data,
  };
  return NextResponse.json(result);
}

function buildDraftContext(genre: string, topic: string, language: string, outputs?: PipelineOutputs): string {
  const parts = [`Genre: ${genre}`, `Topic: ${topic}`, `Language: ${language}`];
  if (outputs?.thesis?.userInput) parts.push(`Thesis: ${outputs.thesis.userInput}`);
  if (outputs?.outline?.userInput) parts.push(`Outline:\n${outputs.outline.userInput}`);
  if (outputs?.hook?.userInput) parts.push(`Hook concept: ${outputs.hook.userInput}`);
  return parts.join("\n\n");
}

export async function POST(req: NextRequest) {
  try {
    const body: WritingAssistRequest = await req.json();
    const { action } = body;

    switch (action) {
      case "guide":
        return body.mode === "dialogue"
          ? await handleGuideDialogue(body)
          : await handleGuideStep(body);

      case "analyze":
        return await handleAnalyze(body);

      case "expand":
        return await handleExpand(body);

      case "daily-tip":
        return await handleDailyTip(body);

      case "lab-rewrite":
        return await handleLabRewrite(body);

      case "auto-execute":
        return await handleAutoExecute(body);

      case "report":
        return NextResponse.json({ error: "Not available yet" }, { status: 501 });

      default:
        return NextResponse.json(
          { error: `Invalid action: ${action}` },
          { status: 400 },
        );
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
