import { NextRequest, NextResponse } from "next/server";
import {
  processTokens,
  computeGLTR,
  computeEntropyStats,
  computeVocabulary,
  computeSentences,
  computeSentenceLengthStats,
  computeBurstiness,
  computeSlidingWindow,
  computeOverallPerplexity,
  computeFeatureScores,
  computeSentenceScores,
  computeAISimilarityTags,
  detectAIVocab,
  type AnalysisResult,
  type ClassificationResult,
} from "@/lib/analysis";

const PERPLEXITY_SERVER =
  process.env.PERPLEXITY_SERVER_URL || "http://127.0.0.1:5001";
const MAX_TEXT_LENGTH = 10000;

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

    const pyRes = await fetch(PERPLEXITY_SERVER, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed }),
      signal: AbortSignal.timeout(60000),
    }).catch(() => {
      throw new Error(
        "Cannot reach analysis server. Is 'npm run server' running?"
      );
    });

    if (!pyRes.ok) {
      throw new Error(`Analysis server returned HTTP ${pyRes.status}`);
    }

    const pyResult = await pyRes.json();

    if (pyResult.error) {
      return NextResponse.json({ error: pyResult.error }, { status: 500 });
    }

    const rawTokens: {
      token: string;
      logprob: number;
      rank: number;
      entropy: number;
    }[] = pyResult.tokens;

    // Map Python server's classification format to frontend's ClassificationResult
    let classification: ClassificationResult | undefined;
    if (pyResult.classification) {
      const c = pyResult.classification;
      classification = {
        label_name: c.prediction,
        probabilities: c._4class ?? {},
        ai_score: c.ai_score,
      };
    }

    if ((!rawTokens || rawTokens.length === 0) && !classification) {
      return NextResponse.json(
        { error: "No analysis models available. Install llama3.2:1b or DeBERTa classifier." },
        { status: 500 }
      );
    }

    const tokens = processTokens(rawTokens || []);
    const gltr = computeGLTR(tokens);
    const entropy = computeEntropyStats(tokens);
    const vocabulary = computeVocabulary(trimmed);
    const sentences = computeSentences(trimmed);
    const sentenceLength = computeSentenceLengthStats(sentences);
    const burstiness = computeBurstiness(sentences);
    const slidingWindow = computeSlidingWindow(tokens);
    const { perplexity } = computeOverallPerplexity(tokens);
    const { featureScores, overallScore: heuristicScore } = computeFeatureScores(
      perplexity,
      gltr,
      entropy,
      burstiness,
      vocabulary
    );

    // Fused score from Python (ensemble: DeBERTa + PPL + length gating)
    const fused = pyResult.fused as AnalysisResult["fused"];

    // Use fused score as primary (it includes length gating and PPL override)
    // Fall back to DeBERTa raw score, then heuristic
    const overallScore = fused?.ai_score ?? classification?.ai_score ?? heuristicScore;

    const hasTokenData = tokens.length > 0;
    const sentenceScores = computeSentenceScores(sentences, tokens);
    const wordCount = trimmed.split(/\s+/).filter((w) => w.length > 0).length;

    // Only compute heuristic tags when we have real token data
    const aiSimilarityTags = hasTokenData
      ? computeAISimilarityTags(perplexity, gltr, entropy, burstiness, vocabulary, sentences)
      : [];
    const aiVocabMatches = detectAIVocab(trimmed);

    const result: AnalysisResult = {
      tokens,
      sentences,
      slidingWindow,
      gltr,
      entropy,
      vocabulary,
      burstiness,
      sentenceLength,
      overallPerplexity: perplexity,
      overallScore,
      featureScores: hasTokenData ? featureScores : [],
      sentenceScores,
      wordCount,
      scoringEligible: classification ? wordCount >= 50 : wordCount >= 300,
      classification,
      fused,
      hasTokenData,
      aiSimilarityTags,
      aiVocabMatches,
    };

    return NextResponse.json(result);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
