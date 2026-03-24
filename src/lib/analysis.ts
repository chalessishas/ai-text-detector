export interface TokenData {
  token: string;
  logprob: number;
  perplexity: number;
  rank: number;
  entropy: number;
  position: number;
}

export interface SentenceData {
  index: number;
  text: string;
  wordCount: number;
  avgWordLength: number;
}

export interface SlidingWindowData {
  position: number;
  score: number;
  label: string;
}

export interface GLTRData {
  top10: number;
  top100: number;
  top1000: number;
  above1000: number;
}

export interface EntropyData {
  mean: number;
  std: number;
}

export interface VocabData {
  ttr: number;
  hapaxRatio: number;
  uniqueWords: number;
  totalWords: number;
}

export interface BurstinessData {
  lengthCV: number;
  complexityCV: number;
}

export interface SentenceLengthData {
  mean: number;
  std: number;
  histogram: { length: number; count: number }[];
}

export interface SentenceScoreData {
  index: number;
  text: string;
  wordCount: number;
  aiScore: number;
  perplexity: number;
  label: string;
}

export interface FeatureScore {
  name: string;
  score: number;
  weight: number;
  raw: number;
  humanRef: string;
  aiRef: string;
}

export interface ClassificationResult {
  label_name: string;
  probabilities: Record<string, number>;
  ai_score: number;
}

export interface AnalysisResult {
  tokens: TokenData[];
  sentences: SentenceData[];
  slidingWindow: SlidingWindowData[];
  gltr: GLTRData;
  entropy: EntropyData;
  vocabulary: VocabData;
  burstiness: BurstinessData;
  sentenceLength: SentenceLengthData;
  overallPerplexity: number;
  overallScore: number;
  featureScores: FeatureScore[];
  sentenceScores: SentenceScoreData[];
  wordCount: number;
  scoringEligible: boolean;
  classification?: ClassificationResult;
  aiSimilarityTags: AISimilarityTag[];
  aiVocabMatches: AIVocabMatch[];
}

// --- Processing ---

export function processTokens(
  raw: { token: string; logprob: number; rank: number; entropy: number }[]
): TokenData[] {
  return raw.map((t, i) => ({
    token: t.token,
    logprob: t.logprob,
    perplexity: Math.exp(-t.logprob),
    rank: t.rank,
    entropy: t.entropy,
    position: i,
  }));
}

// --- Feature computation ---

export function computeGLTR(tokens: TokenData[]): GLTRData {
  if (tokens.length === 0)
    return { top10: 0, top100: 0, top1000: 0, above1000: 0 };

  let t10 = 0,
    t100 = 0,
    t1000 = 0,
    above = 0;
  for (const t of tokens) {
    if (t.rank <= 10) t10++;
    else if (t.rank <= 100) t100++;
    else if (t.rank <= 1000) t1000++;
    else above++;
  }
  const n = tokens.length;
  return {
    top10: (t10 / n) * 100,
    top100: (t100 / n) * 100,
    top1000: (t1000 / n) * 100,
    above1000: (above / n) * 100,
  };
}

export function computeEntropyStats(tokens: TokenData[]): EntropyData {
  if (tokens.length === 0) return { mean: 0, std: 0 };
  const vals = tokens.map((t) => t.entropy);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance =
    vals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / vals.length;
  return { mean, std: Math.sqrt(variance) };
}

export function computeVocabulary(text: string): VocabData {
  const words = text
    .split(/\s+/)
    .map((w) => w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, ""))
    .filter((w) => w.length > 0);
  const freq: Record<string, number> = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  const uniqueWords = Object.keys(freq).length;
  const hapax = Object.values(freq).filter((c) => c === 1).length;
  return {
    ttr: words.length > 0 ? uniqueWords / words.length : 0,
    hapaxRatio: words.length > 0 ? hapax / words.length : 0,
    uniqueWords,
    totalWords: words.length,
  };
}

function avgWordLen(text: string): number {
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  if (words.length === 0) return 0;
  return (
    words.reduce((sum, w) => sum + w.replace(/[^a-z]/gi, "").length, 0) /
    words.length
  );
}

export function computeSentences(text: string): SentenceData[] {
  const sentenceRegex = /[^.!?。！？\n]+[.!?。！？]*/g;
  const matches = text.match(sentenceRegex);
  let segments = matches
    ? matches.map((s) => s.trim()).filter((s) => s.length > 0)
    : text
        .split(/\n+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

  if (segments.length === 0) segments = [text.trim()];

  return segments.map((s, i) => ({
    index: i,
    text: s,
    wordCount: s.split(/\s+/).filter((w) => w.length > 0).length,
    avgWordLength: avgWordLen(s),
  }));
}

export function computeSentenceLengthStats(
  sentences: SentenceData[]
): SentenceLengthData {
  if (sentences.length === 0) return { mean: 0, std: 0, histogram: [] };
  const lengths = sentences.map((s) => s.wordCount);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance =
    lengths.reduce((sum, l) => sum + (l - mean) ** 2, 0) / lengths.length;
  const freqMap: Record<number, number> = {};
  for (const l of lengths) freqMap[l] = (freqMap[l] || 0) + 1;
  return {
    mean,
    std: Math.sqrt(variance),
    histogram: Object.entries(freqMap)
      .map(([k, v]) => ({ length: Number(k), count: v }))
      .sort((a, b) => a.length - b.length),
  };
}

export function computeBurstiness(sentences: SentenceData[]): BurstinessData {
  if (sentences.length < 2) return { lengthCV: 0, complexityCV: 0 };
  const lengths = sentences.map((s) => s.wordCount);
  const meanL = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const stdL = Math.sqrt(
    lengths.reduce((s, l) => s + (l - meanL) ** 2, 0) / lengths.length
  );
  const cx = sentences.map((s) => s.avgWordLength);
  const meanC = cx.reduce((a, b) => a + b, 0) / cx.length;
  const stdC = Math.sqrt(
    cx.reduce((s, c) => s + (c - meanC) ** 2, 0) / cx.length
  );
  return {
    lengthCV: meanL > 0 ? stdL / meanL : 0,
    complexityCV: meanC > 0 ? stdC / meanC : 0,
  };
}

// --- Sliding window (perplexity-based, for visualization) ---

function perplexityToAiScore(ppl: number): number {
  const logPpl = Math.log(Math.max(1, ppl));
  const score = 120 - 30 * logPpl;
  return Math.min(100, Math.max(0, score));
}

export function computeSlidingWindow(
  tokens: TokenData[],
  windowSize = 20,
  step = 5
): SlidingWindowData[] {
  const results: SlidingWindowData[] = [];
  if (tokens.length === 0) return results;

  if (tokens.length < windowSize) {
    const avgLogprob =
      tokens.reduce((sum, t) => sum + t.logprob, 0) / tokens.length;
    results.push({
      position: 0,
      score: perplexityToAiScore(Math.exp(-avgLogprob)),
      label: tokens.map((t) => t.token).join(""),
    });
    return results;
  }

  for (let i = 0; i <= tokens.length - windowSize; i += step) {
    const win = tokens.slice(i, i + windowSize);
    const avgLogprob =
      win.reduce((sum, t) => sum + t.logprob, 0) / win.length;
    const label = win
      .slice(0, 5)
      .map((t) => t.token)
      .join("")
      .trim();
    results.push({
      position: i,
      score: perplexityToAiScore(Math.exp(-avgLogprob)),
      label: label + "...",
    });
  }
  return results;
}

// --- Overall perplexity ---

export function computeOverallPerplexity(
  tokens: TokenData[]
): { perplexity: number } {
  if (tokens.length === 0) return { perplexity: 0 };
  const avgLogprob =
    tokens.reduce((sum, t) => sum + t.logprob, 0) / tokens.length;
  return { perplexity: Math.exp(-avgLogprob) };
}

// --- Per-sentence scoring ---

export function computeSentenceScores(
  sentences: SentenceData[],
  tokens: TokenData[]
): SentenceScoreData[] {
  if (tokens.length === 0 || sentences.length === 0) return [];

  const totalWords = sentences.reduce((s, sent) => s + sent.wordCount, 0);
  if (totalWords === 0) return [];

  const result: SentenceScoreData[] = [];
  let tIdx = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sent = sentences[i];
    const isLast = i === sentences.length - 1;
    const count = isLast
      ? tokens.length - tIdx
      : Math.max(1, Math.round((sent.wordCount / totalWords) * tokens.length));
    const sentTokens = tokens.slice(tIdx, tIdx + count);
    tIdx += count;

    if (sentTokens.length === 0) {
      result.push({
        index: sent.index,
        text: sent.text,
        wordCount: sent.wordCount,
        aiScore: 50,
        perplexity: 0,
        label: "Mixed",
      });
      continue;
    }

    const avgLogprob =
      sentTokens.reduce((s, t) => s + t.logprob, 0) / sentTokens.length;
    const ppl = Math.exp(-avgLogprob);
    const aiScore = perplexityToAiScore(ppl);
    const label =
      aiScore > 70 ? "AI-like" : aiScore > 30 ? "Mixed" : "Human-like";

    result.push({
      index: sent.index,
      text: sent.text,
      wordCount: sent.wordCount,
      aiScore,
      perplexity: ppl,
      label,
    });
  }

  return result;
}

// --- Multi-feature scoring ---

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function computeFeatureScores(
  ppl: number,
  gltr: GLTRData,
  entropy: EntropyData,
  burstiness: BurstinessData,
  vocabulary: VocabData
): { featureScores: FeatureScore[]; overallScore: number } {
  // Calibration data (2026-03-22, 100 samples via llama3.2:1b):
  // PPL: human=23.1±8.8, ai=16.2±12.0 — heavy overlap, weak signal alone
  // Top10: human=72.6±4.5, ai=79.2±8.5 — moderate separation
  // Entropy: human=2.95±0.38, ai=2.41±0.66 — moderate separation
  // Note: llama3.2:1b is too small for high-quality perplexity — 7B+ would help
  const features: FeatureScore[] = [
    {
      name: "Perplexity",
      score: 1 - clamp((Math.log(Math.max(1, ppl)) - 1.0) / 2.7, 0, 1),
      weight: 0.3,
      raw: ppl,
      humanRef: "20-30",
      aiRef: "10-18",
    },
    {
      name: "Token Rank (GLTR)",
      score: clamp((gltr.top10 - 60) / 35, 0, 1),
      weight: 0.25,
      raw: gltr.top10,
      humanRef: "<75% top-10",
      aiRef: ">80% top-10",
    },
    {
      name: "Entropy",
      score: 1 - clamp((entropy.mean - 1.0) / 2.5, 0, 1),
      weight: 0.25,
      raw: entropy.mean,
      humanRef: "2.7-3.3",
      aiRef: "1.7-2.5",
    },
    {
      name: "Burstiness",
      score: 1 - clamp((burstiness.lengthCV - 0.1) / 0.5, 0, 1),
      weight: 0.15,
      raw: burstiness.lengthCV,
      humanRef: "0.35-0.65",
      aiRef: "0.10-0.20",
    },
    {
      name: "Vocabulary",
      score: 1 - clamp((vocabulary.ttr - 0.5) / 0.4, 0, 1),
      weight: 0.05,
      raw: vocabulary.ttr,
      humanRef: "0.75-0.90",
      aiRef: "0.65-0.80",
    },
  ];

  const overallScore =
    features.reduce((sum, f) => sum + f.score * f.weight, 0) * 100;
  return { featureScores: features, overallScore };
}

// --- AI similarity tags (GPTZero-style "Why is this text AI like?") ---

export interface AISimilarityTag {
  label: string;
  description: string;
}

export function computeAISimilarityTags(
  ppl: number,
  gltr: GLTRData,
  entropy: EntropyData,
  burstiness: BurstinessData,
  vocabulary: VocabData,
  sentences: SentenceData[]
): AISimilarityTag[] {
  const tags: AISimilarityTag[] = [];

  if (ppl < 12)
    tags.push({
      label: "Predictable Text",
      description:
        "Low perplexity indicates the text follows highly predictable patterns typical of language model output.",
    });

  if (gltr.top10 > 85)
    tags.push({
      label: "Formulaic Word Choice",
      description:
        "Most tokens are the model's top predictions, suggesting words were chosen for probability rather than nuance.",
    });

  if (entropy.mean < 2.0)
    tags.push({
      label: "Low Uncertainty",
      description:
        "Consistently low entropy means the model was rarely surprised — the text lacks the unpredictability of human writing.",
    });

  if (entropy.std < 0.8)
    tags.push({
      label: "Flat Entropy Profile",
      description:
        "Entropy barely varies across the text, producing a monotone information density unlike human writing's natural peaks and valleys.",
    });

  if (burstiness.lengthCV < 0.25)
    tags.push({
      label: "Uniform Rhythm",
      description:
        "Sentences are nearly the same length, creating a mechanical rhythm. Human writing naturally varies between short and long sentences.",
    });

  if (burstiness.complexityCV < 0.08)
    tags.push({
      label: "Consistent Complexity",
      description:
        "Word complexity is uniform across sentences. Human writers mix simple and complex vocabulary more freely.",
    });

  if (vocabulary.ttr < 0.72 && vocabulary.totalWords > 100)
    tags.push({
      label: "Repetitive Vocabulary",
      description:
        "Low type-token ratio indicates the same words are reused frequently, a common pattern in AI-generated text.",
    });

  if (sentences.length > 3) {
    const avgWords = sentences.reduce((s, sent) => s + sent.wordCount, 0) / sentences.length;
    if (avgWords > 20 && avgWords < 28 && burstiness.lengthCV < 0.3)
      tags.push({
        label: "Template-like Structure",
        description:
          "Sentences cluster around a narrow word count range with similar structure, suggesting template-driven generation.",
      });
  }

  return tags;
}

// --- AI vocab detection ---

const AI_VOCAB: string[] = [
  "delve", "delves", "delving",
  "furthermore", "moreover", "additionally",
  "utilize", "utilizes", "utilizing", "utilization",
  "multifaceted", "holistic", "paradigm", "synergy",
  "leverage", "leverages", "leveraging",
  "comprehensive", "robust", "streamline",
  "facilitate", "facilitates", "facilitating",
  "paramount", "pivotal", "crucial",
  "encompass", "encompasses", "encompassing",
  "intricate", "intricacies",
  "underscore", "underscores", "underscoring",
  "landscape", "tapestry", "realm",
  "noteworthy", "notably",
  "it is important to note",
  "it is worth noting",
  "in conclusion",
  "in today's rapidly",
  "plays a crucial role",
  "shed light on",
  "paves the way",
];

export interface AIVocabMatch {
  word: string;
  startIndex: number;
  endIndex: number;
}

export function detectAIVocab(text: string): AIVocabMatch[] {
  const lower = text.toLowerCase();
  const matches: AIVocabMatch[] = [];

  for (const term of AI_VOCAB) {
    let searchFrom = 0;
    while (searchFrom < lower.length) {
      const idx = lower.indexOf(term, searchFrom);
      if (idx === -1) break;
      const before = idx > 0 ? lower[idx - 1] : " ";
      const after = idx + term.length < lower.length ? lower[idx + term.length] : " ";
      if (/\W/.test(before) && /\W/.test(after)) {
        matches.push({
          word: text.slice(idx, idx + term.length),
          startIndex: idx,
          endIndex: idx + term.length,
        });
      }
      searchFrom = idx + term.length;
    }
  }

  matches.sort((a, b) => a.startIndex - b.startIndex);
  return matches;
}

// --- Sentence explain (template-based) ---

export function explainSentenceScore(
  sent: SentenceScoreData
): string[] {
  const reasons: string[] = [];

  if (sent.aiScore > 70) {
    if (sent.perplexity < 8)
      reasons.push(
        "Very low perplexity — the language model found this sentence highly predictable."
      );
    if (sent.perplexity < 15 && sent.perplexity >= 8)
      reasons.push(
        "Low perplexity suggests this sentence follows common AI generation patterns."
      );
    if (sent.wordCount > 25)
      reasons.push(
        "Long, complex sentence structure is more typical of AI-generated prose."
      );
    if (reasons.length === 0)
      reasons.push(
        "Multiple statistical features align with AI-generated text patterns."
      );
  } else if (sent.aiScore <= 30) {
    if (sent.perplexity > 30)
      reasons.push(
        "High perplexity — the language model found this sentence surprising and unpredictable."
      );
    if (sent.wordCount < 12)
      reasons.push(
        "Short, punchy sentence — AI tends to generate longer, more uniform sentences."
      );
    if (reasons.length === 0)
      reasons.push(
        "This sentence's statistical profile is consistent with human writing patterns."
      );
  } else {
    reasons.push(
      "This sentence shows a mix of human and AI-like characteristics."
    );
  }

  return reasons;
}
