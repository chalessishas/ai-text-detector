/**
 * Calibration script for the analyze prompt.
 * Runs all 12 test articles through Claude 3x each and evaluates 12 pass criteria.
 *
 * Usage: ANTHROPIC_API_KEY=... npx tsx scripts/calibrate-analyze.ts
 */

import Anthropic from "@anthropic-ai/sdk";
import matter from "gray-matter";
import fs from "fs";
import path from "path";

import { ANALYZE_SYSTEM_PROMPT } from "../src/lib/prompts/writing/analyze";

// ── Types ──

interface Annotation {
  id: string;
  paragraph: number;
  startOffset: number;
  endOffset: number;
  trait: string;
  severity: string;
  message: string;
  rewrite?: string;
}

interface AnalyzeResponse {
  annotations: Annotation[];
  traitScores: Record<string, number>;
  summary: string;
  conventionsSuppressed: boolean;
}

interface ArticleMeta {
  quality: string;
  genre: string;
  notes: string;
}

interface ArticleInput {
  filename: string;
  meta: ArticleMeta;
  body: string;
  wordCount: number;
}

interface CriterionResult {
  pass: boolean;
  detail: string;
}

// ── Constants ──

const TEST_DIR = path.resolve(__dirname, "../docs/test-articles");
const OUTPUT_PATH = path.resolve(__dirname, "../docs/prompt-eval-results.md");
const MODEL = "claude-sonnet-4-6";
const RUNS_PER_ARTICLE = 3;
const FORBIDDEN_PHRASES = ["nice work", "good job", "keep it up", "well done", "not bad"];
const SEVERITY_ORDER = ["good", "question", "suggestion", "issue"] as const;

// ── Helpers ──

function loadArticles(): ArticleInput[] {
  const files = fs.readdirSync(TEST_DIR).filter((f) => f.endsWith(".md")).sort();
  return files.map((filename) => {
    const raw = fs.readFileSync(path.join(TEST_DIR, filename), "utf-8");
    const { data, content } = matter(raw);
    const body = content.trim();
    const wordCount = body.split(/\s+/).filter(Boolean).length;
    return {
      filename,
      meta: data as ArticleMeta,
      body,
      wordCount,
    };
  });
}

async function callAnalyze(
  client: Anthropic,
  text: string
): Promise<AnalyzeResponse> {
  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 4096,
    temperature: 0,
    system: ANALYZE_SYSTEM_PROMPT,
    messages: [{ role: "user", content: text }],
  });

  const raw =
    response.content[0].type === "text" ? response.content[0].text : "";

  // Strip markdown fencing if present
  const cleaned = raw.replace(/^```(?:json)?\s*/m, "").replace(/\s*```\s*$/m, "");
  return JSON.parse(cleaned) as AnalyzeResponse;
}

// ── Criterion Evaluators ──

function checkHonesty(article: ArticleInput, responses: AnalyzeResponse[]): CriterionResult {
  if (article.meta.quality !== "poor") {
    return { pass: true, detail: "N/A (not a poor article)" };
  }
  const counts = responses.map(
    (r) => r.annotations.filter((a) => a.severity === "issue").length
  );
  const allPass = counts.every((c) => c >= 3);
  return {
    pass: allPass,
    detail: `Issue counts: ${counts.join(", ")}. Need ≥3 each.`,
  };
}

function checkRewrite(responses: AnalyzeResponse[]): CriterionResult {
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    for (const a of responses[i].annotations) {
      if ((a.severity === "suggestion" || a.severity === "issue") && (!a.rewrite || a.rewrite.trim() === "")) {
        failures.push(`Run ${i + 1}: ${a.severity} annotation ${a.id} missing rewrite`);
      }
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "All suggestion/issue annotations have rewrite" : failures.slice(0, 3).join("; "),
  };
}

function checkNoEmptyPraise(responses: AnalyzeResponse[]): CriterionResult {
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    for (const a of responses[i].annotations) {
      const lower = a.message.toLowerCase();
      for (const phrase of FORBIDDEN_PHRASES) {
        if (lower.includes(phrase)) {
          // Check if followed by specific evidence (quoted text) in the same sentence
          const idx = lower.indexOf(phrase);
          const after = a.message.slice(idx + phrase.length, idx + phrase.length + 80);
          if (!after.includes('"') && !after.includes("'") && !after.includes("\u2018") && !after.includes("\u201C")) {
            failures.push(`Run ${i + 1}: "${phrase}" without evidence in: "${a.message.slice(0, 60)}..."`);
          }
        }
      }
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "No empty praise found" : failures.slice(0, 3).join("; "),
  };
}

function checkGoodSpecificity(responses: AnalyzeResponse[], articleBody: string): CriterionResult {
  const failures: string[] = [];
  const lowerBody = articleBody.toLowerCase();
  for (let i = 0; i < responses.length; i++) {
    for (const a of responses[i].annotations) {
      if (a.severity !== "good") continue;
      // Check if message contains a quoted fragment from the text
      const quotedParts = a.message.match(/["'\u2018\u201C]([^"'\u2019\u201D]+)["'\u2019\u201D]/g);
      if (!quotedParts) {
        failures.push(`Run ${i + 1}: good annotation lacks quoted text: "${a.message.slice(0, 60)}..."`);
        continue;
      }
      // Verify at least one quoted part exists in the article
      const anyMatch = quotedParts.some((q) => {
        const inner = q.slice(1, -1).toLowerCase().trim();
        return inner.length >= 2 && lowerBody.includes(inner);
      });
      if (!anyMatch) {
        failures.push(`Run ${i + 1}: good annotation quotes not found in text: "${a.message.slice(0, 60)}..."`);
      }
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "All good annotations cite specific text" : failures.slice(0, 3).join("; "),
  };
}

function checkQuestionQuality(responses: AnalyzeResponse[]): CriterionResult {
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    for (const a of responses[i].annotations) {
      if (a.severity !== "question") continue;
      if (!a.message.trim().endsWith("?")) {
        failures.push(`Run ${i + 1}: question doesn't end with "?": "${a.message.slice(0, 60)}..."`);
      }
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "All questions end with ?" : failures.slice(0, 3).join("; "),
  };
}

function checkLizLermanOrder(responses: AnalyzeResponse[]): CriterionResult {
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    const severities = responses[i].annotations.map((a) => a.severity);
    let lastIdx = -1;
    let ordered = true;
    for (const s of severities) {
      const idx = SEVERITY_ORDER.indexOf(s as typeof SEVERITY_ORDER[number]);
      if (idx < lastIdx) {
        ordered = false;
        break;
      }
      lastIdx = idx;
    }
    if (!ordered) {
      failures.push(`Run ${i + 1}: order violated: [${severities.join(", ")}]`);
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "Liz Lerman order correct" : failures.slice(0, 3).join("; "),
  };
}

function checkSuppressionFires(article: ArticleInput, responses: AnalyzeResponse[]): CriterionResult {
  // Only applies if Ideas or Organization has an "issue" annotation
  const hasIdeasOrgIssue = (r: AnalyzeResponse) =>
    r.annotations.some((a) => (a.trait === "ideas" || a.trait === "organization") && a.severity === "issue");

  const applicable = responses.filter(hasIdeasOrgIssue);
  if (applicable.length === 0) {
    return { pass: true, detail: "N/A (no ideas/org issues detected)" };
  }

  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    if (!hasIdeasOrgIssue(responses[i])) continue;
    if (!responses[i].conventionsSuppressed) {
      failures.push(`Run ${i + 1}: conventionsSuppressed should be true`);
    }
    const convAnnotations = responses[i].annotations.filter((a) => a.trait === "conventions");
    if (convAnnotations.length > 0) {
      failures.push(`Run ${i + 1}: ${convAnnotations.length} conventions annotations should be suppressed`);
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "Suppression fired correctly" : failures.slice(0, 3).join("; "),
  };
}

function checkSuppressionNoFalseFire(article: ArticleInput, responses: AnalyzeResponse[]): CriterionResult {
  // Only applies to article 11
  if (!article.filename.startsWith("11-")) {
    return { pass: true, detail: "N/A (not article 11)" };
  }
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    const convAnnotations = responses[i].annotations.filter((a) => a.trait === "conventions");
    if (convAnnotations.length === 0) {
      failures.push(`Run ${i + 1}: no conventions annotations — false suppression`);
    }
    if (responses[i].conventionsSuppressed) {
      failures.push(`Run ${i + 1}: conventionsSuppressed=true is a false fire`);
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "Conventions correctly NOT suppressed" : failures.slice(0, 3).join("; "),
  };
}

function checkScoreCalibration(article: ArticleInput, responses: AnalyzeResponse[]): CriterionResult {
  if (article.meta.quality === "poor") {
    const failures: string[] = [];
    for (let i = 0; i < responses.length; i++) {
      const s = responses[i].traitScores;
      if (s.ideas >= 40 && s.organization >= 40) {
        failures.push(`Run ${i + 1}: ideas=${s.ideas}, org=${s.organization} — need at least one < 40`);
      }
    }
    return {
      pass: failures.length === 0,
      detail: failures.length === 0 ? "Poor article scores calibrated" : failures.slice(0, 3).join("; "),
    };
  }
  if (article.meta.quality === "excellent") {
    const failures: string[] = [];
    for (let i = 0; i < responses.length; i++) {
      const s = responses[i].traitScores;
      if (s.ideas <= 75 || s.organization <= 75) {
        failures.push(`Run ${i + 1}: ideas=${s.ideas}, org=${s.organization} — need both > 75`);
      }
    }
    return {
      pass: failures.length === 0,
      detail: failures.length === 0 ? "Excellent article scores calibrated" : failures.slice(0, 3).join("; "),
    };
  }
  return { pass: true, detail: "N/A (not poor or excellent)" };
}

function checkMessageLength(responses: AnalyzeResponse[]): CriterionResult {
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    for (const a of responses[i].annotations) {
      const wordCount = a.message.split(/\s+/).filter(Boolean).length;
      if (wordCount < 15) {
        failures.push(`Run ${i + 1}: ${wordCount} words in: "${a.message.slice(0, 50)}..."`);
      }
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "All messages ≥ 15 words" : failures.slice(0, 3).join("; "),
  };
}

function checkGoodMinimum(article: ArticleInput, responses: AnalyzeResponse[]): CriterionResult {
  if (article.wordCount < 200) {
    return { pass: true, detail: `N/A (${article.wordCount} words < 200)` };
  }
  const failures: string[] = [];
  for (let i = 0; i < responses.length; i++) {
    const goodCount = responses[i].annotations.filter((a) => a.severity === "good").length;
    if (goodCount < 2) {
      failures.push(`Run ${i + 1}: only ${goodCount} good annotations, need ≥ 2`);
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "≥ 2 good annotations present" : failures.slice(0, 3).join("; "),
  };
}

function checkScoreStability(responses: AnalyzeResponse[]): CriterionResult {
  const traits = ["ideas", "organization", "voice", "wordChoice", "fluency", "conventions", "presentation"];
  const failures: string[] = [];
  for (const trait of traits) {
    const scores = responses.map((r) => r.traitScores[trait]);
    const maxDiff = Math.max(...scores) - Math.min(...scores);
    if (maxDiff > 5) {
      failures.push(`${trait}: spread=${maxDiff} (${scores.join(",")})`);
    }
  }
  return {
    pass: failures.length === 0,
    detail: failures.length === 0 ? "All traits within ±5 across runs" : failures.slice(0, 3).join("; "),
  };
}

// ── Main ──

const CRITERIA_NAMES = [
  "1. Honesty",
  "2. Rewrite",
  "3. No empty praise",
  "4. Good specificity",
  "5. Question quality",
  "6. Liz Lerman order",
  "7. Suppression fires",
  "8. Suppression no false-fire",
  "9. Score calibration",
  "10. Message length",
  "11. Good minimum",
  "12. Score stability",
] as const;

function evaluateArticle(
  article: ArticleInput,
  responses: AnalyzeResponse[]
): CriterionResult[] {
  return [
    checkHonesty(article, responses),
    checkRewrite(responses),
    checkNoEmptyPraise(responses),
    checkGoodSpecificity(responses, article.body),
    checkQuestionQuality(responses),
    checkLizLermanOrder(responses),
    checkSuppressionFires(article, responses),
    checkSuppressionNoFalseFire(article, responses),
    checkScoreCalibration(article, responses),
    checkMessageLength(responses),
    checkGoodMinimum(article, responses),
    checkScoreStability(responses),
  ];
}

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ERROR: Set ANTHROPIC_API_KEY environment variable.");
    process.exit(1);
  }

  const client = new Anthropic({ apiKey });
  const articles = loadArticles();
  console.log(`Loaded ${articles.length} test articles.\n`);

  // results[articleIdx][runIdx]
  const allResponses: AnalyzeResponse[][] = [];

  for (let ai = 0; ai < articles.length; ai++) {
    const article = articles[ai];
    console.log(`[${ai + 1}/${articles.length}] ${article.filename} (${article.meta.quality})`);
    const runs: AnalyzeResponse[] = [];

    for (let run = 0; run < RUNS_PER_ARTICLE; run++) {
      process.stdout.write(`  Run ${run + 1}/${RUNS_PER_ARTICLE}...`);
      try {
        const resp = await callAnalyze(client, article.body);
        runs.push(resp);
        console.log(` ${resp.annotations.length} annotations, ${Object.values(resp.traitScores).map((v) => Math.round(v)).join("/")}`);
      } catch (err) {
        console.log(` ERROR: ${err instanceof Error ? err.message : err}`);
        // Push a minimal failure response so we can still evaluate
        runs.push({
          annotations: [],
          traitScores: { ideas: 0, organization: 0, voice: 0, wordChoice: 0, fluency: 0, conventions: 0, presentation: 0 },
          summary: "API call failed",
          conventionsSuppressed: false,
        });
      }
    }
    allResponses.push(runs);
  }

  // ── Evaluate ──

  console.log("\n" + "=".repeat(80));
  console.log("CALIBRATION RESULTS");
  console.log("=".repeat(80) + "\n");

  const allResults: CriterionResult[][] = [];
  for (let ai = 0; ai < articles.length; ai++) {
    allResults.push(evaluateArticle(articles[ai], allResponses[ai]));
  }

  // Print per-article results
  for (let ai = 0; ai < articles.length; ai++) {
    const article = articles[ai];
    const results = allResults[ai];
    const passCount = results.filter((r) => r.pass).length;
    console.log(`\n${article.filename} (${article.meta.quality}) — ${passCount}/${results.length} pass`);
    for (let ci = 0; ci < results.length; ci++) {
      const icon = results[ci].pass ? "PASS" : "FAIL";
      console.log(`  [${icon}] ${CRITERIA_NAMES[ci]}: ${results[ci].detail}`);
    }
  }

  // Summary table
  console.log("\n\n" + "=".repeat(80));
  console.log("SUMMARY TABLE");
  console.log("=".repeat(80));

  const header = ["Criterion", ...articles.map((a) => a.filename.replace(".md", "").slice(0, 12))];
  const colWidth = 14;
  console.log("\n" + header.map((h) => h.padEnd(colWidth)).join(" | "));
  console.log("-".repeat((colWidth + 3) * header.length));

  for (let ci = 0; ci < CRITERIA_NAMES.length; ci++) {
    const row = [
      CRITERIA_NAMES[ci].padEnd(colWidth),
      ...articles.map((_, ai) => {
        const r = allResults[ai][ci];
        return (r.pass ? "PASS" : "FAIL").padEnd(colWidth);
      }),
    ];
    console.log(row.join(" | "));
  }

  // Overall stats
  const totalChecks = allResults.flat().length;
  const totalPass = allResults.flat().filter((r) => r.pass).length;
  console.log(`\nOverall: ${totalPass}/${totalChecks} checks passed (${((totalPass / totalChecks) * 100).toFixed(1)}%)`);

  // ── Save to markdown ──

  const lines: string[] = [
    "# Prompt Evaluation Results",
    "",
    `Generated: ${new Date().toISOString()}`,
    `Model: ${MODEL}`,
    `Runs per article: ${RUNS_PER_ARTICLE}`,
    "",
    "## Summary",
    "",
    `**${totalPass}/${totalChecks}** checks passed (**${((totalPass / totalChecks) * 100).toFixed(1)}%**)`,
    "",
    "## Results by Article",
    "",
  ];

  for (let ai = 0; ai < articles.length; ai++) {
    const article = articles[ai];
    const results = allResults[ai];
    const passCount = results.filter((r) => r.pass).length;
    lines.push(`### ${article.filename} (${article.meta.quality}) — ${passCount}/${results.length}`);
    lines.push("");
    lines.push("| # | Criterion | Result | Detail |");
    lines.push("|---|-----------|--------|--------|");
    for (let ci = 0; ci < results.length; ci++) {
      const icon = results[ci].pass ? "PASS" : "FAIL";
      lines.push(`| ${ci + 1} | ${CRITERIA_NAMES[ci]} | ${icon} | ${results[ci].detail.replace(/\|/g, "\\|")} |`);
    }
    lines.push("");
  }

  lines.push("## Trait Scores by Article");
  lines.push("");
  lines.push("| Article | Run | Ideas | Org | Voice | Word | Fluency | Conv | Pres |");
  lines.push("|---------|-----|-------|-----|-------|------|---------|------|------|");
  for (let ai = 0; ai < articles.length; ai++) {
    for (let ri = 0; ri < allResponses[ai].length; ri++) {
      const s = allResponses[ai][ri].traitScores;
      lines.push(
        `| ${articles[ai].filename.slice(0, 20)} | ${ri + 1} | ${s.ideas} | ${s.organization} | ${s.voice} | ${s.wordChoice} | ${s.fluency} | ${s.conventions} | ${s.presentation} |`
      );
    }
  }

  fs.writeFileSync(OUTPUT_PATH, lines.join("\n") + "\n");
  console.log(`\nResults saved to ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
