# AI Text X-Ray

AI 文本检测 + 人性化改写 + 写作教练三合一平台。四路投票检测（DeBERTa + qwen3:4b PPL + LR + 统计特征）+ 50M 真人语料库 FAISS 改写 + DeepSeek V3 写作辅导。

## Tech Stack

- **Frontend:** Next.js 16 + React 19 + TypeScript + Tailwind CSS 4 + Recharts 3 + Tiptap 3
- **Backend:** Python (DeBERTa + llama.cpp + FAISS + spaCy + SentenceTransformer)
- **AI API:** DeepSeek V3 (OpenAI-compatible, writing assist)
- **Data:** 50M sentence corpus (FAISS IVF+PQ, 161MB index) + 70,000 training samples

## Architecture

```
[Browser]
  |
[Next.js 16, port 3000]
  ├── /              Landing page (X-Ray scan animation + CTA → /app)
  ├── /app           AppShell (VSCode-style 3-panel: Detect | Humanize | WritingCenter)
  ├── /blog          Markdown blog (content/posts/*.md)
  │
  ├── /api/analyze        POST → Python port 5001 (llama3.2:1b perplexity + DeBERTa 4-class)
  ├── /api/humanize       POST → Python port 5002 (FAISS + spaCy, 7 rewrite methods)
  └── /api/writing-assist POST → DeepSeek API (7 actions: guide/analyze/expand/tip/lab/report)
```

### Component Tree

```
AppShell.tsx (430 lines, state: activePanel / text / result / activeTab)
  ├── Detect panel: 7 visualization tabs
  │   ├── FeatureOverview    (bar chart: 5 feature scores + AI similarity label)
  │   ├── SentenceAnalysis   (interactive cards: per-sentence scores)
  │   ├── GLTROverlay        (stacked bar: token rank distribution)
  │   ├── PerplexityCurve    (line + scatter: token-level perplexity)
  │   ├── EntropyChart       (histogram: entropy distribution)
  │   ├── BurstinessChart    (histogram: sentence length distribution)
  │   └── SlidingWindowChart (area chart: 10-token sliding window)
  ├── Humanize panel: HumanizeDashboard (lazy, per-sentence 7-method comparison)
  └── WritingCenter (lazy, 3-stage state machine: welcome → conversation → writing)
       ├── Editor.tsx        (Tiptap + trait-color inline annotations)
       ├── ChatPanel.tsx     (Socratic dialogue)
       ├── LabPanel.tsx      (cold→warm rewrite lab, temperature 0/0.7/1.5)
       ├── DailyTipCard.tsx  (35 static tips + AI fallback)
       ├── StepCard.tsx      (step-by-step article structure)
       └── storage.ts        (localStorage persistence)
```

## Directory Structure

```
ai-text-detector/
├── src/
│   ├── app/
│   │   ├── page.tsx, layout.tsx          # Landing + root layout (Geist font, JSON-LD SEO)
│   │   ├── app/page.tsx                  # Main app entry → <AppShell />
│   │   ├── blog/                         # Blog list + [slug] dynamic route
│   │   └── api/{analyze,humanize,writing-assist}/route.ts
│   ├── components/                       # All UI components (see tree above)
│   └── lib/
│       ├── analysis.ts                   # Types (TokenData, AnalysisResult, etc.) + compute functions
│       ├── posts.ts                      # Blog Markdown loader
│       ├── prompts.ts                    # DEAD FILE — replaced by prompts/writing/
│       ├── prompts/writing/              # 7 system prompts (analyze, guide-step, guide-dialogue, expand, daily-tip, lab-rewrite, report[stub])
│       └── writing/                      # types.ts, storage.ts, daily-tips.ts, lab-examples.ts
├── scripts/
│   ├── perplexity.py                     # Detection server (port 5001): llama3.2:1b + DeBERTa
│   ├── humanizer.py                      # Humanize server (port 5002): FAISS + spaCy + 7 methods
│   ├── generate_dataset.py               # Dataset gen: 23 models × 6 styles × 20 topics
│   ├── build_corpus_colab.py             # Colab: build 50M sentence FAISS index
│   └── train_classifier_colab.ipynb      # Colab: DeBERTa 4-class training
├── models/detector/                      # DeBERTa weights (738MB, downloaded from Colab GDrive)
├── corpus/                               # FAISS index (161MB) + sentences.jsonl (6.7GB)
├── content/posts/                        # 7 Markdown blog posts
└── .env.local                            # API keys (see Environment section)
```

## Key Conventions

- **Design system:** Warm ivory (#f9f5ef) background, burnt orange (#c96442) accent, subtle tan borders (#e8e2d9). VSCode-style activity bar (dark sidebar) + light content area.
- **Font:** Geist Sans 300 for headings, system sans-serif for body, Geist Mono for code. No Instrument Serif (banned).
- **No emoji in UI.**
- **DeBERTa ai_score formula:** `(P(ai) + P(ai_polished)) × 100` — used as primary score when classifier is available.
- **Heuristic fallback:** When DeBERTa unavailable, weighted 5-feature score used (requires 300+ words minimum).
- **Text limit:** 10,000 chars max (both analyze and humanize endpoints).
- **Lazy loading:** HumanizeDashboard and WritingCenter are `dynamic()` imports.
- **Writing Center 6+1 Traits:** Ideas, Organization, Voice, Word Choice, Sentence Fluency, Conventions, Presentation — each has a trait color for inline annotations.

## Known Pitfalls

1. **DeBERTa fp16 explodes** — Must call `model.float()` for fp32. M-series Mac + Colab both fail with fp16 DeBERTa-v3.
2. **tokenizer_config.json compat** — Colab transformers 5.x saves `extra_special_tokens` as list; local 4.x needs `additional_special_tokens: []`.
3. **Human training data quality** — Old human samples were incoherent (random sentence concat). Code fixed in generate_dataset.py but 70K dataset NOT regenerated yet.
4. **FAISS Python version** — Use Python 3.11. FAISS has issues with 3.12+.
5. **llama-cpp-python on Apple Silicon** — Metal acceleration compilation can be flaky.
6. **report action is 501 stub** — `/api/writing-assist` report action returns 501, not implemented.
7. ~~**lib/prompts.ts is dead code**~~ — RESOLVED: deleted in be706a4.
8. ~~**WritingCenter always shows welcome**~~ — RESOLVED: current code starts at "workbench" phase directly, no welcome screen.
9. **FAISS index + jsonl = 9GB+ total** — Not suitable for serverless, needs persistent disk.
10. **Tiptap v3 API instability** — Plugin API still evolving, custom extensions may need updates.
11. ~~**llama-cpp-python doesn't support qwen3.5**~~ — RESOLVED: dynamic Ollama model resolution falls back to llama3.2:1b automatically.
12. **tokenizer.vocab_size != model vocab** — Qwen3.5 tokenizer reports 248044 but model has 248320. Use `model(mx.array([[1]])).shape[-1]`.
13. **CoPA contrastive decoding doesn't fool DeBERTa** — 0/6 pass rate. Only changes surface stats, deep classifier catches it. StealthRL (GRPO LoRA) is the correct approach.
14. **PPL human ranges are model-specific** — Don't copy from literature. Calibrate on dataset.jsonl with the same model used for detection.
15. **DeBERTa cross-domain AUROC is 0.5-0.6** — Essentially random on unseen domains. Learns model fingerprints, not universal AI patterns. See docs/research/2026-03-25-14.md.
16. **LR model must match PPL model** — Switching PPL model (e.g. llama3.2:1b → qwen3:4b) requires retraining LR via `scripts/train_lr_local.py`.
17. ~~**Fusion logic is 140+ lines of if-else**~~ — RESOLVED: refactored to simple weighted average with XGBoost meta-learner path. Old if-else removed.
18. **Ollama blob hashes are fragile** — Never hardcode Ollama model paths. Use `_resolve_ollama_blob()` which calls `ollama show --modelfile` to dynamically resolve. Hash changes when Ollama updates.
19. **PPL model must be loaded for reliable detection** — Without PPL, fusion degrades to DeBERTa-only which has 0.5-0.6 AUROC on unseen domains. Server should check PPL availability on startup and warn.
20. **dataset_merged_noised.jsonl is corrupted** — 10% of words replaced with random junk strings. Do not use for training. Only dataset_v4.jsonl is clean and balanced.
21. **Log-rank signal weak with llama3.2:1b** — Mean log-rank shows no separation (AI ~1.04, human ~1.05). Needs ≥8B model for useful signal. Kept in API as informational only.

## Environment Setup

```bash
npm run dev        # Next.js frontend (port 3000)
npm run server     # Python detection service (port 5001, loads llama3.2:1b + DeBERTa)
npm run humanizer  # Python humanize service (port 5002, loads FAISS + spaCy)
```

**Requirements:**
- Node.js + npm
- Python 3.11 (FAISS compat) + pip: `llama-cpp-python==0.3.16`, `transformers`, `torch`, `faiss`, `spacy`, `sentence-transformers`
- Ollama with `llama3.2:1b` model pulled
- DeBERTa weights in `models/detector/` (download from Google Drive, trained on Colab)

**Environment variables** (in `.env.local`):
- `DEEPSEEK_API_KEY` — Writing Center AI (required for /api/writing-assist)
- `PERPLEXITY_SERVER_URL` — Python detection (default: 127.0.0.1:5001)
- `HUMANIZER_SERVER_URL` — Python humanize (default: 127.0.0.1:5002)
- `CLASSIFIER_PATH` — DeBERTa model dir path
- `MODEL_PATH` — llama3.2:1b model path
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `QWEN_API_KEY`, `GLM_API_KEY` — dataset generation only

## Detector Scoring

### Four-Signal Fusion (2026-03-25 overhaul)

Detection uses 4 independent signals combined via adaptive weighted voting:

| Signal | Source | What it measures | Reliability |
|--------|--------|-----------------|-------------|
| DeBERTa | `models/detector/` | Token-level style fingerprint | **Unreliable** — AUROC 0.5-0.6 cross-domain, both false positives and false negatives |
| PPL | qwen3:4b via llama-cpp | Token perplexity (how predictable text is) | Moderate — AI ppl ~5, human ~10-25 |
| LR | `models/perplexity_lr_v2.pkl` | 16-feature logistic regression (PPL + DivEye + SpecDetect) | 91.2% CV (80 samples) |
| Stat | Inline in perplexity.py | Sentence length CV, transition word density, punctuation diversity | Typo-resistant but coarse |

**Fusion logic** (perplexity.py lines 558-620):
- Consensus: when 2+ signals strongly agree (>80), simple average overrides dissent
- DeBERTa dampening: when PPL+LR or PPL+stat disagree with DeBERTa, DeBERTa weight drops to 10%
- Uncertainty zone: fused score within threshold ±8 → prediction = "uncertain"
- No LR fallback: DeBERTa 40% + PPL 35% + stat 25%

**DeBERTa 4-class** (unchanged):

| Class | Label | Test Accuracy |
|-------|-------|---------------|
| 0-3 | 4-class overall | **98.5%** (retrained 2026-03-23 on clean dataset, but poor real-world generalization) |

Binary mode: P(human) = P(class 0) + P(class 3), P(ai) = P(class 1) + P(class 2).

**Known limitations** (from 20+ adversarial tests):
- 0% false positive on human text (all human texts correctly identified)
- Typos/misspellings bypass all signals (DeBERTa + PPL + LR)
- Quillbot paraphrase bypasses PPL and LR
- Dialogue-format and short-sentence AI text bypasses all signals
- Wikipedia-style AI text bypasses DeBERTa and LR
- Code-mixed text confuses PPL

### Legacy 5-Feature Heuristic (frontend only, 300+ words)

| Feature | Weight | AI Range | Human Range |
|---------|--------|----------|-------------|
| Perplexity | 30% | 3-8 | 20-50 |
| GLTR Token Rank | 25% | >90% top-10 | <75% |
| Entropy | 25% | 1.0-2.0 | 2.5-3.5 |
| Burstiness | 15% | 0.10-0.20 | 0.35-0.65 |
| Vocabulary TTR | 5% | 0.65-0.80 | 0.75-0.90 |

## Humanizer Methods

| Method | Strategy |
|--------|----------|
| corpus | Semantic nearest human sentence from FAISS |
| structure | Keep content, change sentence structure using human template |
| transplant | Graft AI entities into human sentence template |
| inject | Insert AI clause objects into human sentence base |
| harvest | Extract best clauses from multiple human matches and combine |
| remix | Rebuild using only corpus vocabulary (fully de-AI) |
| anchor | Human sentence as anchor + append AI facts |

Deprecated methods in `scripts/_deprecated_methods.py`: phrase, collocation, noise, splice.

### CoPA (contrastive decoding) — FAILED
Scripts: `copa_mlx.py`, `copa_proof.py`, `copa_sweep.py`, `copa_utils.py`
Verified FAILED against DeBERTa 98.5% (0/6) and GPTZero (92-98% AI).
Only fools perplexity heuristics, not deep classifiers. May still be useful as part of a pipeline.

### StealthRL (GRPO LoRA) — IN PROGRESS
Script: `train_stealth_colab.ipynb`
Approach: RL fine-tune Qwen2.5-3B with DeBERTa as reward detector.
Reward: `1.0 * (1 - P_ai) + 0.3 * cosine_sim(original, paraphrase)`
Training: Colab T4, ~2-3 hours, LoRA adapter ~50MB → `models/stealth_lora/`
Paper: [StealthRL (arXiv 2602.08934)](https://arxiv.org/abs/2602.08934), reported 97-99% evasion rate
