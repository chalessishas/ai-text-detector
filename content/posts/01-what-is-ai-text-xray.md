---
title: "AI Text X-Ray: What It Does and Why We Built It"
date: "2026-03-01"
summary: "An overview of the project — a free, explainable AI text detector that shows you why text looks AI-generated, not just that it does."
tags: ["overview"]
---

## The Problem

Most AI detectors give you a single number — "87% AI-generated" — and nothing else. You have no idea why, no way to verify, and no way to learn from it.

We wanted something different: a tool that **shows its work**.

## What AI Text X-Ray Does

The app has three modes:

**AI Detector** — Paste any text and get a multi-dimensional breakdown using real statistical metrics:
- Perplexity (how predictable the text is to a language model)
- GLTR token ranking (how often words match the model's top predictions)
- Entropy (uncertainty at each token position)
- Burstiness (variation in sentence length and complexity)
- Vocabulary diversity (type-token ratio, hapax legomena)

Each metric has its own interactive chart. You can see sentence-by-sentence scoring, sliding window analysis, and the exact numbers that drive the overall score.

**AI Humanizer** — Takes AI-generated text and rewrites it using corpus-based matching, entity swapping, and structural transformation. Not prompt-based — the humanizer works at the sentence level using a real FAISS index of 2.95 million human-written sentences.

**Writing Center** — Generates optimized prompts for ChatGPT/Claude that encode the statistical fingerprint of natural human writing. The prompts specify sentence length ranges, transition word patterns, and vocabulary constraints that match our human corpus.

## Tech Stack

- **Frontend**: Next.js 16 (App Router) + React 19 + TypeScript + Tailwind CSS 4
- **Charts**: Recharts for all visualizations
- **Backend**: Python microservices — llama.cpp with Llama 3.2:1b for token probability analysis, FAISS for corpus search
- **Corpus**: ~1.1 GB of human text from C4, CNN/DailyMail, and Wikipedia, indexed into 2.95M sentence embeddings

## Why These Metrics

We chose metrics that have clear, published research behind them:

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| Perplexity | How surprised the model is | AI text has low perplexity (~3-8) vs human (~20-50) |
| GLTR | Token rank distribution | AI heavily uses top-10 predicted tokens |
| Entropy | Prediction uncertainty | AI text has uniformly low entropy |
| Burstiness | Sentence variation | Humans vary sentence length; AI doesn't |
| Vocabulary | Word diversity | AI uses a narrower, more repetitive vocabulary |

No single metric is reliable alone. The overall score is a weighted combination of all five.
