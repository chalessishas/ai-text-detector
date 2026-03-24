---
title: "Building a 2.95 Million Sentence Corpus"
date: "2026-03-10"
summary: "How we collected, cleaned, and indexed 1.1 GB of human text from C4, CNN/DailyMail, and Wikipedia into a FAISS vector database."
tags: ["technical", "humanizer"]
---

## Why a Corpus

The humanizer doesn't use an LLM to rewrite text. Instead, it finds **real human sentences** that are semantically similar to each AI sentence and swaps them in. This means we need a massive, high-quality collection of human-written sentences.

## Data Sources

We combined three datasets:

| Source | Type | Why |
|--------|------|-----|
| C4 (Colossal Clean Crawled Corpus) | Web text | Diverse topics and writing styles |
| CNN/DailyMail | News articles | Clean, well-edited prose |
| Wikipedia | Encyclopedia | Factual, structured writing |

Total raw data: ~1.1 GB of text.

## Processing Pipeline

1. **Sentence splitting** — Break all documents into individual sentences
2. **Filtering** — Remove sentences that are too short (<5 words), too long (>50 words), or contain URLs/HTML artifacts
3. **Deduplication** — Remove exact and near-duplicate sentences
4. **Embedding** — Encode each sentence into a dense vector using a sentence transformer model
5. **Indexing** — Build a FAISS index for fast nearest-neighbor search

The final index contains **2.95 million sentences** in a 161 MB FAISS file, with metadata stored in a JSONL sidecar.

## How the Humanizer Uses It

When you submit text to humanize:

1. Each sentence is embedded using the same sentence transformer
2. FAISS finds the top-k most similar human sentences from the corpus
3. The humanizer applies different strategies:
   - **Corpus match**: Direct replacement with the most similar human sentence
   - **Entity swap**: Take a corpus sentence and replace named entities to match your topic
   - **Structure match**: Use the syntactic structure of a human sentence but fill it with your content

The result reads naturally because every sentence is grounded in actual human writing patterns.

## Notes

- The corpus is intentionally English-only for now
- We exclude any text that appears in common AI training data test sets to avoid contamination
- The FAISS index uses flat L2 search — brute force but exact, no approximation errors
