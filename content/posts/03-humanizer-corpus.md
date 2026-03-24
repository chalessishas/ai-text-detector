---
title: "Building a 50-Million Sentence Human Corpus"
date: "2026-03-20"
summary: "How we built the world's largest sentence-level human writing corpus for AI text humanization — and why corpus size matters."
tags: ["technical", "humanizer"]
---

## Why Corpus Size Matters

Our Humanizer works by finding semantically similar human-written sentences from a massive corpus and using them to replace AI-sounding text. The bigger and more diverse the corpus, the better the matches.

We started with 2.95 million sentences. Detection bypass rate: ~40%. We scaled to 50 million sentences. Bypass rate: the humanized text now scores in the human range (perplexity 30+) on pure corpus replacements.

## The Build

50 million sentences from 5 pre-2019 sources (no AI contamination):

| Source | Sentences | Type |
|--------|-----------|------|
| C4 | 28M | Diverse web text |
| Wikipedia | 2.8M | Encyclopedia |
| CC-News | 5.3M | News articles |
| CNN/DailyMail | 5.7M | News |
| Gutenberg | 7.5M | Classic literature |

Built on Google Colab A100 GPU in ~5 hours. FAISS IVF+PQ index for fast semantic search. The humanizer finds the closest human sentence and offers 11 different replacement strategies.

## The Trade-off

More corpus = better semantic matches = higher bypass rate. But also: more corpus = larger index (2.7 GB) = more memory. Our SentenceStore uses byte-offset indexing to keep memory under 400MB even at 50M scale.
