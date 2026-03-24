---
title: "Why We Built AI Text X-Ray"
date: "2026-03-15"
summary: "Most AI detectors give you a score and nothing else. We wanted to show you exactly why text looks AI-generated — and help you write better."
tags: ["mission", "product"]
---

## The Problem with AI Detectors

Every AI detector on the market gives you the same experience: paste text, get a percentage, wonder if it's right. GPTZero says 87% AI. Originality.ai says 72% AI. Turnitin flags 3 sentences. None of them explain *why*.

We built AI Text X-Ray because we believe detection without explanation is useless. If you're a student trying to improve your writing, knowing "your essay scored 85% AI" tells you nothing about what to fix. If you're a teacher, it gives you a number but no evidence to discuss with the student.

## What We Do Differently

AI Text X-Ray breaks down the analysis into the actual signals that distinguish human from AI writing:

- **Perplexity** — How predictable is each word? AI text is extremely predictable. Human text surprises.
- **GLTR Token Ranking** — Are the model's top predictions always chosen? AI picks the "safe" word. Humans pick the interesting one.
- **Entropy** — How uncertain is the model at each position? Low entropy = AI. High entropy = human.
- **Burstiness** — Do sentence lengths vary? AI writes uniform 20-word sentences. Humans write 5-word punches followed by 40-word monsters.
- **Vocabulary Diversity** — How rich is the word choice? AI repeats. Humans explore.

Every metric is shown on an interactive chart. You can see exactly which sentences triggered the detector and why.

## Beyond Detection

Detection is only half the story. That's why we also built a Humanizer (using a 50-million-sentence human corpus for sentence-level replacement) and a Writing Center (an AI writing coach that teaches you to write with authentic voice).

The goal isn't to help people cheat. It's to help people understand what makes writing human — and get better at it.
