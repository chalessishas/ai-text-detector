---
title: "Writing Center: Generating Detection-Resistant Prompts"
date: "2026-03-15"
summary: "How we reverse-engineered human writing patterns from our corpus to create prompts that produce more natural AI text."
tags: ["feature", "writing-center"]
---

## The Idea

If you're going to use AI to write, the output should at least *read* like a human wrote it. Most AI text fails detection because it has specific statistical fingerprints — uniform sentence length, low entropy, predictable word choices.

The Writing Center generates prompts that explicitly counteract these patterns.

## What the Prompts Encode

We analyzed our 2.95M sentence corpus and extracted the statistical fingerprint of natural human writing:

- **Sentence length**: Average 23 words, range 18-28
- **Comma density**: 1-2 commas per sentence
- **Transition words**: Heavy use of "However,", "Although", "Despite", "While"
- **Voice mix**: ~25% passive voice ("was established", "were discovered")
- **Sentence starters**: Varied — "It was", "In the", "According to", "This was"
- **Word complexity**: ~4.5 characters per word (not too simple, not too academic)

## What the Prompts Avoid

The prompts explicitly tell the AI to avoid phrases that are statistically overrepresented in AI output:

- "delve into"
- "it's important to note"
- "furthermore"
- "leveraging"
- "holistic"
- "paradigm"
- "multifaceted"

These phrases are rare in our human corpus but extremely common in ChatGPT/Claude output, making them strong detection signals.

## Six Prompt Styles

| Style | Use Case |
|-------|----------|
| Essay | Standard academic essay |
| Academic | Research paper tone |
| Casual | Blog post, informal |
| Persona | Writing as a specific character |
| Anti-detect | Maximum detection evasion |
| Rewrite | Transform existing AI text |

Each style applies the same corpus-derived constraints but adjusts tone and structure.

## The Workflow

1. Enter a topic in the Writing Center
2. Select a style
3. Copy the generated prompt to ChatGPT or Claude
4. Bring the output back to the Humanizer for further refinement
5. Verify with the AI Detector

The goal isn't to cheat — it's to produce AI-assisted text that reads naturally and doesn't trip false positives on AI detectors.
