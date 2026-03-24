---
title: "UI Design: Why It Looks Like a Code Editor"
date: "2026-03-18"
summary: "Design notes on the VSCode-inspired layout — activity bar, tab navigation, warm ivory theme, and the thinking behind it."
tags: ["design", "notes"]
---

## The VSCode Metaphor

The app uses a layout inspired by code editors:
- A narrow **activity bar** on the left for switching between modes (Detect / Humanize / Writing Center)
- A **title bar** showing the current context
- **Tabbed panels** for different analysis views
- A **status bar** at the bottom showing word count and perplexity

Why? Because the target users are technical — students, writers, developers. They're already familiar with this layout pattern. It also naturally supports the "multiple analysis views" paradigm without feeling cluttered.

## The Warm Ivory Theme

We deliberately avoided the typical blue/purple AI aesthetic. The color palette:

| Variable | Value | Usage |
|----------|-------|-------|
| `--background` | `#f9f5ef` | Page background — warm ivory |
| `--foreground` | `#2d2b28` | Text — warm dark brown |
| `--card` | `#ffffff` | Card surfaces |
| `--card-border` | `#e8e2d9` | Subtle warm borders |
| `--accent` | `#c96442` | Primary accent — terracotta |
| `--accent-light` | `#f4e8e1` | Light accent for tags/badges |
| `--muted` | `#8b8580` | Secondary text |

The warm palette was chosen to feel approachable and non-threatening — important for a tool that's essentially judging your writing.

## Dynamic Loading

All chart components are loaded with `next/dynamic` and `ssr: false`:

```
const PerplexityCurve = dynamic(
  () => import("@/components/PerplexityCurve"),
  { ssr: false, loading: () => <PanelSkeleton /> }
);
```

This keeps the initial page load fast (no Recharts in the server bundle) and shows skeleton loaders while charts load. The skeleton uses the same warm palette with subtle pulse animations.

## Score Color System

The AI score uses a simple three-tier color system:
- **Red** (`>70%`): Likely AI — draws attention
- **Amber** (`30-70%`): Inconclusive — neutral caution
- **Green** (`<30%`): Likely human — reassuring

These thresholds match the scoring engine's confidence boundaries.
