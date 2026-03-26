# Routing-Aware AI Text Detection Ensemble

**Date:** 2026-03-24
**Source:** User-provided deep research report
**Core thesis:** The 31-point gap from 55% to 86% adversarial accuracy is entirely a routing problem, not a detector-quality problem.

## Key Papers

| Paper | Venue | Contribution |
|-------|-------|-------------|
| MoSEs (Wu et al.) | EMNLP 2025 | Stylistics-Aware Router: +11.3% standard, +39.2% low-resource |
| DoGEN (Tripathi et al.) | 2025 | Domain Gating Ensemble: 95.8% AUROC, outperforms 32B single model |
| FairOPT (Jung et al.) | 2025 | Group-specific thresholds: -27.4% error disparity, -0.5% accuracy |
| Binoculars (Hans et al.) | ICML 2024 | Cross-perplexity ratio: >90% TPR at 0.01% FPR, RAID #1 zero-shot |
| PHD (Tulchinskii et al.) | NeurIPS 2023 | Intrinsic dimensionality: paraphrase-resistant, human ~9 vs AI ~7.5 |

## Architecture: Genre Router → Domain-Specific Detectors

```
Text → Genre Router (TF-IDF+LR or XLM-RoBERTa)
         ├── standard/professional → PPL-based (Binoculars preferred)
         ├── casual/essay/anti-detect → DeBERTa
         ├── creative → both detectors (soft blend)
         └── confidence < 0.7 → soft routing (weighted average)
```

## Three-Week Roadmap (55% → 82-88%)

### Week 1: Routing gap (55% → 75-80%)
- Drop LR v2 (48% FP rate actively hurts)
- Platt scaling on 500+ diverse labeled examples
- Genre router: TF-IDF+LR, ~500 examples per category
- Soft routing for ambiguous cases

### Week 2: Domain expansion (80% → 84-88%)
- Generate 1000+ AI samples per professional domain (GPT-4/Claude/Llama-3/Mistral)
- LoRA adapters (rank 4-8) on frozen DeBERTa base
- Confounding neuron suppression (zero-retraining, +2-7% OOD)

### Week 3: Orthogonal signals (88% → 90%+)
- Binoculars (Falcon-7B pair, 14GB VRAM, RAID #1)
- PHD intrinsic dimensionality (RoBERTa-base, 125M params, paraphrase-resistant)
- XGBoost stacking meta-learner on all signals

## What Commercial Systems Do

| System | Architecture |
|--------|-------------|
| GPTZero | 7 components: burstiness + sentence classifier + PPL + education module + internet search + adversarial shield + deep learning classifier |
| Pangram | Hard negative mining with synthetic mirrors, iterative false-positive mining, 0.004% FPR |
| Turnitin | Dual model: AIW-2 (primary) + AIR-1 (paraphrase-specific), AIR-1 activates only when AIW-2 detects ≥20% |
| Originality.ai | Custom ELECTRA from scratch on 160GB, Blue/Red team adversarial training loop |

## Key Insight for Our System

Our DeBERTa (98.5% on essays) and PPL signals are complementary but currently blended with fixed weights. Oracle analysis shows 86% ceiling with perfect routing. The gap is routing, not detector quality.

**Immediate actionable items (no GPU needed):**
1. Confounding neuron suppression on existing DeBERTa
2. Platt scaling calibration
3. Genre router (TF-IDF+LR)

**GPU-dependent items:**
1. Binoculars (Falcon-7B pair, 14GB VRAM)
2. LoRA domain adapters training
3. PHD (can run on CPU but slow)
