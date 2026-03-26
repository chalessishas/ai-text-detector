# Research Report: False Positive Reduction & Adversarial Robustness

**Date**: 2026-03-26 05:00
**Context**: After DeBERTa v4 shows 55-75% false positive on human casual/essay text

---

## 1. Multi-Detector Aggregation Eliminates False Positives

**Source**: [Aggregated AI detector outcomes to eliminate false positives (APS)](https://journals.physiology.org/doi/full/10.1152/advan.00235.2024)

**Finding**: Using multiple independent detectors and only flagging text when 2+ agree dramatically reduces false positive rate while maintaining true positive rate.

**Relevance**: Our 4-signal fusion (DeBERTa + PPL + LR + Stats) already implements this principle. The fusion dampening we added today (DeBERTa weight → 10% when uncertain) is the right direction. The literature validates that consensus-based detection is the most reliable approach.

**Actionable**: Consider adding a 5th signal (e.g., AI-phrase density or readability score) that's completely independent of the perplexity pipeline. More independent signals = more robust consensus.

---

## 2. Hybrid Ensemble: 3 Paradigms Beat Any Single Approach

**Source**: [Theoretically Grounded Hybrid Ensemble (arXiv 2511.22153)](https://arxiv.org/html/2511.22153v1)

**Finding**: Best results come from fusing three DIFFERENT detection paradigms:
1. Transformer classifier (deep semantic features) — our DeBERTa
2. Probabilistic detector (perturbation-induced likelihood) — our PPL
3. Statistical linguistic analyzer (stylometric patterns) — our LR + Stats

**Relevance**: Our pipeline already implements all 3 paradigms! This validates our architecture. The key insight: each paradigm catches what the others miss.

**Actionable**: Our fusion weights (DeBERTa 30%, PPL 25%, LR 25%, Stats 20%) could be optimized via learned weights on a held-out set instead of hand-tuned.

---

## 3. RADAR: Adversarial Learning for Robust Detection

**Source**: [RADAR: Robust AI-Text Detection via Adversarial Learning](https://arxiv.org/abs/2307.03838)

**Finding**: RADAR trains a detector and paraphraser adversarially — the paraphraser tries to evade detection while the detector learns to catch evasions. This produces a detector that's inherently robust to paraphrase attacks.

**Relevance**: We have 69K adversarial samples from 14 attack types. Currently these are only used for evaluation. Training the LR on adversarial examples (as additional "AI" samples) would make it more robust.

**Actionable**: Add adversarial samples to LR v3 training data. Label them as AI (they are). This teaches the model to detect disguised AI text. Expected: better robustness to typos, casual injection, dialogue wrapping.

**Priority**: HIGH — we already have the data, just need to include it in training.

---

## 4. PIFE: Perturbation-Invariant Feature Engineering

**Source**: [Detecting AI-Generated Text by Quantifying Adversarial Perturbations (arXiv 2510.02319)](https://arxiv.org/html/2510.02319v1)

**Finding**: PIFE explicitly models adversarial artifacts and maintains 82.6% TPR under strict conditions. Standard adversarial training fails against semantic attacks, but feature-level invariance works.

**Relevance**: Our DivEye features (surprisal statistics) are already somewhat perturbation-invariant — they measure distribution shape, not specific words. The SpecDetect DFT energy is also invariant to surface changes.

**Actionable**: Add perturbation-invariance test to our evaluation: take the same text, apply all 14 attacks, check if the LR/PPL scores remain consistent. Features with high variance under perturbation should be down-weighted or removed.

---

## 5. False Positive Rates in Production (2026 Benchmark)

**Source**: [False Positives in AI Detection: Complete Guide 2026](https://proofademic.ai/blog/false-positives-ai-detection-guide/)

**Finding**: Commercial detectors in 2026 show FPR of 43-83% on authentic student writing. This is the #1 reason universities are dropping AI detectors.

**Relevance**: Our 0% FPR claim (from STATUS.md) is based on limited test cases. The real FPR on diverse human text (casual, non-native English, creative writing) is likely higher. The DeBERTa sanity tests showing 55-75% AI on human text confirm this.

**Actionable**: Build a proper FPR test suite with 100+ diverse human texts (native/non-native, formal/casual, academic/creative). Report FPR honestly. A detector with 5% FPR is more trustworthy than one claiming 0%.

---

## 6. Stacking Meta-Learner for Fusion Weight Optimization

**Source**: [Optimizing Ensemble Weights (arXiv 1908.05287)](https://arxiv.org/abs/1908.05287), [Ensemble Framework for Text Classification (MDPI)](https://www.mdpi.com/2078-2489/16/2/85)

**Finding**: Instead of hand-tuning fusion weights, train a meta-learner (LR or isotonic regression) on a held-out set. Input = 4 signal scores (DeBERTa, PPL, LR, Stats), output = final AI probability. Benefits:
- Automatically discovers optimal weight per signal
- Regularization can eliminate redundant signals
- O(M) complexity, trains in seconds
- Adapts to any future signal additions without re-tuning rules

**Implementation**:
```python
# Collect 4 signal scores on held-out set
X_meta = np.column_stack([deb_scores, ppl_scores, lr_scores, stat_scores])
y_meta = true_labels
meta_lr = LogisticRegression().fit(X_meta, y_meta)  # replaces 140 lines of if-else
```

**Priority**: HIGH — replaces the fragile 140-line fusion logic with a 5-line learned model.

---

## Recommended Priority Actions

1. **Include adversarial samples in LR training** — 69K samples already generated, free improvement (1 hr)
2. **Build FPR test suite** — 100+ diverse human texts, honest FPR measurement (2 hr)
3. **Stacking meta-learner for fusion** — replace 140 lines of if-else with learned weights (1 hr)
4. **Add perturbation-invariance test** — measure feature stability under 14 attacks (1 hr)
5. **5th independent signal** — AI-phrase density or readability, completely separate from PPL (30 min)
