# Next Steps After XGBoost v3 Deployment (2026-04-03)

> Research Loop. Context: XGBoost v3 deployed (94.6% OOD accuracy, 112 samples), 122/124 tests pass, all services running.

## 1. XGBoost Fusion — Improving with More Data

### Current State
- 112 OOD samples, 94.6% CV accuracy
- DeBERTa still dominates (73% importance)
- Other signals contribute but weakly (top10 14%, LR 4%, PPL 4%)

### Recommended Next Steps

**A. Probability calibration before stacking** (high impact, low effort)
- DeBERTa outputs are overconfident (99-100% for most texts)
- Calibrate base classifier probabilities before feeding to meta-learner
- Methods: Platt scaling or isotonic regression on held-out set
- Source: [Stacking Principle in ML](https://medium.com/@ayasc/the-stacking-principle-in-machine-learning-base-learners-meta-learners-and-the-art-of-learning-524a085f3255)

**B. Cross-validation stacking (Hyb-Stack)**
- Instead of simple train/test split, use k-fold stacking
- Each base model generates OOF (out-of-fold) predictions
- This effectively 5x the meta-learner training data without collecting more
- Source: [Hybrid Stacked Ensemble](https://www.nature.com/articles/s41598-026-38172-9)

**C. Expand OOD dataset to 300+ samples**
- Current 112 is marginal for 7-feature GradientBoosting
- Focus areas: academic essays (FP-prone), code-mixed text, non-native English
- Consider using HC3/RAID benchmark datasets filtered for non-overlap with v4

## 2. DeBERTa Cross-Domain Improvement

### Self-Domain Adversarial Training (no extra data needed)
- Train DeBERTa with adversarial perturbations to its own domain
- Forces model to learn domain-invariant features
- DA-BAG method: self-domain adversarial training on single dataset extracts multi-domain features
- Source: [DA-BAG](https://link.springer.com/article/10.1007/s10844-024-00889-2)

### Adversarial Data Augmentation
- Use `dataset_adversarial_v4.jsonl` (69K samples, 14 attack types) for DeBERTa retraining
- This teaches DeBERTa to resist known bypass techniques
- Already planned in STATUS.md, notebook ready: `train_detector_v5_colab.ipynb`

## 3. Deployment Architecture

### Problem
- FAISS index: 2.8GB + sentences.jsonl: 6.7GB = 9.5GB
- DeBERTa: 738MB, llama3.2:1b: 1.3GB
- Total: ~12GB — too large for serverless (Vercel 5-min timeout, cold start issues)

### Recommended Architecture
```
[Vercel] Next.js frontend (static + API routes)
    ↓ proxy
[Fly.io / Railway] Docker container
    ├── Detection server (port 5001)
    ├── DeBERTa + llama3.2:1b
    └── XGBoost fusion

[Separate service or removed]
    └── Humanizer (FAISS 9.5GB — needs persistent disk)
```

**Option A: Fly.io** ($5-15/month)
- Persistent volumes for model files
- GPU machines available (A10G, $0.50/hr)
- Docker deploy, auto-scaling
- Limitation: FAISS 9.5GB needs large volume

**Option B: Railway** ($5-10/month)
- Simpler deploy (Dockerfile)
- No GPU
- 8GB RAM max on starter — tight for DeBERTa + llama

**Option C: Split architecture**
- Frontend: Vercel (free tier)
- Detection: Fly.io Docker (models loaded on startup)
- Humanizer: Defer to v2 (API-based rewriting instead of FAISS corpus)

### Key Decision
The FAISS humanizer is the deployment bottleneck (9.5GB). Consider replacing with API-based rewriting (DeepSeek/GPT to paraphrase) for production, keeping FAISS as local-only dev feature.

## 4. Priority Ranking

| # | Action | Impact | Effort | ROI |
|---|--------|--------|--------|-----|
| 1 | DeBERTa adversarial retraining (Colab) | High | Medium (2-3h GPU) | **High** |
| 2 | Calibrate base model probabilities | Medium | Low (30 min) | **High** |
| 3 | Expand OOD to 300+ samples | Medium | Low (1h) | **Medium** |
| 4 | Deploy to Fly.io (detection only) | High | Medium (2h) | **Medium** |
| 5 | Replace FAISS humanizer with API | Medium | High (4h+) | **Low** |

## Sources
- [Stacking Principle in ML](https://medium.com/@ayasc/the-stacking-principle-in-machine-learning-base-learners-meta-learners-and-the-art-of-learning-524a085f3255)
- [Hybrid Stacked Ensemble (Hyb-Stack)](https://www.nature.com/articles/s41598-026-38172-9)
- [XStacking: Explainable Stacking](https://www.sciencedirect.com/science/article/pii/S1566253525004312)
- [DA-BAG: Self-Domain Adversarial Training](https://link.springer.com/article/10.1007/s10844-024-00889-2)
- [Data Augmentation for Domain Generalization Survey](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- [Deploy AI to Production: Docker & Serverless](https://dev.to/paxrel/how-to-deploy-an-ai-agent-to-production-vps-docker-amp-serverless-2026-4p9i)
- [Vercel AI Review 2026](https://www.truefoundry.com/blog/vercel-ai-review-2026-we-tested-it-so-you-dont-have-to)
