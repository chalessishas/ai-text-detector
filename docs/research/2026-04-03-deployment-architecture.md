# Deployment Architecture Research (2026-04-03)

> Research Loop. Focus: how to deploy ai-text-detector for real users.

## Current Bottlenecks

| Component | Size | Problem |
|-----------|------|---------|
| DeBERTa v4 | 738MB | Loads in ~5s, needs 2GB RAM |
| llama3.2:1b | 1.3GB | Needs Metal/CUDA, 3GB RAM |
| FAISS index | 2.8GB | Read-only, needs mmap |
| sentences.jsonl | 6.7GB | Random access via offset index |
| **Total** | **~12GB** | **Too large for serverless** |

## Recommended: Split Architecture

```
┌─────────────────────────────────┐
│  Vercel (free tier)             │
│  Next.js frontend + API proxy  │
│  Writing Center (DeepSeek API)  │
└──────────┬──────────────────────┘
           │ POST /api/analyze → proxy
           ▼
┌─────────────────────────────────┐
│  Fly.io ($5-15/month)           │
│  Docker: Python detection       │
│  - DeBERTa (738MB)              │
│  - llama3.2:1b (1.3GB)          │
│  - XGBoost fusion               │
│  - LR model                     │
│  Persistent volume: models/     │
└─────────────────────────────────┘

Humanizer: DEFER to v2 (API-based paraphrasing)
  - FAISS 9.5GB too large for any affordable hosting
  - Replace with DeepSeek/GPT API call for production
  - Keep FAISS as local-dev-only feature
```

## Fly.io Specifics

- **Machine**: `shared-cpu-2x` (2 vCPU, 4GB RAM) — $10/month
- **Volume**: 5GB persistent for model files — $0.50/month
- **Deploy**: `fly deploy` with Dockerfile
- **Model loading**: Store weights in Tigris object storage, download on first boot
- **Scaling**: min 1, max 1 (single instance sufficient for MVP)
- Source: [Deploying ML on Fly.io](https://community.fly.io/t/deploying-ml-models-on-fly-in-a-couple-of-steps/10387)

## Model Distillation (Future Optimization)

For reducing DeBERTa 738MB:
- **DistilBERT**: 40% smaller, 97% accuracy retained, 60% faster
- **TinyBERT**: 7.5x smaller (4 layers vs 12), 96.8% on GLUE
- **Approach**: Distill DeBERTa v4 → 4-layer student model on dataset_v4.jsonl
- **Expected**: ~100MB model, <1s inference, deployable on 1GB RAM

## Action Items

1. **Create Dockerfile** for detection server (perplexity.py)
2. **Create fly.toml** with volume mount for models/
3. **Test locally** with `docker build && docker run`
4. **Deploy to Fly.io** with `fly deploy`
5. **Update Next.js** API routes to proxy to Fly.io URL in production

## Sources
- [Deploying ML on Fly.io](https://community.fly.io/t/deploying-ml-models-on-fly-in-a-couple-of-steps/10387)
- [Fly.io Persistent Volumes](https://community.fly.io/t/does-fly-ignore-docker-volumes/15434)
- [Tigris + Fly.io for Model Weights](https://www.tigrisdata.com/docs/model-storage/fly-io/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [TinyBERT Paper](https://openreview.net/forum?id=rJx0Q6EFPB)
- [Deploy Transformer as REST API](https://zghrib.medium.com/deploy-a-transformer-as-a-restapi-8c37266a229d)
