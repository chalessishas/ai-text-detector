# Research Note: Benchmark Datasets for Detector Evaluation

**Date**: 2026-03-26 06:00

## Available Datasets for FPR/TPR Testing

| Dataset | Size | Domains | Source |
|---------|------|---------|--------|
| **DetectRL** (NeurIPS 2024) | Multi-domain | Academic, news, creative, social media + attack variants | [OpenReview](https://openreview.net/forum?id=ZGMkOikEyv) |
| **ai-text-detection-pile** | Large | Multiple generators + human | [HuggingFace](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) |
| **AH&AITD** | 11,580 | Multi-domain human + AI | [Figshare](https://figshare.com/articles/dataset/AH_AITD_Arslan_s_Human_and_AI_Text_Database/29144348) |
| **NYT Comprehensive** | 58,000+ | NYT articles + 6 LLM variants | [arXiv 2510.22874](https://arxiv.org/html/2510.22874v1) |
| **NLPCC 2025 DetectRL-ZH** | Chinese | Academic, news, creative, social | [Springer](https://link.springer.com/chapter/10.1007/978-981-95-3352-7_21) |

## Key Insight

[arXiv 2603.23146](https://arxiv.org/html/2603.23146) ("Why AI-Generated Text Detection Fails") shows that benchmark accuracy does NOT predict real-world performance. Models scoring 95%+ on in-distribution benchmarks drop to 50-60% on out-of-distribution text — exactly our DeBERTa problem.

## Actionable

Download `artem9k/ai-text-detection-pile` from HuggingFace as our standard evaluation benchmark. It's directly downloadable and covers multiple generators. Use it to measure honest FPR/TPR before claiming any accuracy numbers.
