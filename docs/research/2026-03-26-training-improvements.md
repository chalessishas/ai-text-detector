# Research Report: Next-Step Training Improvements

**Date**: 2026-03-26
**Context**: After LR v3 (86.1%) + DeBERTa v4 (97.6%) training session

---

## 1. StyloMetrix: 196-Feature Stylometric Analysis (spaCy-based)

**Source**: [Stylometry recognizes human and LLM-generated texts in short samples](https://arxiv.org/html/2507.00838v2)

**Finding**: Boosted tree classifier with StyloMetrix's 196 spaCy-based features can distinguish human/AI text even in extremely short samples (10 sentences). Our LR v3 uses only 8 linguistic features.

**Actionable**: Install `stylo_metrix` package and extract full 196-feature vector. Run feature importance analysis to find the top ~20 that complement our existing 16 perplexity features. Expected: significant accuracy boost from function word bigrams, readability scores, and lexical diversity metrics we're currently missing.

**Priority**: HIGH — low implementation cost, directly extends our spaCy pipeline.

---

## 2. XGBoost Outperforms LR in Cross-Domain Detection

**Source**: [AI Generated Text Detection (arXiv 2601.03812)](https://arxiv.org/html/2601.03812)

**Finding**: XGBoost with extended feature set (sentence-level perplexity CV, connector density, AI-phrase density) achieves strong cross-domain results, outperforming LR. TF-IDF LR baseline is 82.87% while ensemble/boosted methods reach 88-92%.

**Actionable**: Replace `LogisticRegression` with `XGBClassifier` in train_lr_v3.py. XGBoost handles feature interactions (e.g., low PPL + high contraction_rate) that LR misses. sklearn-compatible API means minimal code changes.

**Priority**: HIGH — drop-in replacement, ~5 min to implement, likely +3-5% accuracy.

---

## 3. Domain-Adversarial Neural Networks (DANN) for DeBERTa

**Source**: [GenAI Content Detection Task 3: Cross-Domain Challenge](https://arxiv.org/html/2501.08913v1)

**Finding**: XLM-RoBERTa + DANN minimizes domain-specific biases and improves generalization. The DANN approach adds a domain classifier with gradient reversal layer, forcing the encoder to learn domain-invariant features.

**Actionable**: For next DeBERTa retrain, add a domain adversarial head during training. Requires labeling training data with domain tags (essay, news, forum, technical). Implementation: ~100 lines of custom Trainer code.

**Priority**: MEDIUM — significant effort but addresses root cause of DeBERTa cross-domain failure (AUROC 0.5-0.6).

---

## 4. Missing High-Value Features

**Source**: [Feature-Based Detection of AI-Generated Text](https://www.researchgate.net/publication/398588043)

**Features we should add**:
- **Function word bigrams** — top discriminator in multiple studies
- **Readability scores** (Flesch-Kincaid, Coleman-Liau) — AI text is typically more "readable"
- **AI-phrase density** — count of known AI-typical phrases ("it is important to note", "delve", "tapestry")
- **Connector density** — transition words per sentence (AI overuses "Furthermore", "Moreover")

**Priority**: HIGH — these are cheap to compute (no model needed) and proven effective.

---

## 5. Diversity in Training Data Boosts Detection

**Source**: [Diversity Boosts AI-Generated Text Detection (arXiv 2509.18880)](https://arxiv.org/pdf/2509.18880)

**Finding**: Training on text from diverse generators significantly improves detection generalization. Models trained on single-generator data fail on unseen generators.

**Relevance**: Our dataset_v4 already has 23 different AI models — this is a strength. But human data diversity was weak until we added HC3 + IMDB in this session.

**Actionable**: Continue expanding human data diversity (academic papers, forum posts, creative writing, technical docs). Each new domain reduces false positive rate.

---

## Recommended Next Actions (Priority Order)

1. **Switch LR to XGBoost** — drop-in, likely +3-5% (30 min)
2. **Add AI-phrase density + readability + connector density features** — cheap, proven (1 hr)
3. **Explore StyloMetrix 196 features** — find top 20 discriminators (2 hr)
4. **DANN for DeBERTa** — addresses root cause of cross-domain issue (1 day)
5. **More diverse human data** — reddit posts, academic abstracts, blog posts (ongoing)
