# Human Training Data Sources (2026-04-03)

> Research Loop. Problem: dataset_v4 human samples are mostly news/wiki concatenations, lacking diversity.

## Open-Source Datasets for Human Text

### 1. AI Text Detection Pile (HuggingFace)
- **URL**: https://huggingface.co/datasets/artem9k/ai-text-detection-pile
- **Size**: 1.3M examples (~990K human + 340K AI)
- **Domains**: Long-form essays, GPT-2/3/ChatGPT/GPT-J
- **Value**: Massive scale, essay-focused — complements our news/wiki bias
- **Action**: Download human subset, filter by length 200-2000 words, sample 10K

### 2. GRiD (GPT Reddit Dataset)
- **Size**: Context-prompt pairs with human + ChatGPT responses
- **Domains**: Reddit conversations across subreddits
- **Value**: Authentic casual/informal human writing — our weakest area
- **Action**: Extract human responses, deduplicate against HC3/IMDB

### 3. SuperAnnotate Multi-Domain
- **Domains**: Wikipedia, Reddit, arXiv, conversational
- **Value**: Curated, domain-labeled, verified human-written
- **Action**: Use as validation set for cross-domain generalization testing

### 4. MAGE Benchmark (ACL 2024)
- **URL**: https://github.com/yafuly/MAGE
- **Domains**: CNN/DailyMail, DialogSum, PubMedQA, IMDb
- **Value**: Multi-domain benchmark with GPT-4 counterparts
- **Action**: Use for OOD evaluation, not training

### 5. RAID Benchmark (June 2024)
- **Coverage**: Multiple domains, models, attacks, generation params
- **Value**: Standardized evaluation across detectors
- **Action**: Already partially used (dataset_raid_extract.jsonl consumed into v4)

## Recommended Data Augmentation Plan

### Phase 1: Quick wins (1-2 hours)
1. Download AI Text Detection Pile human subset from HuggingFace
2. Filter: English only, 200-2000 words, deduplicate against v4
3. Sample 5K diverse texts → add to DeBERTa v5 training

### Phase 2: Targeted gaps (2-4 hours)
4. Add Reddit casual writing (GRiD or direct scrape)
5. Add student essays (IvyPanda already partially included)
6. Add technical writing (Stack Overflow, GitHub READMEs)
7. Add non-native English (HC3 multilingual subset)

### Phase 3: Validation (1 hour)
8. Use MAGE + SuperAnnotate as held-out cross-domain test set
9. Measure DeBERTa v5 AUROC on each domain separately
10. Identify remaining weak domains → targeted data collection

## Impact on XGBoost Fusion

More diverse human data also helps XGBoost:
- Current OOD set (127 samples) is hand-curated
- With 5K+ diverse samples, XGBoost can learn when DeBERTa fails
- Expected: DeBERTa weight drops from 73% → 40-50%, other signals gain

## Sources
- [AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)
- [MAGE Benchmark (ACL 2024)](https://github.com/yafuly/MAGE)
- [Awesome Machine-Generated Text](https://github.com/ICTMCG/Awesome-Machine-Generated-Text)
- [Best Datasets for AI Detection Training](https://detector-checker.ai/devlopers/training-ai-detection-models/)
- [SuperAnnotate AI Detection](https://www.superannotate.com/blog/ai-content-detection-superannotate)
