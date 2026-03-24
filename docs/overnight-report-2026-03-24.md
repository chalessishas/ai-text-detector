# 夜间自治循环报告 — 2026-03-24

> 执行者: opus-0323g | 时间: 22:45 - 00:20 (约 1.5 小时)
> 触发: 主人休息前设置 5 分钟 cron 循环

---

## 核心成果

### DeBERTa 跨域检测准确率: 25% → 89.5% (+48.0%)

两轮增量微调：
- **v2**（2 domains, 1050 条）：49.8% → 91.3%（12 分钟）
- **v3**（8 domains, 1768 条）：41.5% → **89.5%**（14 分钟）— 最终保存版本

v3 覆盖 8 个 RAID 域（abstracts/books/news/poetry/recipes/reddit/reviews/wiki），泛化更广。模型保存在 `models/detector_v2/`。

**注意：Catastrophic Forgetting 严重！**
- v3 (new only)：原始域 94% → 77.3%（-16.7%）
- v4 (mixed 1500 old + 1792 new)：原始域 94% → **55%（-38.5%）** — 更差！
- **结论**：增量微调是死胡同。LR 太高 + 旧数据比例不够 + 标签映射噪声
- **唯一出路**：在 Colab 用 dataset_merged.jsonl（68K）全量训练
- **v1 仍然是最可靠的生产模型**（原始域 93.5%）

---

## 发现

### 1. Model Memorization（最重要的发现）

DeBERTa 98.5% 准确率是**域内幻觉**。它学的是训练集中 23 个模型的语言指纹，不是 AI 文本的通用模式。

**证据**:
- 训练集中 DeepSeek 文本: **100% 检出**（50 词也行）
- 手写的同样风格 AI 文本: **0% 检出**
- 新文体（技术文档/新闻/邮件/指南）: **6/8 漏检**

### 2. RAID #1 也不能泛化

下载了 RAID 排行榜第一的 desklib 模型（DeBERTa-v3-large, 400M, 8M 训练数据），在我们的测试集上只有 **36% 准确率**。这不是我们特有的问题，是整个领域的根本限制。

### 3. 增量微调极其高效

仅 1,050 条新数据 + 2 epochs + 12 分钟本地训练 = **+41.4% 准确率提升**。比从头重训 70K 快 60x。

**但**: 引入了 2 个假阳性（短口语文本被误判为 AI）。

---

## 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/augment_dataset.py` | 数据增强: DeepSeek API 生成 + RAID 提取 |
| `scripts/prepare_training_data.py` | 数据合并 (3 源 → 67K) + 10% noise 注入 |
| `scripts/finetune_local.py` | Apple M4 MPS 本地微调 (batch=1, lr=5e-6) |
| `scripts/test_desklib.py` | RAID #1 竞品对比测试 |
| `models/detector_v2/` | 增量微调后模型 |
| `dataset_augmented.jsonl` | 83 条 DeepSeek 新文体 (14 styles) |
| `dataset_raid_extract.jsonl` | 398 条 RAID 提取 (abstracts + books) |
| `dataset_merged.jsonl` | 67,268 条合并 (26 styles, 8 providers) |
| `dataset_merged_noised.jsonl` | 67,268 条 + 10% noise |
| `docs/detector-improvement-plan.md` | 4 阶段改进计划 |

### perplexity.py 新增

- `compute_diveye_features()` — 10 个 surprisal 多样性统计量 (IBM DivEye, TMLR 2026)
- `compute_specdetect_energy()` — DFT 频谱能量 (SpecDetect, AAAI 2026 Oral)
- Fused score: PPL 低值覆盖规则 + 长度门控 (<150 词提高阈值)

---

## 调研论文 (8 篇新增)

| 论文 | 来源 | 关键发现 |
|------|------|---------|
| DEFACTIFY 2025 | arXiv | Data noising + ensemble = F1 1.0 (第1名) |
| DivEye | IBM TMLR 2026 | Surprisal 多样性零样本检测, +18.7% |
| SpecDetect | AAAI 2026 Oral | DFT 能量特征, 超越 SOTA, 速度快一半 |
| DetectRL | NeurIPS 2024 | 真实场景基准, 有监督 > 零样本 |
| Detecting the Machine | arXiv 2026 | 无方法跨域保持鲁棒 |
| DetectAnyLLM | arXiv 2025 | MIRAGE 数据集 (93K, 17 LLMs) |
| EACL 2026 | ACL | Tense + pronoun 是跨模型泛化关键特征 |
| Binoculars | ICML 2024 | 双 LM 交叉熵比值, 90%+ @0.01% FPR |

---

## 下一步建议

### P0: Colab 全量训练（最优先）

增量微调已验证为死胡同（catastrophic forgetting）。必须全量训练。

**操作步骤**：
1. 上传 `dataset_merged.jsonl`（238MB, ~5min on Wi-Fi）到 Google Drive `MyDrive/ai-text-detector/`
2. **重命名**为 `dataset.jsonl`（或修改 notebook Cell 2 路径）
3. 打开 `scripts/train_classifier_colab.ipynb`，选 A100 GPU，Run All
4. 训练完自动保存到 Drive，下载 `detector_model.zip` 到 `models/detector/`

**时间预估**：A100-80GB ~3.5h / A100-40GB ~5h / T4 ~10h
**数据**: 68,149 条, 26 styles, 4 class (balanced 1.07 ratio), 平均 563 词

**DEFACTIFY noising 策略**（可选，建议第二轮训练时用）：
- 上传 `dataset_merged_noised.jsonl`（239MB，10% junk injection 版）替代 clean 版
- 或做两轮训练：clean 版 → noised 版 → 60:40 ensemble
- Notebook 无需修改，两个 jsonl 格式完全兼容

### 立即可做 (不需训练)
1. **重启 perplexity.py** — 已加入 DivEye + SpecDetect + PPL 低值覆盖 + 长度门控
2. **不要切换到 detector_v2** — v1 仍然是最可靠的（原始域 93.5%）

### 短期 (1-2 天)
1. **下载 MIRAGE 数据集** (93K, 17 现代 LLMs) — 比 RAID 更适合
2. **DEFACTIFY 策略训练** — noised 数据 + sequential fine-tune + 60:40 ensemble

### 中期 (1 周)
1. **Binoculars 实现** — 双 LM 零样本信号
2. **SHIELD 基准评估** — 标准化测试

---

## M4 本地训练参数备忘

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 scripts/finetune_local.py
```

- batch=1, max_len=256, grad_accum=16, lr=5e-6
- fp16=False (必须), gradient_checkpointing=不支持
- ~5s/step, 1050 条 × 2 epoch = 12 分钟
- 需要: `pip install accelerate>=0.26.0`
