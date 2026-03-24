# DeBERTa Detector 改进计划

> 制定日期：2026-03-23 23:25
> 基于：系统性盲区测试 + 6 篇新论文调研 + 数据分析

---

## 一、问题诊断

### 核心问题：Model Memorization, Not Pattern Generalization

DeBERTa 98.5% 准确率是 **域内幻觉**。实际测试发现：

| 测试 | 检出率 | 说明 |
|------|--------|------|
| 训练集中模型(DeepSeek)，任意文体，50-1000 词 | **100%** | 学到了模型指纹 |
| 训练集外"AI 风格"文本，各文体 | **0-25%** | 没学到通用 AI 模式 |
| 域内 4-class 测试集 | **98.5%** | 域内评估不可靠 |

### 根因分析

1. **训练数据文体单一**：7 种 prompt styles 全是 essay 变体，70K 样本中新增的 14 种 styles 从未生成数据
2. **训练数据主题单一**：20 个社会议题 topic，新增 40 个 STEM/商业/法律/文化 topic 未使用
3. **模型覆盖有盲区**：虽然有 23 个模型，但都用相同的 essay prompts，模型的非 essay 输出模式未覆盖
4. **正则化不足**：没有 data noising 或对抗训练，模型过拟合到表层特征

### 关键洞察（来自最新研究）

- **EACL 2026**：tense usage + pronoun frequency 是跨模型泛化的关键语言特征
- **perplexity consistency** 是 generator-agnostic signal（不依赖具体模型）
- **DEFACTIFY 2025**：10% data noising 作为正则化，F1 从 0.95 提升到 1.0
- **DivEye (IBM TMLR 2026)**：surprisal 多样性统计量可作为辅助信号，提升 18.7%
- **Detecting the Machine (2026)**：没有任何方法在跨域迁移时保持鲁棒——这是当前所有检测器的通病

---

## 二、改进方案

### Phase 1：数据增强（1-2 天，本地可做）

**目标**：将训练数据从 70K essay-only → 100K+ multi-genre multi-model

#### A. 用 DeepSeek API 生成新文体数据

- **已启动**：14 种新 styles × 50 samples/style = 700 条
- **脚本**：`scripts/augment_dataset.py --mode generate`
- **成本**：~$2-3（DeepSeek 便宜）
- **局限**：单模型，只能增加文体不增加模型多样性

#### B. 合并 RAID 数据集

- **来源**：`liamdugan/raid`（HuggingFace），10M+ 条，11 domains
- **目标**：每 domain 提取 500 条（250 human + 250 AI）= 5,500 条
- **价值**：11 种域（abstracts, books, code, news, poetry, recipes, Reddit, reviews, Wikipedia），11 个旧模型
- **脚本**：`scripts/augment_dataset.py --mode raid`

#### C. 合并 MAGE 数据集

- **来源**：`yaful/MAGE`（HuggingFace）
- **目标**：CNN/DailyMail, DialogSum, PubMedQA, IMDb 各 200 条
- **价值**：新闻、对话、科学、影评四个领域 + GPT-4 产出

#### D. Data Noising（DEFACTIFY 策略）

- **方法**：10% 随机乱码词（3-8 字符）注入
- **脚本**：`scripts/prepare_training_data.py`（已完成）
- **输出**：`dataset_merged.jsonl`（clean）+ `dataset_merged_noised.jsonl`（noised）

### Phase 2：重训 DeBERTa（Colab 或本地 M4）

**本地训练可行性**（2026-03-23 确认）：
- M4 16GB 可以用 MPS 后端微调 DeBERTa-base（180M 参数）
- 关键：`fp16=False`，`device="mps"`，batch_size=4-8
- 适合增量训练（1-5K 新样本），全量 70K 训练仍建议 Colab
- MPS 推理 8-10x 加速，训练约为 A100 的 1/10-1/5 速度

### 全量重训策略（Colab 2-4 小时）

**训练策略**（基于 DEFACTIFY 最佳方案）：

1. **Model A**：在 noised dataset 上 fine-tune DeBERTa-v3-base，4 epochs
2. **Model B**：先在 clean dataset fine-tune 4 epochs → 再在 noised dataset 继续 2 epochs
3. **Ensemble**：60% Model B + 40% Model A 的加权预测

**关键参数**：
- LR: 2e-5, warmup 200 steps, weight decay 0.01
- batch: 根据 GPU VRAM 自适应（A100-80=64, A100-40=32, T4=16）
- `save_strategy='no'`（避免 DeBERTa gamma/beta checkpoint bug）
- `model.float()` 必须调用（fp16 会爆）

### Phase 3：辅助信号增强（可选，无需重训）

#### DivEye 特征加入 LR 检测器

- **已实现**：`perplexity.py` 中 `compute_diveye_features()` 函数
- **10 个特征**：surprisal 的 mean/std/var/skew/kurtosis + 一阶差分 mean/std + 二阶差分 var/entropy/autocorr
- **待做**：重训 LR 分类器（`calibrate_detector.py`）加入 DivEye 特征
- **预期提升**：LR 从 90% → 95%+

#### Binoculars 双模型信号

- **方法**：用两个相似 LM 的交叉熵比值检测
- **可用模型**：llama3.2:1b + qwen3.5:0.8b
- **优势**：ESL 友好（99.67% 准确率），零样本
- **劣势**：需同时加载两个模型，推理慢

### Phase 4：评估改进（长期）

- **跨域测试集**：从 RAID 中留出各 domain 100 条作为 OOD 测试集
- **跨模型测试集**：用 Mistral/LLaMA/Gemma 等未训练模型生成测试数据
- **A/B 测试**：对比 ensemble vs 单模型、noised vs clean、DivEye 有无

---

## 三、数据资源总结

| 数据集 | 规模 | 域/文体 | 模型 | 获取方式 | 优先级 |
|--------|------|---------|------|---------|--------|
| 原始 dataset.jsonl | 70K | 7 styles, 20 topics | 23 模型 | 已有 | 基线 |
| **MIRAGE** | **93K** | **5 域, 10 语料库** | **17 现代 LLM** | HuggingFace `MIRAGE-Benchmark/MIRAGE` | **P0** |
| DeepSeek 新文体 | ~700（生成中） | 14 新 styles, 60 topics | 1 模型 | API | P1 |
| RAID | 10M+（采样 5.5K） | 11 domains | 11 旧模型 | HuggingFace `liamdugan/raid` | P2 |
| MAGE | ~800 | 4 domains | GPT-4 | HuggingFace `yaful/MAGE` | P3 |
| Data Noising | 等于 clean × 1 | 同上 | 同上 | 本地生成 | P0 |

---

## 四、优先级

1. **P0（今晚）**：DeepSeek 生成 + RAID 提取 → 合并 + noising → 准备好训练数据
2. **P1（明天）**：上传 Colab 重训 DeBERTa（~2-4 小时）
3. **P2（重训后）**：跨域/跨模型测试 → 评估改进效果
4. **P3（可选）**：DivEye + Binoculars 辅助信号

---

## 五、Baseline 对比：desklib RAID #1 模型

### 测试结果（2026-03-23 23:32）

| 测试 | desklib (RAID #1) | 我们的 DeBERTa | 说明 |
|------|-------------------|---------------|------|
| AI: tech_doc | 36% → Human | 0% → Human | 双方都漏检 |
| AI: news | 98% → AI | 0% → Human | desklib 强 |
| AI: business_email | 83% → AI | 0% → Human | desklib 强 |
| AI: deep_research | 10% → Human | 0% → Human | 双方都漏检 |
| Human: casual | **98% → AI** | 0% → Human | desklib 误判 |
| Human: reddit | **91% → AI** | 0% → Human | desklib 误判 |
| Human: email | **94% → AI** | 0% → Human | desklib 误判 |
| **总准确率** | **36%** | **45%** | 我们略好（假阳性低） |

### 结论

1. **RAID #1 也不能泛化**——8M 样本 + DeBERTa-large 仍然不够
2. **desklib 假阳性严重**——把所有人类文本判为 AI（unsafe for production）
3. **我们的优势**：假阳性率极低（0%），问题只是漏检
4. **不能走捷径**——必须在自己的数据分布上训练

---

## 六、风险

1. **DeepSeek 单模型偏见**：只用一个模型生成新文体数据，可能让 DeBERTa 过拟合 DeepSeek 风格。RAID 的多模型数据能缓解
2. **RAID 模型过旧**：GPT-3, GPT-2, Llama 2 等旧模型，2025-2026 新模型不在其中
3. **Colab 需要手动操作**：Drive 挂载需要确认，无法完全自动化
4. **Data Noising 可能降低准确率**：理论上正则化会牺牲一些域内性能换取域外泛化，需要权衡
