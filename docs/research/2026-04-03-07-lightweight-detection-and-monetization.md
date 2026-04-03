# 轻量检测路径与 Humanizer 变现研究

> 背景：XGBoost meta-learner 训练流程太重（Colab 训练 -> 下载 -> 集成），需要更轻量的检测改进路径。

## 1. Zero-shot 免训练检测方法

**Binoculars**（ICML 2024）是当前最强 zero-shot 方案。用两个近似 LLM（observer + performer）对比 perplexity 与 cross-perplexity 的差值打分，无需任何训练数据。在 FPR=0.01% 时检测率 >90%，优于 GPTZero、DetectGPT、DNA-GPT。缺点：需要 GPU 跑两个模型推理，不适合纯前端/serverless。[论文](https://arxiv.org/abs/2401.12070) | [代码](https://github.com/ahans30/Binoculars) | [API 服务](https://binocularsai.com/)

**DetectGPT** 用 log-probability 曲率判断，不需训练分类器，但需要目标模型的 log-prob API（GPT-4 已不提供），实用性下降。[论文](https://arxiv.org/abs/2301.11305)

**IPAD**（NeurIPS 2025）通过「逆向还原 prompt」判断文本是否由 AI 生成，在对抗改写（DIPPER attack）下比 RoBERTa baseline 高 9-13%。这种方法只需 LLM API 调用，无需本地训练。[论文](https://arxiv.org/abs/2502.15902) | [代码](https://github.com/Bellafc/IPAD-Inver-Prompt-for-AI-Detection)

**推荐**：IPAD 最适合我们——只需 LLM API、不需 GPU、抗改写能力强，可直接集成到现有的 DeepSeek/GPT 调用架构中。

## 2. LLM-as-Judge 检测

直接用 GPT-4 / Claude 判断「这段文本是否 AI 生成」，研究显示与人类判断的一致率约 70-80%。但存在系统性偏差（如偏好更长的回答），多模型投票（3-5 个 LLM）可降低偏差 30-40%，但成本线性增长。[参考](https://labelyourdata.com/articles/llm-as-a-judge) | [Galtea 研究](https://galtea.ai/blog/exploring-state-of-the-art-llms-as-judges)

现实中 SOTA 检测器在 real-world 场景下仍表现不稳定，尤其是面对改写后的文本。[ACL 2025 调研](https://aclanthology.org/2025.cl-1.8.pdf)

**推荐**：作为辅助信号加入 ensemble，不单独依赖。我们已有的 DeepSeek API 可零成本增加一路 LLM-judge 信号。

## 3. Humanizer 竞品定价与变现

| 产品 | 基础价 | 高级价 | 字数额度 |
|------|--------|--------|----------|
| **Undetectable.ai** | $9.99/月 | $49/月 (Business) | 10K-50K 词/月 |
| **WriteHuman** | $18/月 | $48/月 (Ultra) | 80 次-无限次，250-3000 词/次 |
| **StealthWriter** | ~$15/月 | $49/月 (Business) | 最高 50K 词/月 |

Undetectable.ai 是标杆：**34 人团队、零融资、月收入 $370K（2025.9）**，年化 ARR ~$4.4M。[来源](https://getlatka.com/companies/undetectable.ai)

定价共识：入门 $10-18/月，专业 $27-49/月，按字数/次数限制分层。

**推荐**：我们的 Humanizer 可以从 $9.99/月起步（10K 词），对标 Undetectable.ai。差异化方向：adversarial paraphrasing 多轮迭代质量更高 + 检测器自验证闭环。
