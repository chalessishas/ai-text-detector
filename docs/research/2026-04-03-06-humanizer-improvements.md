# Humanizer 改进研究报告（2025-2026）

> 生成时间：2026-04-03 | 范围：学术论文 + 商业产品 + 检测器动态

---

## 一、最新学术方案

### 1. StealthRL（2026.02，与我们方向最接近）

用 GRPO + LoRA 在 Qwen3-4B 上训练 paraphrase policy，对抗 4 个检测器的 ensemble。在 MAGE 测试集上将平均 AUROC 从 0.79 降到 0.43，攻击成功率 97.6%。

**关键设计**：composite reward = 检测器逃逸分 + 语义保持分（cosine similarity），LoRA rank=32，batch=16，group_size=8，仅需 3 epoch。

- 论文：[arxiv.org/abs/2602.08934](https://arxiv.org/abs/2602.08934)

### 2. AuthorMist（2025.03）

首创 "API-as-Reward" 方法——将 GPTZero / Originality.ai 等商用检测器 API 直接接入 GRPO reward loop，在 3B 模型上微调。攻击成功率 78.6%-96.2%，语义相似度 >0.94。

- 论文：[arxiv.org/abs/2503.08716](https://arxiv.org/abs/2503.08716)

### 3. Adversarial Paraphrasing（NeurIPS 2025）

**免训练**方案：用现成指令模型在检测器引导下迭代改写。对 RADAR（对抗训练过的检测器）检测率降低 64.49%，对 Fast-DetectGPT 降低 98.96%。同时有效移除 watermark。

- 论文：[arxiv.org/abs/2506.07001](https://arxiv.org/abs/2506.07001)
- 代码：[github.com/chengez/Adversarial-Paraphrasing](https://github.com/chengez/Adversarial-Paraphrasing)

---

## 二、Turnitin 最新动态

2025 年 8 月上线 **AI Bypasser Detection**，核心机制：

- **跨 humanizer 泛化训练**：用多种 humanizer 工具的输出训练检测模型
- **多层分析**：句子可预测性 + 节奏一致性 + 过渡行为 + 结构逻辑
- 不依赖 perplexity/burstiness，而是 transformer-based deep learning

但：Yale、Johns Hopkins、Northwestern 等 12+ 所大学已关闭 Turnitin AI 检测，Curtin 大学 2026.01 起全面停用。市场信号：**检测器公信力正在下降**。

- 来源：[turnitin.com/press](https://www.turnitin.com/press/turnitin-expands-capabilities-amid-rising-threats-posed-by-ai-bypassers)

---

## 三、商业 Humanizer 产品现状

| 产品 | 绕过率 | 弱点 |
|------|--------|------|
| Undetectable AI | ~94% | 最全面，但 Turnitin 新版后下降 |
| Humanize AI Pro | ~99% | 500 样本测试最高 |
| StealthWriter | ~91% | Originality.ai 仅 28%，语法差 |
| WriteHuman | 不稳定 | 2000 字以上仅 78%，Copyleaks 失败 |
| UndetectedGPT | ~96% | 可读性最好 |

共同趋势：**纯改写方案天花板明显**，Turnitin bypasser detection 上线后普遍下降。

- 来源：[thehumanizeai.pro](https://thehumanizeai.pro/articles/best-ai-humanizer-2026-independent-ranking)

---

## 四、对本项目的具体改进建议

### 高优先级

1. **升级 StealthRL 实现**：从 Qwen2.5-3B 切换到 Qwen3-4B（StealthRL 论文验证了这个 base），LoRA rank 提升到 32，加入 multi-detector ensemble reward（至少包含 RoBERTa-based + Fast-DetectGPT + 我们自己的检测器）

2. **引入 API-as-Reward（AuthorMist 方法）**：将 Turnitin/GPTZero 的 API 分数直接作为 reward signal。即使没有 API 访问，可以用我们自己的 DeBERTa 检测器 + 开源 RADAR 作为 proxy

3. **Adversarial Paraphrasing 作为 fallback**：对 RL 训练失败的 case，用 NeurIPS 2025 的免训练方案做 iterative refinement（检测器引导的逐步改写），成本低、效果好

### 中优先级

4. **Composite Reward 设计**：StealthRL 的 reward = α×(1-detection_score) + β×cosine_sim + γ×fluency_score。建议加入 **Turnitin-specific 信号**（节奏一致性、过渡行为分析的反向指标）

5. **训练数据多样化**：不只用 ChatGPT 输出，加入 Claude/Gemini/Llama 的输出做训练，提升泛化能力

### 低优先级

6. **watermark removal 能力**：Adversarial Paraphrasing 论文证明其方法可同时移除统计 watermark，如果未来 OpenAI/Google 部署 watermark，这是现成方案
