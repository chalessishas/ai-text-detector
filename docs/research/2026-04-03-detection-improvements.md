# AI 文本检测改进方案调研 (2026-04-03)

> Research Loop 自动调研。基于当前项目诊断：DeBERTa 跨领域 AUROC 0.5-0.6、PPL 模型断路、8/27 测试失败。

## 1. 跨领域泛化（最紧迫）

### LCFA + DCL 框架
- **论文**: [Exploring Generalized Features For LLM-Generated Text Detection](https://link.springer.com/chapter/10.1007/978-981-95-3729-7_25) (ICIG 2025)
- **核心**: LLM-conditional feature alignment (LCFA) 引导模型学习跨领域不变特征；Dynamic Contrastive Learning (DCL) 增强对扰动的鲁棒性
- **对项目的价值**: 直接解决 DeBERTa AUROC 0.5-0.6 的根因（学到了模型指纹而非通用 AI 特征）
- **局限**: 需要多领域训练数据 + 可能需要改模型架构

### Sci-SpanDet
- **方法**: Section-conditioned stylistic modeling + multi-level contrastive learning
- **优势**: 捕捉 human-AI 差异的同时减轻 topic 依赖
- **对项目的价值**: 可以借鉴 contrastive learning 思路改进 DeBERTa 训练

## 2. 零样本检测（减少对训练数据的依赖）

### Fast-DetectGPT
- **论文**: [ICLR 2024](https://arxiv.org/abs/2310.05130) | [GitHub](https://github.com/baoguangsheng/fast-detect-gpt) | [Demo](https://fastdetect.net)
- **核心**: Conditional probability curvature — 用 sampling model 生成替代样本，比较条件概率曲率
- **性能**: 比 DetectGPT 快 340x，白盒/黑盒均提升 ~75%
- **2026 更新**: Llama3-8B 作为 scoring model 大幅提升对 LRM 生成文本的检测
- **对项目的价值**: **高优先级** — 可作为第 5 路信号加入融合。零样本 = 不需要训练 = 天然跨领域。需要 logprobs，现有 PPL 基础设施可复用
- **实现成本**: 中等 — 需要两次前向推理（sampling + scoring），但可用同一个模型

### Binoculars
- **当前状态**: 项目中已实现但 **已禁用**（llama3.2:1b + 3b 无区分度）
- **改进方向**: 用 Llama3-8B 级别模型可能恢复区分度，但本地 8B 模型资源开销大

## 3. 集成投票优化（直接适用于当前四路架构）

### Theoretically Grounded Hybrid Ensemble
- **论文**: [arXiv 2511.22153](https://arxiv.org/html/2511.22153v1) (2025.11)
- **核心**: RoBERTa 分类器 + GPT-2 概率检测 + 统计语言特征，三路加权投票，权重在概率单纯形上学习优化
- **关键成果**: **假阳性率降低 35%**
- **对项目的价值**: **直接可用** — 当前项目已有类似架构（DeBERTa + PPL + LR + Stat），但权重是手调的 140 行 if-else。改用学习优化的权重 = 同样的信号更好的融合
- **实现**: 用 held-out validation set 训练一个 meta-learner（简单 LR 或 MLP）来融合四路分数

### Domain Gating
- **论文**: [Domain Gating Ensemble Networks](https://arxiv.org/html/2505.13855) (2025.05)
- **核心**: 学习何时使用、何时不使用特定领域特征
- **对项目的价值**: 可以学习"DeBERTa 在此类文本上不可靠，降权"vs "DeBERTa 在此类文本上准确，升权"

## 4. 水印检测（辅助信号）

### Google SynthID
- **状态**: 2026 年标准水印方案，修改 token 生成概率分布
- **特点**: 对 copy-paste 鲁棒、人类不可感知、机器可验证
- **局限**: 只对 Google 模型生效，可被 paraphrase 去除
- **对项目的价值**: 低 — 无法检测非 Google 生成的文本

### OpenAI Watermark
- **状态**: 2026 年实验中，nudge word choices toward certain token sequences
- **对项目的价值**: 低 — 同上，只对 ChatGPT 生效

## 5. 减少假阳性的具体技术

### Hard Negative Mining with Synthetic Mirrors
- 在高数据领域（如 reviews）可以将假阳性率降低数量级
- **实现**: 用 AI 模型改写真人文本作为 hard negatives 加入训练

### ESL/非母语去偏
- 多个平台已针对非母语写作者去偏
- **对项目的价值**: 当前 DeBERTa 可能对正式/结构化的非母语写作产生假阳性

---

## 推荐行动（按优先级）

### P0: 立即修复
1. **修复 PPL 模型加载** — 更新 Ollama blob hash，恢复四路融合
2. **改用 Ollama API 而非 blob path** — 防止 hash 再次失效

### P1: 高 ROI 改进
3. **融合权重学习化** — 替换 140 行 if-else，用 held-out set 训练 meta-learner
4. **加入 Fast-DetectGPT 作为第 5 路信号** — 零样本跨领域，复用现有 PPL 基础设施
5. **Hard negative mining** — 用 AI 改写真人文本训练 DeBERTa，降低假阳性

### P2: 中期改进
6. **Contrastive learning 重训 DeBERTa** — 借鉴 LCFA/DCL 思路提升跨领域能力
7. **Domain gating** — 学习各信号在不同文本类型上的可靠度

### P3: 长期/备选
8. **SynthID/水印检测** — 仅作为辅助，不可作为主力
9. **Binoculars 用更大模型重试** — 需要 8B+ 模型，资源开销大

---

## Sources
- [Exploring Generalized Features For LLM-Generated Text Detection (LCFA)](https://link.springer.com/chapter/10.1007/978-981-95-3729-7_25)
- [Fast-DetectGPT (ICLR 2024)](https://arxiv.org/abs/2310.05130)
- [Fast-DetectGPT GitHub](https://github.com/baoguangsheng/fast-detect-gpt)
- [Theoretically Grounded Hybrid Ensemble](https://arxiv.org/html/2511.22153v1)
- [Domain Gating Ensemble Networks](https://arxiv.org/html/2505.13855)
- [AI Text Watermarks Guide 2026](https://www.aidetectors.io/blog/ai-text-watermarks-explained)
- [Comprehensive Review of AI-Generated Text Detection](https://www.sciencedirect.com/org/science/article/pii/S1546221826000482)
- [False Positives in AI Detection Guide 2026](https://proofademic.ai/blog/false-positives-ai-detection-guide/)
- [Efficient Detection with Lightweight Transformer](https://www.nature.com/articles/s41598-026-35203-3)
- [Checkfor.ai Technical Report](https://arxiv.org/html/2402.14873v1)
