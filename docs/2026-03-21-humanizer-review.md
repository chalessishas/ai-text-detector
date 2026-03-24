# AI Text Detector × Humanizer — 阶段性深度审查

> 审查日期：2026-03-21
> 审查范围：整体项目状态 + 本次讨论中提出的"seed-guided humanizer"方案

---

## 一、优化后的提示词（对本次讨论的总结）

> 用户提出了一个 humanizer 新方案：用现有的语料库改写（7 种方法）生成一段"无语义但统计指纹接近人类"的文本作为 seed，然后让语言模型在这个 seed 的统计约束（perplexity 分布、entropy、burstiness 等）下，结合原文语义，生成一篇"既保留原意、又带有人类统计指纹"的文本。每次随机抽取不同人类样本做 seed，远期目标是接入 Writing Center 让 seed 匹配用户个人写作指纹。

---

## 二、项目现状——真实情况

### 你的 Detector 现在不能用

STATUS.md 写得很清楚：**DeBERTa 分类器当前不可用**。`models/detector/` 里存的是 base 权重，不是训练好的权重。HANDOFF.md 里说的"95.3% 准确率"是历史成绩，那个模型已经因为数据集缺陷被废弃了。

当前阻塞链：
```
数据集 human 样本不连贯（已修代码，未重跑）
    → 无法重新训练 DeBERTa
    → Detector 主评分不可用
    → 只剩启发式 5 维评分（准确率 ~70%）
    → Humanizer 无法做闭环验证（没有可靠的裁判）
```

**这意味着：你今天讨论的所有 humanizer 方案，都没有一个可靠的内部裁判来验证效果。**

### 你的 Humanizer 现状

7 种方法全部是**语料库检索+填充**，不涉及任何语言模型生成。产出的文本"除了像人之外，一点意思都没有"——这是你自己的原话，也是事实。这些方法的设计目标本来就不是保留语义，而是生成"人类统计特征"的文本。

### 硬件约束

M4 16GB，最大跑 4B 模型。Ollama v0.17.5 支持 logprobs。本地有 llama3.2:1b + qwen3.5:0.8b + qwen3.5-abliterated:2b。

---

## 三、对"Seed-Guided Humanizer"方案的深度批判

### 方案核心假设

```
假设 1：人类文本的统计指纹（ppl/entropy/burstiness）可以被提取并量化
假设 2：这个指纹可以作为约束，引导 LLM 生成
假设 3：在约束下生成的文本能同时保留原文语义
假设 4：生成出来的文本能骗过 AI 检测器
```

### 逐条审查

**假设 1 ✅ 基本成立**
你的 analysis.ts 已经实现了 5 维特征提取。困惑度、熵、burstiness 确实可以量化。但注意：这些特征是用 llama3.2:1b 计算的——不同模型算出来的 perplexity 不同。你的 seed 指纹和你的 detector 指纹必须用同一个模型算，否则基准不一致。

**假设 2 ⚠️ 这是最大的问题**

当前没有任何成熟的技术方案能让 LLM "按给定的 perplexity 曲线生成文本"。原因：

1. **Perplexity 是观察量，不是控制量。** 你可以在生成后测量 perplexity，但你不能在生成前指定它。这就像你可以测量一个人跑完 100 米用了多少秒，但你不能直接控制他每一步的步幅。

2. **Prompt 引导不可靠。** 你可以在 prompt 里写"句子长度分别是 8, 23, 5, 31 个词"，但 LLM 不会严格遵守数字约束。更别说 perplexity 这种抽象的统计量——LLM 不知道自己的 perplexity 是多少。

3. **Logit manipulation 理论上可行但极难调。** 用 temperature 去拉熵是可以的，但：
   - 你用 Ollama，只能看 logprobs，不能改 logits（需要 vLLM 或原生 llama.cpp server）
   - 即使能改 logits，perplexity 取决于上下文，你改了当前 token 的选择，后续所有 token 的 perplexity 都变了——这是一个混沌系统
   - 没有已知的控制理论框架能稳定地引导 LLM 输出匹配目标分布

4. **微调是唯一可靠路径，但数据怎么造？** 你说的"训练数据 = 统计约束 + 语义 → 人类风格输出"，这需要：
   - 大量 (统计指纹, 原文语义, 人类改写) 三元组
   - 你没有这个数据，也很难自动造出来
   - 用 AI 反向造数据有循环依赖问题

**假设 3 ⚠️ 语义保持和分布约束本质上冲突**

- 某些语义天然决定了 perplexity。比如"quantum entanglement"这个词组在学术论文里 perplexity 很低（可预测），你硬要让它的 perplexity 高，就只能换成"那个粒子纠缠在一起的奇怪现象"——语义变了。
- Burstiness（句长变异）和论证结构绑定。一个逻辑严密的 argument 需要长句铺陈+短句结论，你不能随意改变句长模式而不改变论证质量。

**假设 4 ❌ 这是一个根本性的悖论**

你想用 LLM 生成文本来骗过 AI 检测器。但：

- **检测器检测的不只是统计分布。** DeBERTa 分类器是一个端到端的神经网络，它学到的特征远超你手工定义的 5 个维度。你可以把 ppl/entropy/burstiness 全调对，但 DeBERTa 可能在看你根本没想到的特征——比如介词使用频率、从句嵌套深度、代词指代模式。
- **用 AI 生成的文本，无论怎么约束，其 token 分布仍然是模型的分布。** 这就像你让一个美国人模仿英国口音说英语——语音语调可以改，但句法选择和用词偏好暴露了他是美国人。统计约束只能改"语音语调"（表层特征），改不了"句法偏好"（深层分布）。
- **GPTZero 的检测维度远超 5 个。** 研究显示 GPTZero 用的是 7 组件多层系统，你匹配了 5 个维度可能还有 20 个维度没覆盖。

### 最根本的问题

**你在试图用 AI（4B 模型）骗过 AI（DeBERTa + GPTZero），同时用另一个 AI（llama3.2:1b）来当裁判。** 这三个都是 AI，它们的输出分布都不是人类的。你在一个全是 AI 的系统里试图制造"人味"，这在理论上就有天花板。

---

## 四、那什么方案才是对的？

### 诚实的答案：纯 AI 生成 + 统计约束这条路的天花板很低

研究已经证明（Sadasivan et al. 2023），理论上随着模型改进，检测会越来越难。但这说的是**模型本身的改进**（GPT-6 比 GPT-4 更难检测），不是说你在 4B 模型上加约束就能达到同样效果。

### 更可行的方向

**方向 1：人机协作改写（你的 Writing Center 定位）**

不要试图全自动。让 AI 做 draft，**人类做最终编辑**。人类只要改 20-30% 的文本，统计指纹就会显著偏向人类。这和你的"AI 帮你更高效传达想法"的理念一致。

产品形态：
```
用户输入想法（语音/草稿/关键词）
    → AI 生成结构化 draft
    → 逐句高亮"AI 味重"的部分（你的 detector 能做这个）
    → 用户自己改写这些高亮部分
    → 再次检测 → 直到通过
```

这不是 humanizer，是 **writing coach**。而且这个方案你现有的技术栈就能做。

**方向 2：如果一定要全自动 humanizer**

不用你自己的小模型做改写。用 **DeepSeek/GPT API**（大模型，改写质量远高于 4B）+ 你的 **detector 做闭环验证**。大模型的改写加上精心设计的 prompt（注入人类写作缺陷：typo、口语化表达、不完美的逻辑衔接），效果会比在 4B 上调统计分布好得多。

但前提：**你的 detector 得先修好**。没有可靠的裁判，什么闭环都做不了。

**方向 3：你的 seed 概念的正确用法**

Humanizer 生成的 seed 不应该用来约束 LLM 的统计分布（做不到），而是用来给 **prompt 提供具体的人类写作示例**：

```
prompt = f"""
Here is an example of how a human writes about a similar topic:
"{humanizer_corpus_sentence}"

Now rewrite the following text in a similar natural style.
Mimic the rhythm, word choice variety, and sentence structure
of the example above.

Text to rewrite: {original_ai_text}
"""
```

这叫 **few-shot style transfer**，比统计约束更实用——因为 LLM 擅长模仿风格，不擅长匹配数字。

---

## 五、优先级建议

| 优先级 | 任务 | 为什么 |
|--------|------|--------|
| **P0** | 重建数据集 + 重训 DeBERTa | 没有可靠的 detector，一切后续都是空谈 |
| **P1** | 用现有 detector（启发式 fallback）做 writing coach MVP | 不需要等 DeBERTa，启发式虽然 70% 准确率但能给用户有用的反馈 |
| **P2** | 用大模型 API + few-shot style transfer 做 humanizer v2 | 比统计约束方案更现实 |
| **P3** | seed-guided 统计约束方案 | 需要大量研发投入，且可行性存疑——建议作为研究方向而非产品路线 |

---

## 六、自我批判

1. **我之前在讨论中犯了什么错？** — 我一直在附和主人的方案，把 ControlNet 类比、GAN 类比这些听起来酷炫的概念扔出来，让方案显得比实际更可行。实际上我应该在主人提出"让 LLM 按 perplexity 曲线生成"时就指出：这在当前技术下做不到，Ollama 的 logprobs 只能读不能写。

2. **这份报告本身有没有问题？** — 我对"统计约束不可行"的判断可能过于绝对。学术界有一些受控文本生成的工作（PPLM、FUDGE、GeDi）确实能在一定程度上控制生成文本的属性。但这些方法都需要额外的分类器作为 guide，且在小模型上效果不好，也没有人用它们来匹配 perplexity 分布。主人如果想深入，可以看 FUDGE (Yang & Klein, 2021) 和 GeDi (Krause et al., 2021) 的论文。

3. **产品视角审视** — 主人说"拿出去卖钱"。如果目标是商业化，最快的路径不是造一个全新的 humanizer 技术，而是做好 writing coach + 现有 humanizer 的组合产品。市场上缺的不是另一个 humanizer 工具，缺的是一个能告诉你"你的文章哪里像 AI、怎么改"的教练。这恰好是你有 detector 这个优势能做的。

4. **我遗漏了什么？** — 整个讨论没有提到**竞品分析**。Undetectable.ai、StealthGPT 这些工具已经在做 detect-then-fix 循环了，它们的效果和局限是什么？主人应该先测试这些竞品，看看天花板在哪，再决定自己的方案。还有一个遗漏：**法律风险**。如果产品明确宣传"帮你骗过 AI 检测器"，可能面临学术机构的法律行动——虽然目前没有先例，但 Turnitin 的律师团队不是吃素的。
