# Cloud LLM API Logprobs for Perplexity Computation — Research Report

**Date:** 2026-04-03
**Context:** ai-text-detector 项目需要 token-level logprobs 来计算 perplexity，用于 AI 文本检测。当前方案用本地 llama-cpp-python (qwen3:4b)，调研云端替代方案。

---

## 核心发现

**计算 perplexity 需要 input/prompt token 的 logprobs，而非 output token 的。** 大多数云 API 只返回 output token 的 logprobs。要对已有文本计算 perplexity，需要用 "echo" 或 "prompt_logprobs" 参数，或使用 Completions API（非 Chat Completions）将文本放入 prompt 并要求回显。

---

## 1. 云 API Logprobs 支持对比

| Provider | Logprobs 支持 | Prompt/Input Logprobs | top_logprobs 上限 | 关键限制 | 适合计算 Perplexity？ |
|----------|-------------|----------------------|------------------|---------|---------------------|
| **OpenAI** | Yes (Chat + Completions) | Completions API 有 `echo=true`；Chat API **仅 output** | 5 (Chat), 5 (Completions) | Chat API 无法获取 input logprobs；Completions API 仅支持旧模型 (gpt-3.5-turbo-instruct) | **部分可行** — 需用旧 Completions API + echo，或将文本放入 prompt 让模型续写 1 token |
| **Anthropic (Claude)** | **No** | N/A | N/A | 官方 API 不提供 logprobs | **不可行** |
| **Google Gemini** | Yes (Vertex AI + Developer API) | `response_logprobs=True` 仅返回 output logprobs | 未明确限制 | 仅 output token logprobs | **需变通** — 同 OpenAI Chat，需要 prompt trick |
| **DeepSeek** | Yes (Chat) | **仅 output** logprobs | 20 | deepseek-reasoner 不支持 logprobs | **需变通** |
| **Together AI** | **Yes (含 echo)** | **支持 `echo=True`**，返回 prompt token logprobs | 不限（Completions API） | 仅 Completions API 支持 echo | **最佳选择** — 开源模型 + echo 参数直接获取 input logprobs |
| **Fireworks AI** | Yes | 支持 `prompt_logprobs` | 未明确 | — | **可行** |
| **Groq** | **No (文档有参数但未实现)** | N/A | N/A | 所有模型均不支持 logprobs | **不可行** |
| **Mistral AI** | Yes | 支持 `return prompt` logprobs | 20 | — | **可行** |
| **xAI (Grok)** | 部分 | Grok 4.20 不支持；旧版支持 | 8 | 模型版本差异大 | **不稳定** |
| **Ollama** | **Yes (v0.12.11+)** | 支持 prompt logprobs | 支持 top_logprobs | 需自行部署 GPU 服务器 | **完全可行** — 需要自己管理基础设施 |
| **vLLM** | **Yes** | `prompt_logprobs` 参数 | 无限制 | 需自行部署；长文本 prompt_logprobs 可能较慢 | **完全可行** — 最灵活但需自管理 |

### 关键结论

**能直接对 input text 计算 perplexity 的 API（无需 trick）：**
1. **Together AI** — Completions API + `echo=True` + `logprobs=1`
2. **Fireworks AI** — 支持 prompt_logprobs
3. **Mistral AI** — 支持 return prompt logprobs
4. **Ollama / vLLM (自部署)** — 完全控制

**需要 workaround 的 API：**
- OpenAI / DeepSeek / Gemini — 将文本放入 prompt，生成 1 token（max_tokens=1），只获取该 token 的 logprob。或者将文本拆成 sliding window 逐段续写。精度有限，不如直接获取所有 input token logprobs。

---

## 2. 成本对比（~1000 texts/day × ~500 tokens/text = 500K tokens/day）

### 假设
- 每天处理 1000 篇文本，每篇 ~500 tokens
- 每日总量：~500K input tokens（0.5M tokens/day）
- 每月：~15M tokens/month
- 对于 perplexity 计算，主要消耗 **input tokens**（文本本身），output 极少（仅 1 token 或不需要）

### 2.1 托管 API 方案

| Provider | Model | Input Price ($/1M tokens) | 每日成本 (0.5M tokens) | 每月成本 (15M tokens) | 备注 |
|----------|-------|--------------------------|----------------------|---------------------|------|
| **Together AI** | Llama 3.1 8B | ~$0.10 | **$0.05** | **$1.50** | 最便宜，支持 echo |
| **Together AI** | Llama 4 Maverick | $0.27 | $0.14 | $4.05 | GPT-4 级别 |
| **Together AI** | Llama 3.1 70B | $0.90 | $0.45 | $13.50 | 更强但贵 |
| **Fireworks AI** | Qwen3 8B | $0.20 | $0.10 | $3.00 | 接近当前本地模型 |
| **Fireworks AI** | Llama 3.1 8B | ~$0.20 | $0.10 | $3.00 | — |
| **Mistral AI** | Mistral Nemo (12B) | $0.02 | **$0.01** | **$0.30** | 极便宜但需验证 logprobs 质量 |
| **Mistral AI** | Mistral Small 3.1 | $0.20 | $0.10 | $3.00 | — |
| **DeepSeek** | DeepSeek V3 | $0.14 | $0.07 | $2.10 | 仅 output logprobs，需 trick |
| **OpenAI** | GPT-4o-mini | $0.15 | $0.08 | $2.25 | 仅 output logprobs |
| **OpenAI** | GPT-3.5 Turbo Instruct | ~$1.50 | $0.75 | $22.50 | 支持 echo 但贵且老旧 |
| **Google** | Gemini 2.5 Flash-Lite | $0.10 | $0.05 | $1.50 | 仅 output logprobs |
| **Google** | Gemini 2.5 Flash | $0.30 | $0.15 | $4.50 | 仅 output logprobs |

### 2.2 自部署方案

| 方案 | GPU | 每小时成本 | 每日成本 (假设 2h 处理) | 每月成本 | 优点 | 缺点 |
|------|-----|-----------|----------------------|---------|------|------|
| **RunPod Pod** (A40) | A40 48GB | $0.35/hr | $0.70 | $21.00 | 完全控制，prompt_logprobs | 需管理，冷启动 |
| **RunPod Pod** (L4) | L4 24GB | $0.44/hr | $0.88 | $26.40 | 较新 GPU | L4 更贵 |
| **RunPod Serverless** | 按需 | ~$0.0004/sec | ~$0.50-1.00 | $15-30 | 自动缩放，按秒计费 | 冷启动 2-5s |
| **Modal Labs** | A10G/L4 | ~$0.50-0.80/hr | $1.00-1.60 | $30-48 | 极低冷启动(2-4s)，scale-to-zero | 比 RunPod 略贵 |
| **Modal Labs** (Flex) | 按需 | 30-40% 折扣 | $0.70-1.12 | $21-34 | 降本 | 需要 Flex 方案 |

**自部署处理时间估算：**
- 1000 texts × 500 tokens = 500K tokens
- 8B 模型在 A40 上：~50-100 tokens/sec prompt processing
- 总处理时间：500K / 100 = ~83 min ≈ 1.5 hours
- 如果用 vLLM batching：可以并行处理，~30-60 min

---

## 3. Ollama 云部署方案

| 平台 | 部署难度 | GPU 可用性 | 定价模式 | 适合场景 |
|------|---------|-----------|---------|---------|
| **RunPod** | **简单** — 官方有 Ollama 教程，Docker 镜像直接用 | A40/L4/A100 | $0.35+/hr | 最推荐，文档齐全 |
| **Modal Labs** | **简单** — 有 vLLM 一键部署，Ollama 需自定 Docker image | A10G/L4/A100/H100 | ~$0.50/hr+ | 适合 serverless，scale-to-zero |
| **Fly.io** | 中等 — 需自配 Docker，GPU 实例有限 | L40S | $2.50/hr (GPU) | 贵，不推荐 |
| **Railway** | **不适合** — 无 GPU 支持 | 无 | N/A | CPU 只能跑极小模型，不可行 |
| **Replicate** | 简单 — 但不适合自定 logprobs 场景 | 按需 | $0.0005-0.005/sec | 预打包模型，灵活性差 |

### 推荐部署方式：RunPod + Ollama Docker

```bash
# RunPod Pod 设置
# GPU: A40 (48GB VRAM)
# Docker Image: ollama/ollama
# Expose Port: 11434
# Env: OLLAMA_HOST=0.0.0.0

# 进入 Pod 后
ollama pull qwen3:4b   # 或任意模型
# API 即可通过 https://<pod-id>-11434.proxy.runpod.net 访问
```

---

## 4. 专门的 Perplexity/Logprob 服务

| 服务/工具 | 类型 | 说明 |
|-----------|------|------|
| **Perplexity AI API** | 搜索 API，**非** logprob 服务 | 名字误导 — 这是一个搜索引擎 API，不提供 logprobs |
| **GPTZero API** | AI 检测 SaaS | 提供 AI 检测分数，内部用 perplexity 但不暴露 logprobs |
| **Binoculars** (开源) | 本地工具 | 双模型交叉 perplexity 检测，需本地 GPU |
| **openlogprobs** (GitHub) | 开源库 | 从各种 LLM API 提取完整 next-token 概率分布 |
| **EleutherAI lm_perplexity** | 开源工具 | 直接计算 perplexity，需本地模型 |

**结论：没有专门的 "Perplexity-as-a-Service" API。** 需要自己通过 LLM API 的 logprobs 功能计算。

---

## 5. 推荐方案（按优先级）

### 方案 A：Together AI Completions API（最佳性价比）
- **模型：** Llama 3.1 8B（$0.10/1M tokens）
- **方法：** `echo=True` + `logprobs=1`，直接获取 input token logprobs
- **月成本：** ~$1.50（15M tokens/月）
- **优点：** 无需管理基础设施、原生支持 prompt logprobs、极低成本
- **缺点：** 模型与当前本地 qwen3:4b 不同，perplexity 分布会变化，需重新校准阈值
- **实施难度：** 低（API 调用 + 阈值重新校准）

### 方案 B：Fireworks AI + Qwen3 8B
- **模型：** Qwen3 8B（$0.20/1M tokens）
- **方法：** prompt_logprobs 参数
- **月成本：** ~$3.00
- **优点：** 模型家族与当前方案接近（同为 Qwen），迁移成本低
- **缺点：** 比 Together AI 贵 2x

### 方案 C：RunPod + Ollama/vLLM（最大灵活性）
- **GPU：** A40，按需 $0.35/hr
- **方法：** 自部署 qwen3:4b 或任意模型，完全控制 prompt_logprobs
- **月成本：** ~$15-25（每天跑 1.5-2 小时）
- **优点：** 可用与本地完全相同的模型，无需重新校准；完全控制
- **缺点：** 需管理部署、冷启动、监控

### 方案 D：Mistral Nemo（最低成本试验）
- **模型：** Mistral Nemo（$0.02/1M tokens）
- **月成本：** ~$0.30
- **优点：** 极其便宜，适合快速验证 API logprobs 管线是否可行
- **缺点：** 需验证 logprobs 质量；12B 模型可能 perplexity 分布与 4B 不同

---

## 6. 对当前项目的建议

当前项目用本地 `qwen3:4b` via `llama-cpp-python` 计算 perplexity。迁移到云 API 需要考虑：

1. **模型差异 = 阈值重新校准**：不同模型的 perplexity 分布不同。切换模型后，LR 模型和所有阈值都需要重新训练/校准。
2. **Prompt logprobs 是关键**：必须能获取 input text 的 logprobs，而非只是 output。Together AI 的 `echo=True` 是最干净的方案。
3. **batch processing vs real-time**：如果是离线批处理（训练数据标注），可以用便宜方案慢慢跑。如果是实时检测（用户提交文本），需要低延迟方案。
4. **混合方案**：生产环境保持本地 Ollama（零延迟），用云 API 做大批量数据标注和训练。

---

## Sources

- [OpenAI Logprobs Cookbook](https://cookbook.openai.com/examples/using_logprobs)
- [OpenAI Completions API Reference](https://developers.openai.com/api/reference/resources/completions/methods/create)
- [OpenAI Community: Prompt Token Logprobs](https://community.openai.com/t/get-logprobs-for-prompt-tokens-not-just-for-completion/717289)
- [Together AI Logprobs Docs](https://docs.together.ai/docs/logprobs)
- [Together AI Pricing](https://www.together.ai/pricing)
- [Fireworks AI Text Models](https://docs.fireworks.ai/guides/querying-text-models)
- [Fireworks AI Pricing](https://fireworks.ai/pricing)
- [Google Gemini Logprobs Blog](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/)
- [DeepSeek API Docs](https://api-docs.deepseek.com/api/create-chat-completion)
- [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Mistral AI Pricing](https://mistral.ai/pricing)
- [Groq Community: Logprobs Feature Request](https://community.groq.com/t/add-support-for-logprobs-in-model-api-response/193)
- [Ollama v0.12.11 Release (logprobs)](https://github.com/ollama/ollama/releases/tag/v0.12.11)
- [Ollama Generate API](https://docs.ollama.com/api/generate)
- [vLLM Prompt Logprobs](https://discuss.vllm.ai/t/what-is-the-purpose-of-prompt-logprobs/1714)
- [RunPod Ollama Tutorial](https://docs.runpod.io/tutorials/pods/run-ollama)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [Modal Labs vLLM Deploy](https://modal.com/blog/how-to-deploy-vllm)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Binoculars Paper](https://arxiv.org/html/2401.12070v3)
- [openlogprobs GitHub](https://github.com/justinchiu/openlogprobs)
- [Anthropic OpenAI SDK Compatibility](https://platform.claude.com/docs/en/api/openai-sdk)
- [Leveraging Logprobs (Sophia Willows)](https://sophiabits.com/blog/leveraging-logprobs)
