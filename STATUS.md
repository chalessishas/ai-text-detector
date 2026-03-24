# AI Text X-Ray — 项目状态

> 最后更新: 2026-03-24 00:15

---

## 最近更新（新的在上面）

### [2026-03-24 00:15] — 夜间自治：盲区诊断 → 数据增强 → 本地微调 → 91.3%

- **做了什么**（22:45 - 00:15 夜间自治循环）：
  1. **系统性盲区测试**：8 种文体 AI 文本，DeBERTa v1 只检出 2/8
  2. **颠覆性发现**：DeBERTa 学的是模型指纹(model memorization)，不是 AI 通用模式
  3. **RAID #1 对比**：下载 desklib 模型（RAID 排行榜第一），测试只有 36% 准确率——证明所有有监督检测器都有此问题
  4. **深度调研 8 篇论文**：DEFACTIFY、DivEye、SpecDetect(AAAI 2026)、DetectRL、DetectAnyLLM 等
  5. **数据增强**：DeepSeek 生成 83 条（14 新文体）+ RAID 提取 398 条（8 domains），合并 67,268 条训练数据
  6. **DivEye + SpecDetect 实现**：零样本辅助信号加入 perplexity.py
  7. **本地 M4 增量微调**：1,050 新数据 × 2 epochs × 12 分钟 → **49.8% → 91.3%（+41.4%）**
  8. **Fused score 改进**：PPL 低值覆盖规则

- **新增文件**：
  - `scripts/augment_dataset.py` — 数据增强（DeepSeek + RAID）
  - `scripts/prepare_training_data.py` — 数据合并 + noising
  - `scripts/finetune_local.py` — Apple M4 本地微调
  - `scripts/test_desklib.py` — 竞品对比
  - `models/detector_v2/` — 增量微调后模型
  - `docs/detector-improvement-plan.md` — 完整改进计划

- **踩过的坑**：
  - MPS OOM with batch=4 → batch=1 + max_len=256
  - gradient_checkpointing 和 DeBERTa-v3 不兼容
  - RAID 10M+ 条流式扫描极慢
  - SpecDetect DFT energy 在 llama3.2:1b 上方向不一致

- **当前状态**：detector_v2 在新域数据上 91.3%，但有假阳性问题（短口语文本）。v1 + v2 ensemble 待优化

### [2026-03-23 23:28] — DeBERTa 盲区诊断 + 数据增强启动（被上方更新取代）

- **架构决策**：
  - DeBERTa 重训策略三管齐下：数据扩充 + RAID 合并 + Data Noising
  - 训练方案参照 DEFACTIFY（sequential fine-tune + 60:40 ensemble）
  - DivEye 作为 LR 辅助信号，不替代 DeBERTa

- **踩过的坑**：
  - Playwright + Chrome 冲突（Chrome 运行时 Playwright 无法启动）→ kill Chrome 后发现 claude.ai session 已过期
  - Python 3.9 不支持 `str | None` type hint → 改为无注解函数签名
  - RAID 数据集 10M+ 条，流式扫描前 500K 全是 abstracts domain，需要遍历更多才能找到其他 domain

- **当前状态**：数据增强进行中
  - DeepSeek 生成：~46/700 条（后台运行）
  - RAID 提取：扫描中
  - 改进计划文档：已完成

- **下一步**：
  1. 等数据生成完 → 合并 + noising
  2. 上传 Colab 重训 DeBERTa
  3. 跨域/跨模型测试验证改进

### [2026-03-23 10:36] — DeBERTa 98.5% + 双系统检测 API 就绪

- **做了什么**：
  1. **DeBERTa 训练管线修复**：发现 3 个 bug 并修复
     - `load_best_model_at_end=True` + DeBERTa gamma/beta 命名不匹配 → 静默损坏模型权重（根因）
     - 按 VRAM 自动选 batch_size（A100-80GB=64, 40GB=32, T4=16）
     - 训练完成后同一 cell 自动保存 + 验证
  2. **数据集重建**（70,000 条）：
     - human 文本：从 C4 语料库连续窗口采样（修复随机拼接导致的不连贯问题）
     - human_polished：40% 本地 spaCy+WordNet 同义词替换（QuillBot 风格）+ 60% DeepSeek API
     - ai / ai_polished：保留原有
  3. **DeBERTa 4-epoch 训练**：Colab A100-40GB, bf16, batch=32, ~1hr
     - 130 样本测试准确率: **98.5%**（128/130）
     - 数据集 human/AI 各 50/50 满分，语料库 human 28/30
  4. **Perplexity 升级**：llama3.2:1b → MLX qwen3.5:4b（信号分离度 3.3x）
     - 5 特征 Logistic Regression: 90% 准确率，补 DeBERTa 正式文本盲区
  5. **API 三层信号**：DeBERTa 二分类 + Perplexity LR + 融合分数
  6. **DeBERTa 简化为二分类**：human vs AI，4-class 保留但暂不暴露

- **踩过的坑**：
  1. DeBERTa-v3 `legacy=True` 导致 LayerNorm 用 gamma/beta 命名，checkpoint reload 时 key 不匹配
  2. `torch.cuda.get_device_properties(0).total_memory`（不是 `total_mem`）
  3. `classifier.weight.std() ≈ 0.02` 不能判断"未训练" — DeBERTa 学习在 backbone 不在 head
  4. Colab Find/Replace 对已执行 cell 不生效 — 本地改完重新上传
  5. MLX qwen3.5:4b 长文本推理极慢（~4min/3000字）— 生产环境需要截断或优化
  6. `mx → numpy` 转换需要 `.astype(mx.float32)` 避免 PEP 3118 buffer 格式错误

- **当前状态**：可用
  - 检测 API: `python3.13 scripts/perplexity.py` (port 5001)
  - 返回: `classification` (DeBERTa) + `perplexity_stats` (LR) + `fused` (融合) + `tokens` (可视化)

### [2026-03-22 10:00] — CoPA Humanizer 原型 + 校准检测器

- **做了什么**：
  1. **深度调研**：AI detector/humanizer 生态全景（论文 15+，产品 10+，基准 3 个）
  2. **CoPA 实现**：基于 EMNLP 2025 论文实现对比式解码 humanizer
     - v1 (`copa_mlx.py`): 基础实现，qwen3.5:4b via MLX
     - v1 参数扫描 (`copa_sweep.py`): 5λ × 3α × 3T = 45 组参数
     - v2 (`copa_proof.py`): 修复 5 个关键问题后的版本
  3. **校准检测器** (`calibrate_detector.py`): 用 dataset.jsonl 训练 logistic regression，校准 perplexity 检测器阈值（进行中）

- **架构决策**：
  - **MLX 而非 llama-cpp-python**：llama.cpp 不认识 `qwen3_5` 架构，MLX 0.31.1 支持。Python 3.13 (/opt/anaconda3) 运行 MLX，Python 3.9 跑不了
  - **CoPA 而非 LoRA 微调**：CoPA 无需训练，直接在解码时用双 prompt 对比。论文报告 87.88% 检测器逃逸率。选它因为：(1) 零训练成本 (2) 能适应检测器更新 (3) 硬件约束下可行
  - **Best-of-N 而非逐 token 精确控制**：用户最初提出"按人类 perplexity 分布生成"，但逐 token 控制不现实（perplexity 是上下文依赖的）。改为生成 20 候选用复合评分选最优
  - **Logistic Regression 校准**：5 特征（PPL/ENT/ENT_STD/BURST/GLTR）→ LR 分类器。可解释性比 XGBoost/SVM 好，系数直接指导 CoPA 调参

- **踩过的坑**：
  1. **llama-cpp-python 不支持 qwen3.5**：`unknown model architecture: 'qwen35'`，PyPI 和 GitHub main 都是 v0.3.16，底层 C 代码未更新
  2. **MLX vocab size 不匹配**：tokenizer.vocab_size=248044 但模型 embedding=248320，英文掩码维度不对。修复：用 `model(dummy).shape[-1]` 获取真实维度
  3. **Python 3.9 装不了 mlx-lm 0.31.1**：需要 mlx>=0.30.4，但 Python 3.9 上 PyPI 最高 0.29.3。必须用 Python 3.13
  4. **v1 输出 43/45 以 "Honestly" 开头**：单一 prompt 模板导致模式锁定。v2 用 5 个随机模板修复
  5. **qwen3.5 输出中文**：高 λ + 高 T 时模型掉入 CJK token 空间。v2 加英文词表掩码（blocked 99052/248320 = 39.9%）
  6. **GPU Timeout**：全精度 qwen3.5:4b 在 16GB M4 上 OOM。改用 4-bit 量化版

- **CoPA v2 实验结果**（3 篇测试文本，每篇 20 候选）：

  | 文本 | 原始 PPL | 最佳 CoPA PPL | 原始 GLTR | 最佳 CoPA GLTR | 语义保持 |
  |------|---------|--------------|----------|---------------|---------|
  | academic | 4.3 | 35.8 | 95% | 66% | 0.673 |
  | blog | 3.8 | 28.2 | 92% | 71% | 0.706 |
  | technical | 3.9 | 35.4 | 98% | 64% | 0.679 |

  最佳参数：λ=0.5, α=1e-5, T=1.1。三篇文本的 top-1 候选全部四项指标落入人类范围 [PEBG]。

- **当前状态**：CoPA 原型可用，校准检测器正在跑
- **下一步**：
  1. 校准结果出来后验证 CoPA 输出是否能骗过校准后的检测器
  2. 如果通过 → 集成到产品（humanizer.py 新增 CoPA 方法）
  3. 如果不通过 → 分析哪个特征暴露了，针对性调 CoPA 参数
  4. DeBERTa 重训完成后用它做最终验证

- **已知局限（必须诚实记录）**：
  1. **自评偏差**：所有指标用 qwen3.5 自己算，不等于外部检测器的判断
  2. **语义保持 0.67-0.71**：改写后保留约 70% 语义，可能不够精确改写场景
  3. **速度**：20 候选 ~3 分钟，生产环境需减到 3-5 候选（~20s）
  4. **emoji 泄漏**：部分 prompt 模板导致输出含 emoji/markdown 符号
  5. **首词多样性不足**：academic 55% 以 "AI" 开头，仍有模式可被检测

### [2026-03-22 09:00] — Humanizer 方向性讨论 + 技术研究

- **做了什么**：与用户讨论 humanizer 产品方向和技术路线
- **用户核心理念**：
  - AI 是帮助高效传达想法的工具，不是替代思考
  - 用 AI 写 code 被推崇，写 essay 被抵触 — 这是双标
  - 远期愿景：个性化 translator（采集用户表达习惯，生成匹配个人风格的文本）
  - Humanizer 的输出不是最终产品，是"统计模具"——用人类文本的 perplexity 分布作为 seed，让 LLM 在这个约束下保持原文语义
- **技术研究结论**：
  - 现有 humanizer 全部做表面文章（同义词替换、句式变换），GPTZero 已能识别
  - DIPPER (11B) 是学术 SOTA 但太大；CoPA (EMNLP 2025) 无需训练、用对比解码
  - GradEscape 用 139M 模型超过 11B DIPPER — 模型大小不是决定因素
  - 理论极限：Sadasivan et al. 证明充分好的语言模型输出不可靠检测

### [2026-03-21 23:30] — Landing Page 重做 + 字体修复

- **做了什么**：重写产品首页为 premium landing page。Hero 区有 X-Ray 扫描动画（文字逐词变色显示 AI 概率）。工具卡片、How It Works 深色区、数字滚动动画、差异化区、CTA。工具界面从 `/` 移到 `/app`。移除 Instrument Serif（用户不喜欢），改用 Geist Sans 300。
- **架构决策**：`/` = 静态 landing（SSR，SEO 友好），`/app` = client-side 工具（AppShell）。用 `Instrument_Serif` 做 display font 被否决，教训：字体选择必须先调研 3+ 个参考产品。
- **踩过的坑**：Instrument Serif 太粗。端口 3000 被其他项目占用，ai-text-detector 在 3003。
- **当前状态**：Landing page 可用 http://localhost:3003。
- **下一步**：Learn 页面、更多 Blog、credibility 数据。字体暂用 Geist Sans，定制字体搁置。

### [2026-03-21 22:00] — Writing Center 对话式 Onboarding

- **做了什么**：重写 WritingCenter.tsx，三阶段状态机 `welcome → conversation → writing`。Welcome = starter 卡片页（类似 claude.ai）；Conversation = 纯 AI 对话（无编辑器）；Writing = 完整编辑器 + 协作面板。
- **架构决策**：`phase` state 控制视图切换。有已保存 draft 的用户直接跳 writing phase。
- **踩过的坑**：最初设计平铺所有功能导致信息过载。参考 claude.ai 后改为渐进展开。
- **当前状态**：代码完成，未做完整 smoke test。
- **下一步**：用户描述了更深的流程（AI 搜索研究 → brainstorm → AI 生成草稿），需要 web search API。

### [2026-03-21 20:00] — Writing Center MVP-1 全部组件完成

- **做了什么**：
  - Phase 0: 12 篇测试文章 + 7 个 role prompts + 35 daily tips + 10 lab examples
  - Prompt calibration: 2 轮，88.9% 通过率（128/144）
  - Phase 1: `/api/writing-assist` route（7 actions, DeepSeek V3）
  - Phase 2: Tiptap Editor + ChatPanel + 4 子组件 + 全部 AI 联动 + 双向 annotation 链接 + streak
  - Phase 3: LabPanel + 集成
- **架构决策**：
  - DeepSeek V3 替代 Claude API（成本低 10-35x）
  - Temperature 按 action 分层：analyze=0, dialogue=0.7, lab-rewrite=per-request
  - Tiptap inline Decoration（不是 gutter，gutter 太复杂留 Phase 2）
  - localStorage 全部状态（兼容 MVP-2 Supabase 迁移）
  - Liz Lerman 顺序：good → question → suggestion → issue
  - Conventions 压制：Ideas/Organization 有 issue 时不显示语法批注
  - SRSD 脚手架递减：genreExperience 每 session 只 +1
- **踩过的坑**：
  1. Tiptap + Next.js：必须 `immediatelyRender: false`
  2. DeepSeek temperature 映射：API temp × 0.3 = 实际 temp
  3. Calibration 第一轮 79.9%：good annotations 不引用原文 → 加 WRONG/RIGHT 示例；分数波动 → 取整到 5 + 放宽 ±10；Conventions 压制不一致 → 显式 4 步流程
  4. `@anthropic-ai/sdk` 装了没用（改用 `openai` 包调 DeepSeek）
  5. 多 subagent 并行改同一文件会覆盖
- **当前状态**：build 通过，未 smoke test。
- **下一步**：浏览器完整走一遍流程修 bug。

### [2026-03-21 04:00] — 50M 语料库构建完成

- **做了什么**：Colab A100 构建 50M 句子语料库，FAISS IVF+PQ 索引。chunk-streaming 构建（不用 memmap）。SentenceStore 字节偏移索引（400MB 内存 vs 8GB）。
- **数据源配置**：C4 30% (28M) / Wikipedia 20% (2.8M，不够 10M 目标，C4 backfill 补) / CC-News 20% (5.3M, 排除 2019+ 防 AI 污染) / CNN-DailyMail 15% (5.7M) / Gutenberg 15% (7.5M)。全部 pre-2019 无 AI 污染。
- **构建脚本**：`scripts/build_corpus_colab.py`。CRC32 双哈希去重。每 1M 句子一个 chunk（.npy + .jsonl）。崩溃恢复靠 hash 集重建。
- **humanizer 内存优化**：`SentenceStore` 类（humanizer.py:35-83），用 numpy 偏移数组替代 Python list。首次启动扫描 JSONL 建偏移索引缓存为 `.offsets.npy`（~381MB），后续秒加载。单文件句柄不是线程安全——当前单线程 HTTP server 无影响。
- **踩过的坑**（13 个，按时间顺序）：
  1. `torch.cuda.get_device_properties(0).total_mem` → 改 `.total_memory`
  2. faiss pip install 被注释 → 改 `subprocess.check_call` 自动安装
  3. CUDA 12.8 需要 `faiss-gpu-cu12`（不是 `faiss-gpu`）
  4. BookCorpus `trust_remote_code` 废弃 → 换 C4
  5. BookCorpus `Dataset scripts no longer supported` → 同上
  6. PubMed 嵌套字典取值脆弱 → 换 CNN/DailyMail
  7. 30M→50M 后 `np.concatenate` 72GB 峰值 → 改 memmap（后又改为 chunk-streaming）
  8. 崩溃恢复 `skip_remaining` 逻辑 bug → 删 skip 逻辑，纯靠 hash 去重
  9. MD5 hash 性能 → 改 CRC32 双哈希
  10. memmap 训练采样随机 I/O 极慢 → `np.sort()` 排序后顺序读
  11. C4 backfill 无限循环 → `backfill_failures` 计数器，3 次后终止
  12. Colab 断连丢 chunk → 改挂载 Google Drive
  13. memmap 写 Drive FUSE → FileNotFoundError → 改 chunk-streaming（不用 memmap）
  14. Colab 磁盘满 → DriveFS lost_and_found 缓存 144GB，手动清理
  15. faiss-gpu 安装后 `get_num_gpus()=0` → 需重启 runtime（C 扩展缓存）
- **当前状态**：`corpus/sentences.faiss` (2.7GB) + `sentences.jsonl` (6.2GB) 已部署。humanizer.py 加载正常（50M 条目，381MB 偏移索引）。
- **下一步**：不需要再动。如需重建：Colab 脚本在 Google Drive `/content/drive/MyDrive/corpus_build/` 有原始数据。

---

## 当前状态：数据集重建中

DeBERTa 分类器尚未可用。训练管线已修复，等待数据集重建后重新训练。

## 阻塞项

### 数据集质量缺陷（正在修复）
`generate_dataset.py` 的 `load_human_texts` 将语料库句子随机打乱后拼接，导致 human 样本（label=0）不连贯——法语翻译混八卦、体育混政策。模型学到"连贯=AI，不连贯=human"，在真实文本上完全失效。

**已修复代码**：`load_human_texts` 改为连续窗口采样，保持文章内连贯性。

**待执行**：用修复后的函数重新生成 human（17,500 条）和 human_polished（17,500 条），保留 ai + ai_polished（35,000 条），重建 `dataset.jsonl`。

### 训练管线缺陷（已修复）
1. **DeBERTa gamma/beta bug**：`load_best_model_at_end=True` 加载 checkpoint 时 LayerNorm 命名不匹配（`.gamma/.beta` vs `.weight/.bias`），静默丢失所有 LayerNorm 权重。**修复**：`load_best_model_at_end=False`，`save_strategy='no'`。
2. **权重丢失**：cell-6 可被重跑覆盖 model 变量，cell-10 保存 base 权重。**修复**：训练完成后在同一 cell 立刻自动保存 + 验证 classifier std。

## 各模块状态

| 模块 | 状态 | 说明 |
|------|------|------|
| 前端 UI | ✅ 可用 | 检测/改写/写作三面板，GPTZero 风格可视化已加 |
| Python 检测后端 | ✅ 代码就绪 | llama.cpp token 分析 + DeBERTa 推理，等模型权重 |
| Python 改写后端（FAISS） | ✅ 可用 | 7 种语义改写方法，语义保持差但可做统计模具 |
| **CoPA Humanizer** | **🔧 原型** | **对比式解码，四项指标进入人类范围，待外部验证** |
| **校准检测器** | **🔧 进行中** | **LR 分类器，用 dataset.jsonl 校准 PPL/ENT/BURST/GLTR 阈值** |
| 50M 句子语料库 | ✅ 完成 | FAISS IVF+PQ 索引 2.6GB |
| 70K 训练数据集 | ❌ 重建中 | human + human_polished 不连贯，另一个 AI 在重新生成 |
| DeBERTa 分类器 | ❌ 重训中 | 等数据集重建完成后 Colab 重训 |
| Colab 训练 notebook | ✅ 已修复 | gamma/beta bug + 自动保存 + 验证 |
| 部署 | ❌ 无 | 无 Docker/Vercel 配置 |

## 下一步（按顺序）

1. 重建数据集：重新生成 human + human_polished（用 DeepSeek + Qwen API）
2. 上传到 Google Drive，Colab 训练 DeBERTa（~30 分钟 A100）
3. 下载训练好的模型到 models/detector/，验证可用
4. 实现 X-Ray Vision（token 级热力图）— 核心差异化功能
5. 部署方案

## 技术栈
Next.js 16 + TypeScript + Tailwind 4 + Recharts — Python: DeBERTa + llama.cpp + FAISS + spaCy

## 快速上手
```bash
npm install && npm run dev          # 前端 → localhost:3000
ollama pull llama3.2:1b
python3 scripts/perplexity.py       # 检测后端 → localhost:5001
python3 scripts/humanizer.py        # 改写后端 → localhost:5002
```

## 关键决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-03-22 | CoPA 对比解码做 humanizer | 无需训练，用双 prompt 对比在解码时去 AI 指纹。论文报告 87.88% 逃逸率 |
| 2026-03-22 | MLX 替代 llama-cpp-python | llama.cpp 不支持 qwen3_5 架构，MLX 0.31.1 原生支持 |
| 2026-03-22 | Best-of-N 选择而非逐 token 控制 | perplexity 是上下文依赖的，无法逐 token 精确控制。生成 N 个候选用复合评分选最优更实际 |
| 2026-03-22 | LR 校准检测器 | DeBERTa 在重训，用 LR 在 5 维特征上做临时检测器。可解释性好，系数直接指导 CoPA 调参 |
| 2026-03-21 | 关闭 load_best_model_at_end | DeBERTa-v3 gamma/beta 命名 bug 导致 checkpoint 加载静默损坏模型 |
| 2026-03-21 | human 文本改用连续窗口采样 | 随机拼接导致模型学习"连贯性"而非"AI 特征" |
| 2026-03-21 | 训练后同 cell 自动保存 + 验证 | 防止 cell-6 重跑覆盖训练结果 |

## 已知问题
- `.env.local` 含真实 API key，上线前必须轮换
- Git remote URL 中嵌了 GitHub token
- 10k 字符限制对长文检测是瓶颈
- 项目总体积 ~14GB（语料库 + 模型）
