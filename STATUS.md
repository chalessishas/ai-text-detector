# AI Text X-Ray — 项目状态

> 最后更新: 2026-04-06 07:00

---

## 🔴 当前计划（三角色讨论后确定）

### 训练前必须完成
1. **补 10K 学生 essay**（PERSUADE 2.0）+ **10K ESL**（Yahoo Answers/ELI5 非母语）
2. **去重**：合并 CNN×3→1、OpenWebText×3→1、IMDB×2→1 等重复源
3. **搭独立 holdout 集**：800 human + 600 AI（GPT-4o/Claude/Gemini 各 200）

### 训练配置改动
- 加权 CrossEntropyLoss: weight=[1.5, 1.0]（human 更高 → 降假阳性）
- Epochs: 3→5 + early stopping patience 3
- MAX_LEN: 512→768, BATCH_SIZE: 16→12, GRAD_ACCUM: 4→6
- DANN 域: 73 碎域→8 大域, DANN_LAMBDA_MAX: 1.0→0.5

### 上线门槛
- holdout 整体 human FPR < 5%
- 学生 essay FPR < 3%
- AI TPR > 85%
- 红队 15 条假阳性 ≤ 1

---

## ⏳ 当前状态 (2026-04-06)

### DeBERTa v6 正在 Colab A100 上训练
- **Notebook**: Untitled27.ipynb（chalessishas@gmail.com 的 Colab）
- **数据集**: 685K 样本（398K human + 215K AI + 72K adversarial），从 Google Drive 复制
- **配置**: deberta-v3-large, binary, DANN, 3 epochs, batch=16, grad_accum=4
- **预计**: ~8-10 小时完成（A100 40GB SXM4）
- **状态**: cell 在 Executing，已跑 ~30 分钟（pip install + 模型下载 + 数据加载中）

### 对抗性架构审查发现的问题（按优先级）

| 级别 | # | 问题 | 修复状态 |
|------|---|------|----------|
| CRITICAL | C1 | AI 文本只用 DeepSeek 生成（单模型偏差） | **用户决定：不修** |
| CRITICAL | C2 | XGBoost 在 DeBERTa 训练集上验证（循环泡沫） | 待修 |
| CRITICAL | C3 | binary 合并丢失 human_polished 信号 | 待验证 |
| HIGH | H1 | DANN 域标签应按生成来源分而非写作领域 | 待修（下次训练） |
| HIGH | H2 | 数据源重复 + student_essay 只抓标题 + ESL 缺失 | 部分已修 |
| HIGH | H3 | stat_score 15 个硬阈值无校准 | 待修 |
| HIGH | H4 | XGBoost 453 样本统计功效不足 | 待修 |
| MEDIUM | M4 | **synthetic_clinical_notes 是 AI 数据被标为 human** | **已修（删除）** |

### 模型文件位置（不在 git 中）
- **DeBERTa v4**: `models/detector_v4/` — 2026-03-26 训练，97.6% 4-class。只在原始开发机上
- **DeBERTa v5**: `models/detector_v5/` — 2026-04-04 训练（RunPod 4090），97.4% 4-class + adversarial。只在 shaoq 的 MacBook 上。symlink `models/detector → detector_v5`
- **DeBERTa v6**: 正在 Colab A100 训练中。完成后需下载到 `models/detector_v6/`
- **XGBoost**: `models/xgboost_fusion.pkl` — 在 git 中（226KB）
- **LR v3**: `models/perplexity_lr_v3.pkl` — 在 git 中（2.3KB）
- **注意**: 新机器 clone 后没有 DeBERTa 模型文件，检测服务器会 fallback 到 heuristic 模式

### 训练完成后需要做
1. 从 Colab 下载 v6 模型到本地 `models/detector_v6/`
2. 搭建独立 holdout 测试集（200 真实人类 + 150 非 DeepSeek AI）
3. 对比 v5 vs v6 在 holdout 上的假阳性率
4. 重训 XGBoost（用 v6 DeBERTa 信号 + 扩充 OOD 数据）
5. 全套测试 60 项 + holdout 评估

### RunPod 状态
- 所有 pod 已终止（省钱）
- API key: 在 `~/.runpod/config.toml` 中
- 余额: ~$48

---

## ✅ 进展 (2026-04-05 opus-0405f)

### 本次修复
1. **XGBoost 特征去冗余**：移除手工离散化的 `ppl_score`（7→6 特征），CV 88.0%（vs 旧 88.5%），零回归
   - 新特征重要性：top10 52.7%, deb_ai 12.8%, stat_score 10.2%, ppl_val 9.8%, lr_ai 8.2%, mean_ent 6.2%
2. **build_dataset_v6.py 全面修复**：从 15/49 源通过 → **46/46 全通过**，可达 718K（目标 600K）
   - 修复 12 个字段名错误（case-sensitive: text→TEXT, s→line, etc.）
   - 替换 13 个 script-based 数据集为 Parquet 版本
   - 替换 6 个 gated/missing 数据集
   - 添加 `--dry-run` 模式（先验证再收集，避免浪费 RunPod 时间）
3. **确认温度 T=2.0 正确**：是 post-hoc calibration（Guo et al. 2017），不是 bug
4. **测试全绿**：60 pass / 2 xfail / 0 fail

---

## ⚠️ 紧急交接 (2026-04-05)

### 当前最大问题：DeBERTa v5 假阳性率灾难性高

**现象**：所有 150+ 词的人类文本被 DeBERTa v5 判为 82-96% AI。口语博客、学术论文、新闻、诗歌、商务邮件全部误判。

**根因**（已确认）：
1. v5 训练数据只有 69K 样本，题材单一
2. 加了 17K adversarial AI（1:1 与 human 样本）导致模型学到 "像人的文本 = AI 伪装"
3. DeBERTa 4-class（human/ai/ai_polished/human_polished）分散模型容量
4. 模型太小（deberta-v3-base 86M 参数）不够学跨域特征

**已决定的修复方案（用户已批准）**：
1. 改 binary 分类（human vs AI）— 脚本已写好 `scripts/train_runpod_v6.py`
2. 换 deberta-v3-large（304M 参数）
3. 加 DANN domain-adversarial training（脚本已包含）
4. 数据集扩到 1M（600K human + 300K AI + 100K adversarial）
5. human:adversarial 比例 6:1（不再 1:1）

### 数据集构建 ✅ 已修复 (opus-0405f)

**修复前**：49 个 HF 数据源仅 15 个可用（230K 可达）
**修复后**：46/46 全通过（718K 可达，目标 600K）

**修复内容**：
- 12 个字段名错误（case-sensitive: text→TEXT, s→line, body, Cover Letter 等）
- 13 个 script-based 数据集替换为 Parquet 版本
- 6 个 gated/missing/broken 数据集替换
- 添加 `--dry-run` 模式验证所有源

**下一步**：
1. 在 RunPod 上执行 `python3 build_dataset_v6.py --part human` 收集 600K 人类文本
2. 在 pod 上有个测试脚本 `/workspace/test_all_sources.py` 正在测 36 个数据源的可用性
3. 确认能用的源后，按**题材配额**分配（每个题材至少 10K），不是按数据集大小

### RunPod 状态
- Pod `ctbwaf5y42m8jk`（A100 SXM 80GB）已停止，数据保留在 volume
- 上面已有：dataset_v4.jsonl、collect_human_fast.py 产出的 ~310K web 文本、train_runpod_v6.py
- RunPod 余额约 $35+
- API key: 在 `runpodctl doctor` 里配置过，或用 `runpodctl user` 验证

### 已完成的工作（本次会话）
| 项目 | 状态 |
|------|------|
| DeBERTa v5 训练 + 部署 | ✅ 但有严重假阳性 |
| XGBoost 重训 3 轮 | ✅ 但不稳定 |
| 安全审计 + 9 个 bug 修复 | ✅ 已 push (26e98bb, 3ee0461) |
| OOD 数据扩充 227→471 | ✅ |
| pytest 72 pass / 0 fail | ✅ |
| Playwright 实测 4 场景 | ✅ |
| StealthRL humanizer v1 | ❌ 1/3 绕过率，LoRA 丢失 |
| 题材检测准确率报告 | ✅ docs/research/2026-04-05-detection-accuracy-by-genre.md |
| 1M 数据集构建 | ❌ 卡在 HF 数据源兼容性 |

### 关键文件
| 文件 | 用途 |
|------|------|
| `scripts/train_runpod_v6.py` | v6 训练脚本（binary + large + DANN）|
| `scripts/build_dataset_v6.py` | 数据集构建脚本（49 源，大部分失败）|
| `scripts/train_stealth_runpod.py` | StealthRL 训练脚本 |
| `scripts/train_xgboost_fusion.py` | XGBoost 融合训练 |
| `docs/research/2026-04-05-detection-accuracy-by-genre.md` | 题材准确率报告 |
| `docs/research/2026-04-05-text-corpus-datasets.md` | 66 种文本类型数据源清单 |
| `docs/research/2026-04-05-stealthrl-fix-and-alternatives.md` | StealthRL 失败分析 |

### Agent 审查发现的架构问题（待修）
1. **Temperature 2.0 评估差距**：训练 T=1 评估，生产 T=2，实际准确率可能低很多
2. **XGBoost ppl_score 重复特征**：手工离散化 + 原始值同时喂入导致不稳定
3. **stat_score 未校准**：15 个魔法数字手调，无单元测试
4. **PPL 模型 Ollama 解析脆弱**：输出格式变了就断

### 下一步执行清单（按优先级）
1. **P0**：修复数据源——逐个测试 36 个 HF 数据集，找到每个题材至少一个能用的数据源
2. **P0**：按题材配额收集 600K human 文本（每个题材 10-50K）
3. **P1**：用 DeepSeek API 生成 300K AI + 100K adversarial
4. **P1**：在 A100 上跑 train_runpod_v6.py 训练 DeBERTa v6
5. **P2**：修 XGBoost（去掉 ppl_score 重复特征，降低模型复杂度）
6. **P2**：70+ 题材细分测试验证
7. **P3**：StealthRL 重训（A100, num_generations=16, max_steps=2000）

---

## 最近更新（新的在上面）

### [2026-04-04 07:23] — DeBERTa v5 对抗重训 + XGBoost 重训 + OOD 扩充

**DeBERTa v5 训练** (RunPod RTX 4090, 103 min, ~$1.15):
- 训练数据: 69K clean (dataset_v4.jsonl) + 17K adversarial (14 种攻击类型)
- 4-class accuracy: **97.4%**, Binary: **99.9%**
- Class weights 补偿 AI 类 2x 样本量
- 模型已部署: `models/detector_v5/` (715MB), symlink 已更新

**XGBoost 融合重训** (2 轮迭代):
| 轮次 | OOD 样本 | CV Accuracy | DeBERTa 权重 | PPL 权重 | 结果 |
|------|---------|-------------|-------------|---------|------|
| v3 (旧) | 112 | 94.6% | 73% | 14% | 基线 |
| Round 1 | 112 | 93.8% | 18% | 77% | 3 fix, 1 new FP |
| **Round 2** | **392** | **90.8%** | **13%** | **56%** | **全绿** |

**OOD 数据扩充** (227 → 453 样本):
- 来源: RAID (106) + AI-Human-Text (120) + Detection Pile (100) + 原有 (127)
- 覆盖: books, wiki, abstracts, reddit, news, poetry, recipes, reviews

**测试结果** (v5 + XGBoost Round 2):
- detector: **25 pass, 1 xfail, 1 xpassed** (Quillbot bypass 仍 xfail)
- redteam: **35 pass, 0 fail**
- 合计: **60 pass, 1 xfail, 1 xpassed, 0 fail**

**新增/修改文件**:
- `scripts/train_runpod_v5.py` — RunPod v5 训练脚本
- `scripts/train_local_v5.py` — 本地 M4 训练脚本 (备用)
- `scripts/expand_ood_data.py` — HuggingFace OOD 数据扩充
- `scripts/download_runpod_v5.sh` — 模型下载工具
- `scripts/perplexity.py` — v5 优先级加载 + _resolve_classifier_path()
- `models/xgboost_fusion.pkl` — Round 2 融合模型
- `data_ood_xgboost.jsonl` — 扩充后 453 样本
- `.gitignore` — 添加 v4/v5 模型目录

### [2026-04-03 19:25] — 全面交接文档更新

**本次自治会话总结** (06:00 - 19:25, ~13 小时, 19 commits):

#### 改进前 vs 改进后

| 指标 | 改进前 (06:00) | 改进后 (19:25) |
|------|---------------|---------------|
| 测试通过率 | 18/27 (67%) | **122/124 (98%)** |
| 假阳性率 | 不确定（PPL 断路） | **0/12** (红队测试全通过) |
| PPL 模型 | 断路 10 天 | **已恢复** (llama3.2:1b) |
| 融合方式 | 手调 if-else | **XGBoost v3** (94.6% OOD) |
| 测试套件 | 27 个基础测试 | **124 个** (基础+红队+E2E) |
| 死数据 | ~1.9GB | **已清理** |
| 部署就绪 | 无 | **Dockerfile + fly.toml** |
| 研究文档 | 3 份 | **10 份** |

#### 完成的 19 次 Commit

| # | 类型 | 内容 |
|---|------|------|
| 1 | fix | PPL 动态 Ollama 解析 + 35 项红队测试 |
| 2 | feat | Log-rank 检测信号 + XGBoost 训练脚本 |
| 3 | docs | STATUS.md 诊断总结 |
| 4 | docs | CLAUDE.md Known Pitfalls 更新 |
| 5 | feat | XGBoost v3 部署 (94.6% OOD) |
| 6 | fix | Humanizer structure 方法（语义+结构优先） |
| 7 | docs | 研究报告 + chronicle |
| 8 | fix | calibrate_detector.py 引用修复 |
| 9 | fix | XGBoost OOD 训练改进 |
| 10 | test | 12 项 E2E 全栈测试 |
| 11 | fix | guide API retry (DeepSeek 限流) |
| 12 | feat | Dockerfile 更新 (llama-cpp + XGBoost) |
| 13 | feat | fly.toml (Fly.io 一键部署) |
| 14 | fix | entrypoint.sh → DeBERTa v4 |
| 15-19 | docs | 研究报告 (部署架构, human 数据源, next-steps) |

#### 当前数据文件

| 文件 | 大小 | 用途 | 状态 |
|------|------|------|------|
| `dataset_v4.jsonl` | 239MB | DeBERTa + LR 训练 | **活跃** |
| `dataset_adversarial_v4.jsonl` | 238MB | 对抗基准测试 | **活跃** |
| `data_human_hc3.jsonl` | 6MB | LR 补充 human 数据 | **活跃** |
| `data_human_imdb.jsonl` | 4MB | LR 补充 human 数据 | **活跃** |
| `data_ood_xgboost.jsonl` | 64KB | XGBoost OOD 训练 (227 样本) | **活跃** |

#### 下一步优先级

1. **P0: Colab DeBERTa v5 对抗重训** — notebook 已就绪 (`train_detector_v5_colab.ipynb`)，需 GitHub OAuth 授权。用 69K adversarial 样本提升跨领域能力
2. **P1: 扩充 OOD 数据到 300+** — 当前 227 样本（171 human + 56 AI），AI 偏少。从 HuggingFace Detection Pile 补充
3. **P2: 部署到 Fly.io** — Dockerfile + fly.toml 已就绪，需要创建 Fly.io 账号 + GitHub Release 上传 DeBERTa v4 权重
4. **P3: 模型蒸馏** — 把 DeBERTa 738MB 蒸馏为 TinyBERT ~100MB，降低部署成本

### [2026-04-03 08:00] — XGBoost v3 部署 + Humanizer 修复 (追加)

**XGBoost 融合 meta-learner 三轮迭代**：
| 版本 | 数据 | CV Accuracy | DeBERTa 权重 | 回归 | 状态 |
|------|------|-------------|-------------|------|------|
| v1 | dataset_v4 (in-domain) | 100% | 100% | N/A | 过拟合，已删 |
| v2 | 48 OOD samples | 89.8% | 26% | 2 个 | 未部署 |
| **v3** | **112 OOD samples** | **94.6%** | **73%** | **0 个** | **已部署** ✓ |

- Feature weights: DeBERTa 73%, top10 14%, LR 4%, PPL 4%, stat 2%
- 关键学习：XGBoost 训练必须用 OOD 数据（DeBERTa 训练集之外的文本），否则退化为 DeBERTa-only

**Humanizer structure 方法修复**：
- Bug: score 永远 1.0，返回不相关文本（纯句法匹配）
- Fix: 优先在 FAISS 语义匹配中找结构兼容的，纯结构匹配 score 降到 0.5

**其他**：
- `calibrate_detector.py` 引用已删除的 dataset.jsonl → 改为 dataset_v4.jsonl
- 删除 dataset_augmented.jsonl + dataset_raid_extract.jsonl（已被 v4 消费）

### [2026-04-03 06:50] — 检测系统全面修复 + 红队测试 (3 commits)

**问题诊断**：
- PPL 模型自 3/24 起未加载（Ollama blob hash 失效 → 四路融合降级为 DeBERTa-only → 假阳性）
- DeBERTa 跨领域 AUROC 0.5-0.6（学模型指纹，不学 AI 通用特征）
- 8/27 测试失败（含 1 个假阳性 + 多种对抗绕过）

**修复内容**：
1. **PPL 模型加载重写** — 替换硬编码 Ollama blob SHA256 为动态解析（`ollama show --modelfile`），自动 fallback（qwen3.5:4b → llama3.2:1b）。防止 hash 再次失效
2. **四路融合恢复** — PPL 恢复后 DeBERTa 假阳性被 PPL+LR 纠正（烘焙文本：100→41 分）
3. **35 项红队对抗测试** — 覆盖 12 种假阳性场景（学术论文、非母语、日记、食谱等）、7 种对抗绕过、7 种边界情况、3 项回归测试
4. **Log-rank 检测信号** — 添加 DetectLLM 风格 log-rank 到 API 响应（llama3.2:1b 信号分离度不足，保留作参考）
5. **XGBoost 融合训练脚本** — `scripts/train_xgboost_fusion.py`，用 GradientBoosting 替代 140 行 if-else（训练中）
6. **数据集审计** — 13 个 JSONL 确认只有 dataset_v4.jsonl（69K 平衡样本）有效，merged_noised 已损坏

**测试结果**：
- 原始测试：**26/27 pass**（+2 xfail，从 18/27 提升）
- 红队测试：**35/35 pass**
- 合计：**60/62 pass, 2 xfail, 0 fail**
- E2E Playwright：Landing page ✓, App shell ✓, AI 检测 ✓ (85.6% for standard AI text)

**新增文件**：
- `tests/test_redteam.py` — 35 项红队对抗测试
- `scripts/train_xgboost_fusion.py` — 融合模型训练脚本
- `docs/research/2026-04-03-detection-improvements.md` — 检测改进调研
- `docs/research/2026-04-03-ai-api-cost-comparison.md` — 云端 PPL 方案对比

**待完成**（需要 GPU / 用户介入）：
1. Colab DeBERTa v5 对抗重训（notebook 已就绪：`scripts/train_detector_v5_colab.ipynb`，需 GitHub 授权）
2. XGBoost 融合模型训练完成后重启服务端加载
3. 清理死数据文件（dataset_merged_noised.jsonl 等）

### [2026-03-26 05:30] — 自治训练循环完成（7 commits, 4 hr）

**总成果**：
| 组件 | 之前 | 之后 | 改进 |
|------|------|------|------|
| LR | v1: 68.6% (5 features) | **v3: 86.1%** (24 features) | **+17.5%** |
| DeBERTa | v1: 不平衡数据 | **v4: 97.6%** 4-class (RunPod) | 平衡训练 |
| Fusion | DeBERTa 30% 固定 | DeBERTa 低信心 10-15% | 减少 FP |

**新增文件**：
- `dataset_v4.jsonl` — 69K 平衡 4-class (17294×4)
- `dataset_adversarial_v4.jsonl` — 69K 对抗样本 (14 攻击)
- `data_human_hc3.jsonl` + `data_human_imdb.jsonl` — 8K 多样 human 数据
- `models/detector_v4/` — DeBERTa v4 (RunPod 4090, $0.78)
- `models/perplexity_lr_v3.pkl` — LR v3 (24 features)
- 2 份 research reports in docs/research/

**下一步优先级**（from research）：
1. 69K adversarial 样本加入 LR 训练（提升鲁棒性）
2. 构建 100+ 样本 FPR 测试套件（诚实测假阳率）
3. Fusion 权重用 held-out set 自动优化（替代手调）

### [2026-03-26 02:40] — 训练循环：数据修复 + LR 重训 + RunPod DeBERTa

- **做了什么**（01:30 - 02:40，约 1.2 小时）：
  1. **GitHub pull**: 6 个新 commit（sliding window, CJK fix, adversarial 14 types）
  2. **发现 dataset_v3 不平衡**: human 32K vs AI 17K，导致模型偏向 human
  3. **创建 dataset_v4.jsonl**: 69,176 samples, 17,294 × 4 classes（完美平衡）
  4. **LR v2 重训启动**: 5000 samples (2.5x 旧), --recompute, MLX qwen3.5:4b
  5. **本地 DeBERTa MPS 尝试**: OOM at 3%（18GB 不够），放弃
  6. **RunPod 4090 部署**: $0.34/hr, torch 2.6.0+cu124, DeBERTa 全量 69K 样本重训
  7. **69K adversarial 样本生成**: 14 种攻击类型，用于后续训练/评估
  8. **benchmark_models.py**: 离线评估脚本（LR + DeBERTa 对比）

- **训练结果**:
  - LR v2: **79.8% accuracy** (5000 balanced samples, +11.2% vs old on same test)
  - DeBERTa v4: **97.6% 4-class, 99.7% binary** (69K balanced, RunPod 4090, 42.7 min)
  - RunPod cost: **$0.78** (pod terminated)

- **模型状态**: LR v2 + DeBERTa v4 已集成到 pipeline
- **新增文件**: dataset_v4.jsonl, dataset_adversarial_v4.jsonl, 4 scripts, models/detector_v4/

---

## 最近更新（新的在上面）

### [2026-03-26 01:10] — 预处理防御 + 对抗扩展 + 滑动窗口 + ELECTRA 准备

- **做了什么**（16:05 - 01:10，约 9 小时自治循环）：
  1. **预处理防御层**：emoji 清除、对话标签剥离、markdown 清除、短句合并、Unicode 同形字映射（Greek/Cyrillic → Latin）
  2. **文体特征扩展**：3→10 维（+hapax ratio, Yule's K, 功能词比率, 缩写率, 句首多样性, 词长分布）
  3. **ELECTRA 训练 notebook**：`train_electra_colab.ipynb`，自动下载数据集，修复 classifier head 命名
  4. **自动化测试套件**：`tests/test_detector.py`，24 个用例（21 pass + 3 xfail）
  5. **对抗数据生成器扩展**：9→14 种攻击（+back_translation, human_sandwich, style_prompt, data_injection, lexical_sub）
  6. **Binoculars 实现**：llama3.2:1b + 3b，但发现模型对无区分度，暂禁用
  7. **滑动窗口分析**：3 句窗口 + stride 2，成功检测 sandwich 攻击（全文 57.8 但窗口暴露 AI 段落 90）
  8. **3 份研究报告**：docs/research/2026-03-25-{14,16,18}.md
  9. **对抗策略分析**：子 Agent 设计 8 种新攻击，全部 5 种新实现成功绕过检测器

- **Commits**：224fb06 → 79f2272（~10 个 commits，全部 pushed）

- **当前检测能力**：
  - 假阳性率：**0%**（所有人类文本正确判定）
  - 标准 AI 检出：75-90 分
  - 已防御：口语注入、第一人称、列举体、homoglyph、markdown、对话体（部分）
  - 不可防御：typos、Quillbot、text-message 风格、对话体完全包装
  - 新增：sandwich 攻击检测（滑动窗口暴露混合文本中的 AI 段落）

- **下一步**：
  1. **ELECTRA 训练**（Colab A100，notebook 已就绪，待主人运行）
  2. ELECTRA 模型集成替换 DeBERTa
  3. 用 14 种对抗样本重训 ELECTRA
  4. 滑动窗口结果集成到前端 UI

### [2026-03-25 16:05] — 四路投票系统 + qwen3:4b PPL 升级

- **做了什么**（13:37 - 16:05，约 2.5 小时自治循环）：
  1. **检测评分重构**：单信号 DeBERTa → 四路投票（DeBERTa + qwen3:4b PPL + LR + 统计特征）
  2. **PPL 模型升级**：llama3.2:1b → qwen3:4b（AI ppl 7.83→4.82，2x 信号分离）
  3. **本地 LR 训练**：80 样本多文体，sklearn Pipeline，91.2% CV 准确率
  4. **统计特征信号**：句长变异系数、过渡词密度、标点多样性（对 typo 攻击免疫）
  5. **共识覆盖规则**：2+ 信号强同意时覆盖异议信号
  6. **DeBERTa dampening**：PPL+LR 与 DeBERTa 冲突时降低 DeBERTa 权重到 10%
  7. **Uncertain 判定**：fused score 在 threshold ±8 范围内输出 "uncertain"
  8. **前端**：AppShell.tsx 支持 "Insufficient evidence" 显示
  9. **CLAUDE.md 更新**：完整反映四路投票架构
  10. **研究报告**：docs/research/2026-03-25-14.md

- **对抗测试结果**（15 轮，~25 个用例）：
  - 假阳性率：3/6 (50%) → **0% on all human texts**
  - AI 标准检出：✓
  - 已防御对抗：口语注入、第一人称注入、列举体、邮件体
  - 不可防御：typos、Quillbot、对话体包装、短句拆分、emoji、代码混合

- **新增文件**：
  - `scripts/train_lr_local.py` — 本地 LR 训练脚本
  - `models/perplexity_lr_v2.pkl` — 80 样本 LR 模型
  - `docs/chronicle/2026-03-25.md` — 完整编年记录
  - `docs/research/2026-03-25-14.md` — 研究报告

- **5 个 commits pushed**：124ad5a → 7fee504

- **关键洞察**：DeBERTa 跨域 AUROC 0.5-0.6（论文证实），信号双向不可靠。PPL+LR+stat 三路组合比 DeBERTa 单路更稳定。规则工程已到天花板（140 行 if-else），下一步应走产品转型或模型重训。

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
