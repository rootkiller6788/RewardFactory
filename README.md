# RewardFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L4 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | RewardFactory |
| 类型 | Factory |
| 真正有意思的核心 | Reward / Judge 模型评估与组合 |

---

## 二、核心命题

这个项目负责：

> 把抽象的"好不好"变成一个可计算的 Reward。

**Reward 不应该只有一个模型。** 单一 RM 会把自己的偏差变成训练目标的偏差。

## 三、整体架构

```
             Response
                │
       ┌────────┼───────────┐
       ▼        ▼           ▼
    RM Model  LLM Judge   Rule
       │        │           │
       ▼        ▼           ▼
      r1       r2          r3
       └────────┼───────────┘
                ▼
           Reward Mixer
                │
                ▼
             Reward
```

## 四、统一 Reward Interface

```
RewardProvider
    score(prompt, response) -> RewardEvidence
```

```
RewardEvidence
├── total_score
├── components
│   ├── helpfulness
│   ├── correctness
│   ├── safety
│   ├── style
│   └── task_reward
├── uncertainty
├── judge
└── evidence
```

任何 Reward 来源（RM 模型 / LLM Judge / 规则）都实现同一个接口，返回带**分量与不确定度**的证据，而不是一个裸标量。这是能组合、能校准、能归因的前提。

## 五、主流水线

```
Input
 ↓
Candidate Response
 ↓
Reward Providers
 ↓
Normalize
 ↓
Calibrate
 ↓
Aggregate
 ↓
Uncertainty Estimate
 ↓
Reward
 ↓
Benchmark
```

## 六、更高级一点

```
R(x, y) = Σ_i w_i · R_i(x, y)
```

然后让：

```
w = [helpfulness, correctness, safety, style]
```

**也进入搜索空间。**

这时 RewardFactory 本质上变成：

> **Objective Function Factory**

也就是说，连"什么才算好"这件事本身都是可搜索、可优化的——而不是在配置里写死一组权重。

## 七、阶段性落地计划

- **M0｜接口骨架**：定义 RewardProvider / RewardEvidence 接口，跑通单一 RM 打分一次（非自动）。
- **M1｜多来源接入**：实现 RM Model / LLM Judge / Rule 三类 Provider 与 Normalize / Calibrate。
- **M2｜权重搜索**：把 `w` 参数化，接评分器搜索权重组合，输出 Uncertainty Estimate。
- **M3｜评测与发布**：以"Reward 与人类偏好的一致性"为 benchmark，输出可复现报告。

## 八、验收标准

能端到端完成「RewardFactory」，并在其「核心」（Reward / Judge 模型评估与组合）上给出可复现的量化结果：

- 给出多 Provider 组合相对单 Provider 的一致性提升
- 权重 `w` 的搜索结果可复现，并输出不确定度

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L4 深化设计稿；可按实际进展修订。_

---

# 附录 A｜骨架、快速开始、CLI 契约、产物

## A.1 目录

```
L4-rewardfactory/
├── configs/default.json            所有口径与门槛（代码里不硬编码任何阈值）
├── data/                           空 —— 真值世界在内存里合成，不需要外部数据
├── src/rewardfactory/
│   ├── rng.py                      确定性随机（sha256 派生种子）+ canonical_json
│   ├── schema.py                   对象模型：ProviderSpec / ProviderOutput / ProviderScore
│   │                               / WeightVector / MethodScore / BenchmarkResult
│   ├── config.py                   配置加载 + 校验 + 两类摘要（protocol / version）
│   ├── world.py                    **真值世界**：120 题面 × 4 候选、5 个潜在分量、
│   │                               注入真值权重、anchor/train/holdout 三分
│   ├── providers.py                5 个 provider 的合成模型 + **尺度族**（grid3 / logit /
│   │                               saturated / ordinal10）+ 方向反转 + 长度偏差
│   ├── normalize.py                方向检测 + 盲区检测（**都只用 anchor**）
│   ├── calibrate.py                5 种校准方法（identity/minmax/zscore/isotonic/quantile）
│   ├── aggregate.py                两层聚合：C[k] = Σ_p w_pk·c_pk（测出来的）→
│   │                               R = Σ_k w_k·C_k（搜出来的）+ u 的传播
│   ├── uncertainty.py              ECE / 分位诊断
│   ├── search/                     4 个搜索器：grid（495 格点，全局最优）/ random /
│   │                               coordinate / forward（单源 → 组合）
│   ├── benchmark.py                9 个方法同台 + bootstrap 权重 + 误差分解 + 消融
│   ├── lineage.py                  血缘 DAG（加边时拒绝成环）
│   ├── pipeline.py                 编排 + 落盘 + load_products
│   ├── acceptance.py               验收清单 + 对抗用例（**本项目的质量闸门**）
│   ├── report.py                   markdown 报告
│   ├── visualize.py                自包含 HTML（零外部资源，4 张内联 SVG）
│   └── cli.py                      2 写 + 13 只读子命令
├── tests/                          入口级冒烟（见 A.6）
├── main.py                         入口（零安装：把 src/ 塞进 sys.path）
└── out/                            产物（见 A.5）
```

## A.2 形状与不变量

**形状：目标组合型（Objective Composition）。** 这是本 monorepo 的**第九种形状**。它与已有
形状的区别值得说清楚，因为选错形状会把整个项目做歪：

| 形状 | 被优化的对象 | 产物 |
| --- | --- | --- |
| 直线流水线 | 没有（`objectives_of()` 返回空） | 一份数据集 / 一份报告 |
| 进化闭环 | 一个解 | 更好的解 |
| 搜索闭环（Optimizer） | 一份**配方**（超参 / 数据配比） | 配方 + 它的不确定度 |
| **目标组合型** | **「什么才算好」本身** | **一个可调用的目标函数 + 它的不确定度** |

它和 `L4-preferencefactory`（对抗裁判池）最容易混：那边扇出的是**裁判**、收拢点是**一致率**、
产物是**偏好数据**；这里扇出的是**打分器**、收拢点是**目标函数**、产物是**可以拿去当 loss 的东西**。

四条纪律，每一条都有代码或验收在守：

1. **搜权重的人不给自己打分，打分的人不搜权重。**
   `search/**` 一行真值都不读；`benchmark.py` 是第三方，读真值算上界。有一条对抗用例
   读源码断言 `search/` 里不出现 `latent_weights`。
2. **方向必须测出来，不许读配置。** `normalize.py` / `calibrate.py` 里**不许出现 `invert`**
   （两条对抗用例读源码文本断言）。`rm_legacy` 自称 `+1`、实际接反了 —— 抄配置的检测器
   永远是对的，于是这条验收永远绿，那才是最坏的结果。
3. **盲区的判据是「没有信息」，不是「输出是常数」。** 见附录 C 的 181–184。
4. **交付的目标函数是 `bagged_w`（bootstrap 均值），不是搜索器的 argmax。**
   搜索目标是分片常数的（一对对地数对错、无梯度），argmax 落在单纯形**顶点**上。

**四层对象链**：

```
World（真值：潜在 5 分量 + 注入真值权重 w_truth + 人类分量评分）
   │  providers：5 个打分器 × 各自尺度族 × 各自盲区 × 各自偏差
   ▼
ProviderOutput（原始标量 + 5 个原始分量；分量可以为 None = 看不见）
   │  normalize：方向对齐 + 盲区标 None + 尺度族记录
   ▼
ProviderScore（方向一致的分数）
   │  calibrate：5 个分量各自的单调映射（isotonic），全部只喂 anchor
   ▼
CalibrateResult（cal_total + 逐分量 C 的可靠来源）
   │  aggregate 第 1 层：C[k] = Σ_p eff[p][k]·c_pk，eff ∝ 1/σ²（逆方差）
   ▼
AggregateResult（C[k] + u_components）
   │  aggregate 第 2 层：R = Σ_k w_k·C_k，w 由搜索给
   ▼
Reward + u（带分量与不确定度的证据，而不是一个裸标量）
   │  benchmark：9 个方法同台 + bootstrap + 误差分解 + 消融
   ▼
BenchmarkResult（含 oracle_truth_w —— 只有第三方能读）
```

## A.3 快速开始

```bash
python main.py run            # 整条流水线 + 落盘（16 个文件，约 2.0 s）
python main.py visualize      # out/report.html（自包含，约 0.4 s）
python main.py acceptance     # 验收清单 17 项 + 对抗用例 52 项
python main.py expectations   # 14 条设计期望 vs 实测（5 条被推翻）
```

不需要 `pip install`，不需要激活 venv。核心链路只用标准库（`requires-python >= 3.12`）。

## A.4 CLI 契约

**2 个写命令**：`run`（16 个产物）/ `visualize`（`report.html`）。

**14 个只读命令**（全部从产物重建，不重跑流水线；产物不全时才在内存里跑一次并明确提示）：

| 命令 | 回答的问题 |
| --- | --- |
| `summary` | 总览：规模、注入真值、w*、九方法一句话、期望被推翻几条 |
| `providers` | provider 池 + **声明 vs 实测**（方向、盲区、覆盖、MAE、单源一致性） |
| `calibration` | 逐 (provider, 分量) 的 MAE 前后 / ρ / 断点数 / 是否盲 |
| `weights` | w* 与注入真值并排 + bootstrap 90% CI + 覆盖率 |
| `search` | 4 个搜索器对照 + **`forward` 的路径**（单源 → 组合） |
| `benchmark` | **9 个方法同台** + 误差分解 + 分家族 + 不确定度诊断 |
| `pairs` | holdout 240 对的逐对明细（Δq / ΔR / u_pair / 标签 / 判定 / 是否噪声） |
| `scores` | 逐候选分数（480 行 × 25 列，含逐分量贡献） |
| `evidence` | `RewardEvidence` 的返回体（分量 + 不确定度 + provider 贡献 + 可靠来源） |
| `score <cid>` / `--text` | 单条打分：`--candidate` 从产物取，`--text` 走 sha256 派生 latent 的 stub |
| `expectations` | 14 条设计期望 vs 实测 |
| `acceptance` | 验收清单 + 对抗用例 |
| `check` | 验收的退出码版本：全过 rc=0，否则 rc=1（CI 闸门） |
| `lineage` | 血缘 DAG（`--dot` 输出 Graphviz） |

**`--json` 只输出 JSON**：说明文字被吞掉，`_emit_json` 直接写真正的 stdout。所以
`python main.py benchmark --json | jq` 是能用的。14 个只读命令全部支持。

## A.5 产物（`out/`，17 个文件）

`run` 写 16 个：

| 文件 | 内容 |
| --- | --- |
| `version.json` | 版本、digest、protocol_digest、counts、`created_at`（唯一不可复现字段） |
| `world.json` | 真值世界：指纹、注入真值权重、分片统计、**候选预览**（`candidates` 不整体导出） |
| `providers.json` | 5 个 provider 的规格 + 诊断 + 单源一致性 |
| `calibration.json` | 逐 (provider, 分量) 的校准映射与 MAE |
| `aggregate.json` | 两层聚合：C[k]、第 1 层权重 `eff`、u 的分量 |
| `weights.json` | w*、注入真值、bootstrap 估计、argmax 支持集 |
| `search.json` | 4 个搜索器 + `forward` 的逐条路径 |
| `benchmark.json` | 9 个方法 × 5 列 + 误差分解 + 分家族 + 消融 + `reward_bias` |
| `acceptance.json` | 17 条门槛 + 52 条对抗 |
| `expectations.json` | 14 条期望 vs 实测 |
| `evidence.json` | 480 条 `RewardEvidence` |
| `scores.csv` | 逐候选 25 列（含逐分量贡献 `contrib_*`） |
| `pairs.csv` | 720 对标注（anchor/train/holdout） |
| `lineage.json` / `lineage_reward.dot` | 血缘 DAG（27 节点 / 36 边） |
| `report.md` | markdown 报告（14 节，0–13） |

`visualize` 写第 17 个：`report.html`（自包含，4 张内联 SVG，零外部资源）。

## A.6 关于 `tests/`

质量闸门是 `acceptance.py`（17 条门槛 + 52 条对抗用例）。`tests/` 只放**入口级冒烟**，
不承担判据：

- `test_import.py`：`src/rewardfactory/` 下每个模块（含子包）都能干净 import；
- `test_cli_contract.py`：`--help` 与每个子命令的 `--help` 都 rc=0。

pytest 只回答「接线有没有断」，不回答「结果对不对」；后者由 `acceptance.py` 负责。

- **`main.py acceptance`** 一次跑两类东西：
  - **17 条门槛**核对这一批结果：组合增益、CI 覆盖率、校准增益、MAE、可靠性离散度、
    **尺度方差比**（「不校准会输」这件事的前提）、不校准差额、ECE、四分位提升、
    规模下限、单源离散度、血缘的三条「一个不多一个不少」、provider 贡献可加性。
  - **52 条对抗用例**，**不读产物**，所以产物过期时仍能跑。它们敲的是**检测器**：
    保序回归的单调性、常数输入的除零、端点外推必须是常数、ρ 判据与方差判据的分工、
    方向在反相关/常数/样本不足时的三种行为、`_csv` 的 `None ≠ 0`、`\n` 换行、
    Dirichlet 提案落在单纯形上、同一个种子给出同一串随机数、以及**两条读源码的墙**
    （`normalize.py`/`calibrate.py` 里不许出现 `invert`）。

---

# 附录 B｜实测结果

## B.1 环境与实测开销

| 项 | 值 |
| --- | --- |
| 解释器 | CPython 3.13.12（Windows） |
| 第三方依赖 | **0**（核心链路纯标准库） |
| `main.py run` | 约 **2.0 s**（含 4 个搜索器 + 200 次 bootstrap 重跑 `coordinate`） |
| `main.py visualize` | 约 **0.4 s** |
| 规模 | 120 题面 / 480 候选 / 720 对标注（anchor 80 个带人工分量评分 / train 360 对 / holdout 240 对） |
| 产物 | 17 个文件，`report.html` 35.8 KB，`report.md` 19.3 KB |

**版本**：`v1-dc354078e1ca`（协议 `dfc05cdf0aa7`）。产物用内容寻址版本，唯一不可复现的
字段是 `version.json` 的 `created_at`。

## B.2 验收标准逐条核对

README 第八节的两条验收标准，逐条落到具体数字：

**① 「给出多 Provider 组合相对单 Provider 的一致性提升」**

| 方法 | 对干净标签的一致性 |
| --- | ---: |
| `chance` | 50.00% |
| `best_single_raw`（最好单源，未校准） | 57.96% |
| `naive_raw_mean`（5 个等权平均，未校准） | 67.04% |
| `calibrated_scalar_mean`（校准标量再平均） | 67.92% |
| `component_argmax_w`（搜索器 argmax） | 79.87% |
| `best_single_calibrated`（最好单源，已校准） | 83.85% |
| `oracle_truth_w`（**注入真值权重的上界**） | 86.06% |
| **`component_searched_w`（交付的目标函数）** | **88.94%** |

**组合 88.94% − 最好单源 83.85% = +5.09 pp**（门槛 3.0 pp）。而且**超过了真值权重的上界
2.88 pp** —— 这件事不是 bug，见 B.5。

**② 「权重 `w` 的搜索结果可复现，并输出不确定度」**

- **可复现**：同一种子下两次独立搜索的 `w*` **逐位相同**（digest `76c8b0d34d3f`）。
- **不确定度**：200 次 bootstrap（每次重跑 `coordinate`）给出 90% CI：

| 分量 | w*（交付） | w（注入真值） | 点误差 | 90% CI | 覆盖真值 |
| :--- | ---: | ---: | ---: | :--- | :---: |
| helpfulness | 0.4422 | 0.2000 | 0.2422 | [0.0000, 0.9100] | 是 |
| correctness | 0.2550 | 0.3400 | 0.0850 | [0.0000, 0.8911] | 是 |
| safety | 0.0031 | 0.1600 | 0.1569 | [0.0000, 0.0251] | **否** |
| style | 0.0031 | 0.1000 | 0.0969 | [0.0000, 0.0251] | **否** |
| task_reward | 0.2967 | 0.2000 | 0.0967 | [0.0000, 0.9000] | 是 |

**CI 覆盖 3/5 = 60.0%**（门槛 60%）。**刻意不设「点估计误差」门槛** —— 一边说「点估计
必然偏离」（标签有噪声、样本有限），一边拿点估计误差当门槛，那是自相矛盾。

## B.3 设计期望 vs 实测（**5 条被推翻**）

14 条期望在跑之前就写死在 `configs/default.json` 里 —— 否则「期望被推翻」这件事就变成
事后编故事。**被推翻的 5 条是这份报告里最值钱的部分**，逐条给机制：

| 期望 | 声明 | 实测 | 结论 |
| :--- | :---: | :---: | :---: |
| calibration_improves_agreement | 成立 | 不成立 | **被推翻** |
| mixture_beats_best_single | 成立 | 成立 | 一致 |
| naive_uncalibrated_mix_underperforms_best_single | 成立 | 不成立 | **被推翻** |
| component_decomposition_beats_scalar_mix | 成立 | 成立 | 一致 |
| searched_weights_recover_latent_pointwise | 成立 | 成立 | 一致 |
| safety_weight_is_identifiable | 成立 | 不成立 | **被推翻** |
| llm_judge_is_best_single_provider | 成立 | 不成立 | **被推翻** |
| direction_detection_changes_agreement | 成立 | 成立 | 一致 |
| direction_detection_recovers_dead_provider | 成立 | 成立 | 一致 |
| uncertainty_is_calibrated | 成立 | 成立 | 一致 |
| weights_are_reproducible | 成立 | 成立 | 一致 |
| bagging_beats_argmax | 成立 | 成立 | 一致 |
| uncertainty_predicts_error_rank | 成立 | 不成立 | **被推翻** |
| reward_has_no_scale_offset | 成立 | 成立 | 一致 |

被推翻的五条，机制分别是：

1. **`calibration_improves_agreement`**：4/5 提高，`rm_small` **逐位不变**
   （84.96% → 84.96%，MAE 0.1446 → 0.1453）。不是「变差」，是**没变** —— 期望写的是
   「都变高」，把「没变」也算成了失败。校准对已经单调的分数只做一次单调重标；
   它真正救回来的是**尺度坏掉**的 provider（`rule` 54.4% → 86.7%）。
2. **`naive_uncalibrated_mix_underperforms_best_single`**：不校准等权平均 67.04% **高于**
   最好单源（未校准）57.96%。设计直觉是「尺度异质性 ⇒ 等权平均被 logit 尺度单方面决定」，
   实测反了：**等权平均的降方差效应压过了尺度异质性的伤害**。
3. **`safety_weight_is_identifiable`**：safety 的 90% CI 只有 0.0251 宽 —— 看起来「很可辨识」
   —— 但注入真值 0.16 落在它外面。**bootstrap CI 只量抽样波动、不量偏差**。safety 的真值
   贡献只占 quality 方差的 3.2%（`correctness` 一个分量就占 58%），而且它是唯一一个两头都
   不靠的分量：5 个 provider 里只有 1 个真的看得见它。这个样本量下本来就没有分辨它的功率。
4. **`llm_judge_is_best_single_provider`**：实测最强单源是 **`rule`**（86.7%）。「贵 = 强」
   是错的：`rule` 的 σ=0、确定性，在它覆盖的那 3 个分量上比谁都准；`llm_judge` 噪声最大、
   还偏爱写作类。
5. **`uncertainty_predicts_error_rank`**：Spearman(u, |误差|) = 0.0521（期望 ≥ 0.15），
   但四分位提升 **1.212**（门槛 1.05）。两条读数必须一起看：u 的**量级**是对的（ECE 0.0176），
   u 也抓得住**最难的那一批**，但它在整个 holdout 上的**秩**几乎不与误差同向。
   「能排序」与「能圈出最危险的少数」是两件事。

## B.4 四个搜索器与 `forward` 的路径

离散化分母 `1/8` ⇒ 单纯形上 **495 个格点**，所以 `grid` 是**真正的全局最优**（oracle），
不是另一个启发式。

| 搜索器 | 新增评估 | 缓存命中 | 最优一致性 | argmax 支持集 | 与 grid 的差 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| `coordinate` | 39 | 12 | 51.389% | task_reward | 0.000% |
| `forward` | 21 | 0 | 51.389% | task_reward | 0.000% |
| `grid` | 495 | 0 | 51.389% | task_reward | 0.000% |
| `random` | 240 | 0 | 48.333% | 全部 5 个 | 3.056% |

**`forward` 只用了 21 次评估就撞上了全局最优**（`grid` 要 495 次）。但它的支持集是
`task_reward` 一个分量 —— 训练集上「加分量」在 51.389% 就停住了。**argmax 落在单纯形
的顶点上**，这正是要 bagging 的原因。

`forward` 的路径（产物里最重要的一张表，它把「组合相对单源的提升」从数字变成过程）：

| 步 | 支持集 | 权重 | 训练集一致性 |
| ---: | :--- | :--- | ---: |
| 0 | helpfulness | helpfulness=1.000 | 50.833% |
| 4 | task_reward | task_reward=1.000 | 51.389% |

## B.5 为什么「组合超过真值权重的上界」不是 bug

`oracle_truth_w` = 86.06%，`component_searched_w` = 88.94%。组合赢了「上帝」2.88 pp。

原因是**可辨识性**：`oracle_truth_w` 用的是**注入的真值权重**，但真值权重是**按 latent
分量**定义的，而 `C[k]` 是 provider 们**校准后**的分量估计。`C[k]` 带着每个 provider 的
系统性缺陷（`rm_large` 把「长」读成「好」、`llm_judge` 偏爱写作类），于是「用真值权重加权
一组有缺陷的分量估计」并不是最优的 —— **在一个被扭曲的坐标系里，真值权重不是最优权重**。
搜出来的 `w` 顺带把 provider 的偏差也补偿掉了一部分。

这不违反「不许读真值」的纪律：`oracle_truth_w` 只在 `benchmark.py`（第三方）里出现，
搜索路径上一行都读不到。

## B.6 不确定度

| 项 | 值 | 说明 |
| --- | ---: | :--- |
| ECE | **0.0176** | 门槛 0.08。用**去偏误差**算 —— `R = Σw·C` 里各分量均值不等，只要 `w ≠ w_truth` 就有常数错位 |
| `reward_bias` | −0.0258 | `mean(R − quality)`，单独报出来，不归不确定度管 |
| u 均值 | 0.0437 | 系统项 + 分歧项，平方和传播（不是算术和） |
| 四分位提升 | **1.212** | 最高四分位 u 的 mean\|误差\| ÷ 最低四分位（门槛 1.05） |
| Spearman(u, \|误差\|) | 0.0521 | 期望 ≥ 0.15 → **被推翻**（见 B.3 第 5 条） |

`u_k` 的构成：**系统项** `Σ_p w_pk²·σ_pk²`（σ = 该 (provider, 分量) 的 CV MAE）
+ **分歧项** `Σ_p w_pk²(c_pk − C_k)² / (1 − Σ_p w_pk²)`（加权样本方差的无偏修正；
不除 `1 − Σw²` 会放大 √N 倍）。两层都用**平方和**传播：算术和会放大最多 √5 ≈ 2.24 倍。

## B.7 决定论与只读纯度

| 检查 | 结果 |
| --- | --- |
| 14 个只读命令跑完 `out/` | **逐位不变**（md5 指纹） |
| 14 个只读命令的 `--json` | 全部 `json.loads` 直接成功（说明文字被吞掉） |
| 两次 `main.py run` | 17 个文件里**只有 `version.json` 的 `created_at` 不同** |
| 两次 `main.py visualize` | `report.html` **逐位相同** |
| `report.html` 自包含 | 无 `http://` / `src=` / `@import` / `<link` / `<script` / `fetch(` / `<img`；4 张内联 SVG 全部能被 `xml.etree` 解析 |

## B.8 这些结论在哪失效

- **`safety` 的权重在这个样本量下不可辨识**（B.3 第 3 条）。要辨识它，得把 anchor 从 80
  提到几百个量级，或者让 safety 的取值不要挤在 0.93 附近。
- **`u` 的秩相关不成立**（B.3 第 5 条）。它只在「圈出最危险的四分之一」这件事上有效。
- **`llm_judge.safety` 被检测器判成全盲**（ρ = 0.169，与噪声不可区分）。它的配置视力是
  0.70，但在这个候选分布下它的 safety 信号确实被噪声吃掉了。
- **真值权重只有 5 个分量、候选只有 480 个**。这是一次**受控的机制演示**，不是对真实 RM
  的评测。所有「A 比 B 好」的结论都是「在这个合成世界里 A 比 B 好」。

---

# 附录 C｜跑起来才发现的问题与不变量

编号**跨项目连续**（ArchitectureFactory 1–57 / MoEFactory 58–74 / ModelGenomeFactory 75–93 /
FineTuneFactory 94–108 / PreferenceFactory 109–125 / AdapterFactory 126–160 /
TrainingGraphFactory 161–180），本项目从 **181** 起。

## C.1 盲区与方向：检测器自己会错（181–185）

### 181. 全盲分量输出**精确常数** ⇒ 一条「常数完美预言」

第一版里全盲分量就是硬编码 `0.5`。后果不是「这条分量没用」，而是**它变得比谁都准**：
`isotonic` 会把这条常数拟合到 anchor 上「人类评分的平均值」，于是它在 holdout 上也
永远给出那个平均值 —— 一个没有任何信息的列，被校准包装成了一个看起来还不错的预测器。
修法是给全盲分量加一点 `blind_noise`，**让它至少不可能是常数**。

### 182. 「标准差 ≈ 0」不是盲区的定义

加了噪声之后，盲区检测（第一版只有「标准差 ≈ 0」这一条判据）**彻底失效**：
`rule` 在 `style` 上 vision=0，标准差却不再是 0，于是它被当成一个有信号的分量。
**「常数」是盲区的实现方式，「没有信息」才是盲区的定义。** 判据改成
`sd < tol or |ρ(输出, 人类分量评分)| < rho_tol`。

顺带澄清一件事：`vision=0.15` 的分量其 ρ 仍有 0.9 —— **视力弱是尺度问题（校准能修），
全盲是信息问题（谁也修不了）**。这两件事必须分开判。

### 183. ρ 判据的阈值必须**按样本量定**

原来盲区判据和方向投票共用 `rho_tol = 0.15`。n = 80 个 anchor 时零假设下
sd(ρ) ≈ 1/√79 ≈ **0.113** —— 0.15 只有 **1.33σ**，每个真盲列有 ~18% 的概率被漏判成
「看得见」。实测 8 个真盲列漏了 3 个（`rm_small.style` ρ=−0.209、`rm_legacy.style`
ρ=+0.190、`rm_legacy.task_reward` ρ=+0.160）。改成独立的 `blind_rho_tol = 0.25`
（≈2.2σ）之后漏检 0/8，并且顺带把 `llm_judge.safety`（ρ=0.169）正确地标成了盲。

**教训**：任何相关系数阈值都必须写成「几个 σ」而不是一个拍脑袋的小数，否则它会随
样本量悄悄失效。配置校验里现在有一条：`blind_rho_tol > direction_rho_tol`。

### 184. 缺失的证据 ≠ 否定的证据

`detect_blind` 在**不传 `ratings_by_cid`** 时，ρ 恒为 0.0，于是 ρ 判据把**所有**分量
判成全盲 —— 一条对抗用例直接全红。修法：只有「真算出了 ρ」（样本数够）时 ρ 判据才生效，
否则退回「标准差」这一个判据，那是它唯一能说的话。**没测到和测到 0 必须分开。**

### 185. 方向检测：`invert` 出现一次就退化成抄答案

配置里每个 provider 都有 `declared_direction` —— 那是它的**自我声明**。`rm_legacy`
自称 `+1`，实际接反了。真实世界里这种事每周都在发生（换了个 wrapper、换了个
tokenizer、把 logprob 和 prob 搞混）。所以方向只能从 anchor 上的人类标注**测**回来，
而且有两条对抗用例**读源码文本**断言 `normalize.py` / `calibrate.py` 里不出现 `invert`。
靠约定不如靠断言。

## C.2 聚合与不确定度：量级口径（186–190）

### 186. 全盲分量**不能进标量合成**

`comps`（对外导出的分量）和「参与标量合成的分量」必须是两份不同的东西。全盲分量要出现
在 `comps` 里（带 `blind_noise`），否则 182 那条判据就永远成立、盲区检测退化成一
行 `is None`；但它**不能进标量** —— 把三个 ~N(0.5, 0.1) 的噪声平均进一个只看得见两三个
分量的 provider，等于用噪声把它的输出拽向 0.5，再被 `grid3` 量化压成**精确常数**
（实测：`rule` 的 `raw_total` 恒为 0.5，480 个候选只有一个取值，于是
`scale_variance_ratio` 变成 max/0 = 0，验收挂）。

### 187. 第 1 层权重：`ρ^power` 的线性映射**会改变权重**

第一版用「可靠性 ρ 的线性映射」当第 1 层权重。问题是 `slope` 只是单调变换，但**变换会
改变权重**：线性映射把 mae ∈ [0.066, 0.215] 压成 [0.87, 0.57]，于是**一个 mae 小 2 倍的
provider 只拿到 1.5 倍权重，而不是 4 倍**。这正是「组合输给最好单源」的真正根因。
改成**逆方差** `w_pk ∝ 1/σ_pk²`（σ = 该 (provider, 分量) 的 CV MAE）之后，
`reliability_spread` 从 0.1363 升到 0.1724，组合增益从负转正。

### 188. `eff` 必须**归一化后再存**

第 1 层权重归一化前就存进 `eff`，于是 `Σ_p contrib[p] == R` 这条可加性验收破掉
（实测偏差 **0.2436**）。修法：`eff[pid][k] = (1/σ²) / Σ_p' (1/σ'²)`，且因为归一化了，
`C_k` 就是那个未除分母的加权和。可加性恢复 1.0000。

### 189. `u` 的传播是**平方和**，不是算术和

`u_k = sqrt(系统项 + 分歧项)`，`u = sqrt(Σ_k (w_k·u_k)²)`。算术和会放大最多
√5 ≈ **2.24 倍** —— 而 ECE 会立刻抓到这个放大（ECE 就是「u 说的量级」与「误差的量级」
的差）。

### 190. `u` 必须**随候选变化**，否则 ECE 与秩相关都无从谈起

`u_k` = 系统项（**跨候选是常数**）+ 分歧项（**逐候选变化**）。只留系统项，`u` 就是一个
与候选无关的常数，ECE 和秩相关都变成在测一个常数。分歧项用加权样本方差的**无偏修正**
`Σ_p w_pk²(c_pk − C_k)² / (1 − Σ_p w_pk²)` —— 不除 `1 − Σw²` 会放大 √N 倍。

## C.3 评价口径：被平局率和偏差主导（191–194）

### 191. 单源基线的 `u` 口径必须与主方法一致

第一版给单源基线的 u 用「原始标量的离散度」，而主方法用「校准后残差」。**同一张表里
两个 u 口径**，于是「不校准会输」这个结论里混进了一个纯粹的口径差。修法：所有方法的
`u_pair` 都从同一个函数出来。

### 192. 「与人类标签逐对相等」被**平局率**主导

人类只判 5.4% 平局，而任何把 `u` 当真的方法判 20~70% 平局。于是「逐对相等」这个指标
主要在量「敢不敢表态」，不是「对不对」。实测的极端例子很直白：一个三档打分的 provider
在 480 个候选上有 464 个输出同一个值，人类标签口径给它 **13.33%** —— 那 13% 说的是
「它不表态」，不是「它错了 87%」。

修法：新增 `agreement_truth` —— 对**干净标签** `sign(Δq)`（`|Δq| > tie_band`）打分、
**平局算半分**。**主结论与两条验收都用这一列。** 而且 provider 表里的「单源·原/校」
两列也必须换成同一口径 —— 否则同一个 provider 在两张表里是两个东西。

### 193. ECE 必须用**去偏**误差

`R = Σ w·C` 里各分量均值不等（correctness 均值 ≈ 0.34 vs safety 均值 ≈ 0.93），
只要 `w ≠ w_truth` 就有常数错位（实测 −0.16）。这个错位会把 ECE 从 0.019 推到 0.13，
而它是**尺度问题**不是**不确定度问题**。修法：偏移单独报为 `reward_bias`，ECE 用去偏
之后的误差；并且加一条期望「`R` 与 quality 无常数错位」（实测 −0.0258，成立）。

### 194. 校准**不是免费的**：MAE 降 ≠ 排序升

保序回归在 1 维带噪分数上会造出**平台段**，平台段在成对比较里就是**平局**。于是可以
出现「点估计更准了、排序反而掉几个点」。更常见的第三种情况是**什么都没变**：
`rm_small` 的一致性 84.96% → 84.96%（逐位不变）—— 因为对已经单调的分数，保序回归
只是一次单调重标。期望写「每个 provider 都变高」，就把「没变」也算成了失败。

## C.4 期望与报告：写死的数字与过期的产物（195–199）

### 195. bootstrap CI 只量**抽样波动**，不量**偏差**

`safety` 的 90% CI 只有 0.0251 宽 —— 按「CI 窄 ⇒ 可辨识」的判据它漂亮地通过。但注入
真值 0.16 **落在它外面**。这个 CI 说的是「换一批标注，『safety 权重≈0』这个答案有多稳」，
不是「它有多对」。判据改成「窄**且**覆盖真值」，于是这条期望被正确地推翻了。
**「对错误答案很自信」看起来和「可辨识」一模一样。**

### 196. 「不校准就一定输」是错的

设计直觉：尺度异质性（logit 方差是 grid3 的 150 倍）会让等权平均被单方面决定，于是
「不校准等权平均 < 最好单源」。实测反了：67.04% vs 57.96%。**等权平均的降方差效应压过
了尺度异质性的伤害。** 尺度真正毁掉的是别的东西：未校准的 `u` 活在自己的尺度上，
`best_single_raw` 因此有 81.7% 的对被判平局 —— 一个 provider 未校准时连它自己的排序
能力都发挥不出来。

### 197. `u` 的秩相关与四分位提升可以**背离**

Spearman(u, |误差|) = 0.0521，四分位提升 1.212。同一份数据里两个看起来该同向的数
指向相反。它们测的不是一件事：**秩相关问「u 能不能给所有候选排序」，四分位提升问
「u 能不能圈出最危险的四分之一」**。只报前者会漏掉一个有用的性质，只报后者会把
「圈得住最难的」说成「会排序」。

### 198. `main.py acceptance` 读的是**产物**，不是现算

改完 `normalize.py` 之后直接跑 `acceptance`，看到的还是上一次 `run` 写下的结论 ——
「对抗 51/52」一直不消失，而现算早就 52/52 了。这类命令必须**先 `run` 再看**，
否则你会在调试一个已经修好的 bug。（对抗用例本身不依赖产物，但它们的结果是**在 run
时算好写进 `acceptance.json`** 的。）

### 199. 报告里的数字**不许写死**

note 里曾经硬编码过「1.13 倍」「correctness 占 58%」「最噪的那个 provider」。
改一次配置，这三句就全变成谎话，而且**报告里的谎话比缺一句话危险得多** ——
读者没法从报告内部发现它。现在这几个数全部由实测计算后插值。

## C.5 工程（200–202）

### 200. `visualize` 是**写命令**，不是只读命令

`report.html` 由 `main.py visualize` 写进 `out/`。把它算进「只读命令跑完 `out/` 逐位
不变」那组，那条检查永远失败；把它排除在交付产物之外，`out/` 就只有 16 个文件、
少了唯一一个给人看的东西。**写命令与只读命令的边界要一开始就划清**，否则「不写盘」
这条验收会变成一个口径争论。

### 201. `data/` 可以是空的

真值世界完全在内存里合成（`world.py`），`data/` 里没有任何文件。这不违反目录契约 ——
本 monorepo 里 `L2-moefactory/data` 也是空的。**目录契约约束的是「东西放哪」，
不是「必须有什么东西」。**

### 202. 这个工作区**当时没有 git**（现已是，但当前改动尚未提交）

批量文本替换（比如把 `res.agg` 改成 `res.aggregate`）之前必须先 `ast.parse` 一遍再落盘，
而且模式要避开跨行匹配 —— 兜底只到旧提交，一次正则手滑就是不可逆的。

