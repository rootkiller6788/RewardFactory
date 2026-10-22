"""配置：口径与门槛全部在这里，代码里不硬编码任何阈值。

**一条特别的分区纪律**（本项目最重要的一条）：配置里同时住着「真值」与「门槛」，
而它们必须被不同的模块读。

| 键 | 谁能读 | 谁**不许**读 |
| --- | --- | --- |
| `world.latent_weights` | `world.py`（生成）、`benchmark.py`（评测/上界参考） | `search/**`、`aggregate.py`、`calibrate.py` |
| `providers[*].invert` | `providers.py`（模拟物理） | `normalize.py`、`calibrate.py` |
| `providers[*].declared_direction` | 只有报告（用来对比「声明 vs 实测」） | `normalize.py`（方向必须**测**出来） |

「搜索权重的人不许看真值权重」是这个项目的纪律核心：`benchmark.py` 是第三方，
它才允许拿真值去算上界与覆盖率。`acceptance.py` 有一条对抗用例**读源码文本**
断言 `search/` 下没有 `latent_weights`，`normalize.py` / `calibrate.py` 里没有 `invert`
—— 靠约定不如靠断言。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rng import canonical_json, sha256_text
from .schema import (
    CALIBRATION_METHODS,
    COMPONENTS,
    PROVIDER_KINDS,
    SCALE_FAMILIES,
    SEARCHERS,
)

__all__ = [
    "DEFAULTS",
    "load_config",
    "deep_merge",
    "validate_config",
    "protocol_digest",
    "version_digest",
    "providers_of",
    "world_of",
    "latent_weights_of",
    "acceptance_of",
    "expectations_of",
    "split_sizes",
]

#: 全部默认口径。`configs/default.json` 是它的落盘副本（两者必须一致，
#: 有对抗用例逐位比对）。
DEFAULTS: dict[str, Any] = {
    "seed": "rewardfactory",
    "world": {
        # 120 个题面 = 20 anchor + 60 train + 40 holdout。三者**互斥**。
        "n_prompts": 120,
        "anchor_prompts": 20,
        "train_prompts": 60,
        "holdout_prompts": 40,
        "n_candidates": 4,
        # 真值：quality = Σ w_true·comp + N(0, residual_sigma)，再 clip 到 [0,1]。
        # 这块残差是「人类偏好里谁也读不到的部分」——没有它，一个完美 provider
        # 就能拿满分，整个验收退化成同义反复。
        "residual_sigma": 0.06,
        "latent_weights": {
            "helpfulness": 0.20,
            "correctness": 0.34,
            "safety": 0.16,
            "style": 0.10,
            "task_reward": 0.20,
        },
        # 安全违规是**稀疏**的（6%）。稀疏 ⇒ 候选间方差极小 ⇒ w_safety 在数学上
        # 不可辨识。这是有意做的，它让「输出不确定度」从装饰品变成必需品。
        "safety_violation_prob": 0.06,
        "safety_violation_drop": 0.55,
        # 长度：与 helpfulness 弱相关（0.40），与一个正交的 verbosity 强相关。
        # 于是「更长 = 更好」是个**近似成立但会翻车**的启发式 —— 长度偏差
        # 才有杀伤力（校准修不掉非单调的那部分）。
        "length_base": 60.0,
        "length_help": 40.0,
        "length_verbosity": 30.0,
        "length_sigma": 12.0,
        "min_tokens": 20,
        "max_tokens": 400,
        # 人类标注：近平局（|Δq| ≤ tie_band）有 70% 概率被标成平局，
        # 其余情况以 label_noise 概率选错一侧。
        "tie_band": 0.02,
        "tie_declare_prob": 0.70,
        "label_noise": 0.12,
    },
    "anchors": {
        # anchor 上的人类分量评分也带噪声 —— 现实中人工标注从来不是神谕。
        # 它同时是「校准的输入」与「可靠性的唯一来源」，所以这个数不能为 0。
        "component_sigma": 0.08,
    },
    "providers": [
        {
            "id": "rule",
            "kind": "rule",
            "label": "机械规则（判据内精确、判据外全盲，但判据只覆盖一小片）",
            # **这一组数字改过一次，改动本身就是一条结论。**
            # 第一版写的是 `correctness 0.97 / safety 1.00 / task_reward 0.90`
            # —— 那不是一个规则，那是一个**神谕**：它在 4 个分量上几乎完美，
            # 只在 style（权重只有 0.10）上全盲。实测的后果是「最好单源」以
            # 88% 的对干净标签一致性压过 5 源组合的 84%，README 的核心命题
            # （组合优于单源）被自己的模拟推翻 —— 而推翻它的是一个不真实的 provider。
            # 规则的真实性格是**窄**：它只能检查它写死的那几条。
            # 于是改成「判据内精确、覆盖窄」：σ 仍为 0（确定性是规则的本性），
            # 但视力全面下调，且盲区落在两个高权重分量上
            # （correctness 0.34 与 style 0.10，合起来 44% 的权重）。
            #
            # 视力**不能压得太低**：`rule` 的尺度是 `grid3`（三档），
            # 视力压到 0.12/0.30 时它的原始标量被量化压成**精确常数 0.5**
            # （实测 480 个候选只有一个取值）—— 一个标量恒为 0.5 的 provider
            # 在「尺度异质性」这件事上什么也演示不了，反而让
            # `scale_variance_ratio` 这条验收变成 max/0。
            # 窄要体现在**覆盖**上（两个分量全盲），不是体现在**分辨率**上。
            "vision": {
                "helpfulness": 0.35,
                "correctness": 0.00,
                "safety": 0.70,
                "style": 0.00,
                "task_reward": 0.50,
            },
            "sigma": 0.0,
            "blind_noise": 0.10,
            "scale": "grid3",
            "declared_direction": 1,
            "invert": False,
            "length_bias": 0.0,
            "family_bias": {},
            "cost_ms": 0.4,
            "note": "确定性。同一输入永远同一输出，所以它的「不可复现」风险为零。"
                    "代价是它只看得见自己写死的那几条判据。",
        },
        {
            "id": "rm_small",
            "kind": "rm_model",
            "label": "小奖励模型（只看得见 correctness 一侧，logit 尺度）",
            "vision": {
                "helpfulness": 0.45,
                "correctness": 0.80,
                "safety": 0.00,
                "style": 0.00,
                "task_reward": 0.35,
            },
            "sigma": 0.10,
            "blind_noise": 0.10,
            "scale": "logit",
            "declared_direction": 1,
            "invert": False,
            "length_bias": 0.0,
            "family_bias": {},
            "cost_ms": 3.1,
            "note": "尺度是无界的 logit，方差比别的 provider 大一个量级。",
        },
        {
            "id": "rm_large",
            "kind": "rm_model",
            "label": "大奖励模型（覆盖最广，但把「长」读成「好」、且**看不见 safety**）",
            # safety 全盲不是随手加的：这是奖励模型的**已知真实缺陷**
            # —— 在通用偏好数据上训出来的 RM，对「安全」这一维几乎没有信号
            # （偏好数据里安全违规的样本极少，且往往被「有用」压过去）。
            # 把它写进模拟有两个好处：
            #   1. 「覆盖最全的那个 provider 也有洞」⇒ 组合才有存在的理由；
            #   2. 洞落在**高权重**分量上（safety 0.16），而不是权重最低的 style
            #      —— 第一版把 rule 的洞放在 style 上，结果那个洞几乎不花钱。
            "vision": {
                "helpfulness": 0.72,
                "correctness": 0.78,
                "safety": 0.00,
                "style": 0.70,
                "task_reward": 0.68,
            },
            "sigma": 0.07,
            "blind_noise": 0.09,
            "scale": "saturated",
            "declared_direction": 1,
            "invert": False,
            "length_bias": 0.35,
            "family_bias": {},
            "cost_ms": 18.0,
            "note": "sigmoid 饱和，原始分被压在 (0.2, 0.8) 里，方差最小。"
                    "「长 = 好」是非仿射偏差，校准修不掉。",
        },
        {
            "id": "llm_judge",
            "kind": "llm_judge",
            "label": "LLM 裁判（覆盖最全，噪声最大，偏爱写作类）",
            "vision": {
                "helpfulness": 0.66,
                "correctness": 0.62,
                "safety": 0.70,
                "style": 0.58,
                "task_reward": 0.60,
            },
            "sigma": 0.14,
            "blind_noise": 0.12,
            "scale": "ordinal10",
            "declared_direction": 1,
            "invert": False,
            "length_bias": 0.0,
            "family_bias": {"writing": 0.12},
            "cost_ms": 240.0,
            "note": "整数 1..10 的格点尺度，且把「文风偏好」混进了风格分。",
        },
        {
            "id": "rm_legacy",
            "kind": "rm_model",
            "label": "历史遗留 RM（**接线接反了**：分数越高越差）",
            "vision": {
                "helpfulness": 0.50,
                "correctness": 0.65,
                "safety": 0.00,
                "style": 0.00,
                "task_reward": 0.00,
            },
            "sigma": 0.12,
            "blind_noise": 0.12,
            "scale": "logit",
            "declared_direction": 1,
            "invert": True,
            "length_bias": 0.0,
            "family_bias": {},
            "cost_ms": 2.6,
            "note": "它**自称**分数越高越好（declared_direction=1），实际相反。方向只能测出来。",
        },
    ],
    "normalize": {
        # 判定「全盲」的阈值：anchor 上的原始分标准差低于它就认为这个 provider
        # 对该分量没有视力。全盲的分量必须标成 `None`（看不见），不是 0（很差）。
        # **它只在「没算成 ρ」时才起作用** —— 真正的判据是下面那条。
        "blind_tol": 0.005,
        # 判定「全盲」的相关系数门槛：|ρ(输出, 人类分量评分)| 低于它就算没信息。
        #
        # **这个数必须按样本量定，不能拍脑袋。** 零假设（真盲）下 ρ 的抽样标准差
        # ≈ 1/√(n−1)；本项目 n = 80 个 anchor ⇒ sd(ρ) ≈ 0.113。原来和方向检测
        # 共用 0.15，那只有 1.33σ —— 每个真盲列有 ~18% 的概率被漏判成「看得见」，
        # 实测 8 个真盲列漏了 3 个（`rm_small.style`、`rm_legacy.style`、
        # `rm_legacy.task_reward`，它们的 ρ 分别是 −0.209 / +0.190 / +0.160）。
        # 0.25 ≈ 2.2σ，把单列漏检率压到 ~3%。
        "blind_rho_tol": 0.25,
        # 方向检测至少要几个 anchor 才敢下结论。不够就报「不可判」而不是猜。
        "min_anchor_for_direction": 8,
        # 方向**投票**的相关系数容差：|ρ| 低于它的分量不参与投票。
        # 与 `blind_rho_tol` 分开是有意的：一个分量可以「看得出方向」但
        # 「看不出多少信息」，两件事的门槛没有理由相同。
        "direction_rho_tol": 0.15,
    },
    "calibrate": {
        "method": "isotonic",
        "min_anchor": 12,
        # `identity` 会被 clip 到 [0,1]；`minmax` 用 anchor 的极值线性拉伸；
        # `zscore` 走正态 CDF；`isotonic` 是 PAVA 保序回归；`beta` 是分位数映射。
        "clip_lo": 0.0,
        "clip_hi": 1.0,
        # 可靠性 ρ = 1 − 2·MAE（MAE 在分量尺度上）。随机猜的 MAE ≈ 0.25 ⇒ ρ ≈ 0.5。
        "reliability_slope": 2.0,
        "reliability_floor": 0.0,
        "reliability_ceiling": 1.0,
    },
    "aggregate": {
        # 分量聚合是**逆方差加权**：`w_pk ∝ 1/σ_pk²`，σ = 该 (provider, 分量) 的
        # 交叉验证残差。这是「对同一个量的多个独立估计」的标准最优组合。
        # 曾经用过 `ρ^power`（ρ = 1 − slope·mae 的线性映射）—— 那是错的：
        # `slope` 是一个任意的单调变换，而变换会改变权重。实测把它换成逆方差后
        # 组合才开始稳定地赢过最好单源。
        # `max_provider_sigma`：残差尺度大于它的 provider 在该分量上**直接出局**
        # —— 它比「不测」还差（噪声比信号大）。
        "max_provider_sigma": 0.35,
        "fallback": "equal",
    },
    "uncertainty": {
        # 是否**计算并报告**权重项 `u_w`。它**不进判定** —— 判定用纯分量项。
        # 理由是一个环：算 u_w 要先有 w，求 w 的搜索又要用 u 判平局。
        # 折进去会让「搜索的目标函数」依赖于「搜索的结果」。
        "include_weight_term": True,
        # 判平局的尺度：|ΔR| ≤ tie_kappa·(u_a + u_b) ⇒ 判平局。
        # 这个数是「不确定度有没有被真的用上」的开关，不是装饰参数。
        "tie_kappa": 1.0,
        # 不确定度下限：即使 anchor 残差为 0，也不许报 0（报 0 会让所有对都判胜负）。
        "floor": 0.01,
        # ECE 的分桶数。
        "ece_bins": 8,
    },
    "search": {
        "space": "simplex",
        # 网格步长 1/den。den=8、5 个分量 ⇒ C(12,4)=495 个格点，可以穷举。
        "den": 8,
        "budget": 240,
        "searchers": ["grid", "random", "coordinate", "forward"],
        "coordinate_moves": [0.1, 0.25, 0.5, 0.75, 1.0],
        "forward_taus": [0.1, 0.2, 0.35, 0.5],
        "start": "uniform",
    },
    "bootstrap": {
        "n_resamples": 200,
        "ci_level": 0.90,
        # bootstrap 里用的搜索器要便宜 —— 200 次重跑用穷举是浪费。
        "searcher": "coordinate",
        "budget": 60,
    },
    "benchmark": {
        # 「p 特有误差」的判据：p 错而至少这么多**别的** provider 对。
        "specific_majority": 3,
        # 误差分解只用在「干净标签」上（|Δq| > tie_band），否则不可约误差没意义。
        "clean_only": True,
    },
    "acceptance": {
        # ① 组合相对最好单源的一致性提升（验收标准一）。
        # **口径是 `agreement_truth`**（对干净标签 `sign(Δq)`、平局算半分），
        # 不是「与人类标签逐对相等」—— 后者被平局率主导：人类只判 5% 平局，
        # 而任何把 u 当真的方法都会判 20~70% 平局，于是那一列比不出谁排得准。
        "min_mixture_gain_over_best_single": 0.03,
        # ② 权重搜索的可复现性与不确定度（验收标准二）。
        # **刻意不设「点估计与真值的偏差」门槛**：`world.py` 的纪律二与
        # `bootstrap_weights` 的文档都写明「点估计必然偏离真值（标签有噪声、
        # 样本有限）」，再拿它当门槛就是自相矛盾 —— 那条期望放在 expectations
        # 里，被推翻，并且是附录 B 里最值钱的一条。
        "min_weight_ci_coverage": 0.60,
        # 校准本身
        "min_calibration_gain": 0.05,
        "max_mae_after": 0.20,
        "min_reliability_spread": 0.15,
        # 尺度异质性必须真的存在（否则「不校准会输」是同义反复）
        "min_scale_variance_ratio": 3.0,
        "min_naive_deficit": 0.02,
        # 不确定度：ECE 量的是「u 与误差**散布**是否匹配」（误差已去偏），
        # `min_uncertainty_lift` 量的是「最不确定的那批确实更差」。
        # 秩相关（`uncertainty_mae_correlation`）只报不设门槛 —— 误差由模型偏差
        # 主导，u 只看得到测量噪声，秩相关在 160 个点上本来就低。
        "max_ece": 0.08,
        "min_uncertainty_lift": 1.05,
        # 结构
        "min_providers": 4,
        "min_holdout_pairs": 150,
        "min_anchor_items": 60,
        "min_solo_spread": 0.03,
    },
    # ---- 事先声明的设计期望。**允许被推翻，被推翻的那条最值钱。**
    "expectations": {
        "calibration_improves_agreement": True,
        "mixture_beats_best_single": True,
        "naive_uncalibrated_mix_underperforms_best_single": True,
        "component_decomposition_beats_scalar_mix": True,
        "searched_weights_recover_latent_pointwise": True,
        "safety_weight_is_identifiable": True,
        "llm_judge_is_best_single_provider": True,
        "direction_detection_changes_agreement": True,
        "direction_detection_recovers_dead_provider": True,
        "uncertainty_is_calibrated": True,
        "weights_are_reproducible": True,
        "bagging_beats_argmax": True,
        "uncertainty_predicts_error_rank": True,
        "reward_has_no_scale_offset": True,
    },
}

#: 期望的中文说法。写在配置旁边而不是报告里 —— 报告是产物，期望是**跑之前**的信念。
EXPECTATION_CLAIMS: dict[str, str] = {
    "calibration_improves_agreement": "校准之后，每个 provider 单独的一致性都变高",
    "mixture_beats_best_single": "组合优于最好的单个 provider（验收标准一）",
    "naive_uncalibrated_mix_underperforms_best_single": "不校准的等权平均比最好单源还差（尺度异质性会主导）",
    "component_decomposition_beats_scalar_mix": "分量分解比「校准标量再平均」更有用",
    "searched_weights_recover_latent_pointwise": "搜出的 w* 逐分量逼近注入的真值权重",
    "safety_weight_is_identifiable": "safety 分量的权重可辨识",
    "llm_judge_is_best_single_provider": "LLM 裁判是最强的单源",
    "direction_detection_changes_agreement": "不检测方向会明显拉低一致性",
    "direction_detection_recovers_dead_provider": "检测方向能把反号的 provider 从「死重」变回「贡献者」",
    "uncertainty_is_calibrated": "输出的不确定度与真实误差量级相符（ECE 小）",
    "weights_are_reproducible": "同一种子下 w* 逐位可复现",
    "bagging_beats_argmax": "bootstrap 均值（bagged）比搜索器的 argmax 更好",
    "uncertainty_predicts_error_rank": "u 越大，误差越大（u 与 |误差| 的秩相关 ≥ 0.15）",
    "reward_has_no_scale_offset": "R 与 quality 之间没有系统性常数错位（|bias| ≤ 0.05）",
}


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """深合并。**列表整体替换**，不做逐元素合并 —— 列表的「合并」没有唯一语义，
    让后写的那份完全接管比猜一个更安全。"""
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    """读配置并深合并到 `DEFAULTS` 上，然后校验。"""
    p = Path(path)
    if not p.is_absolute() and root is not None:
        p = root / p
    if not p.is_file():
        raise FileNotFoundError(f"没有配置文件：{p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = deep_merge(DEFAULTS, raw)
    validate_config(cfg)
    return cfg


def split_sizes(cfg: Mapping[str, Any]) -> dict[str, int]:
    w = cfg["world"]
    return {
        "anchor": int(w["anchor_prompts"]),
        "train": int(w["train_prompts"]),
        "holdout": int(w["holdout_prompts"]),
    }


def providers_of(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(p) for p in cfg["providers"])


def world_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg["world"])


def latent_weights_of(cfg: Mapping[str, Any]) -> tuple[float, ...]:
    """注入的真值权重（按 `COMPONENTS` 顺序）。**只有 world / benchmark 该调它。**"""
    lw = cfg["world"]["latent_weights"]
    return tuple(float(lw[k]) for k in COMPONENTS)


def acceptance_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg["acceptance"])


def expectations_of(cfg: Mapping[str, Any]) -> dict[str, bool]:
    return {str(k): bool(v) for k, v in cfg["expectations"].items()}


# ------------------------------------------------------------------ 校验

class ConfigError(ValueError):
    """配置不合法。**永远在跑之前抛，不许跑一半才炸。**"""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """把配置里的每条隐含假设变成一次显式检查。

    这个函数的存在理由：下面每一条被违反时，流水线**都不会报错**，只会给出
    一个看起来合理的错数字。校验是唯一能在跑之前拦住它们的地方。
    """
    # ---- 划分
    sizes = split_sizes(cfg)
    total = int(cfg["world"]["n_prompts"])
    _need(
        sum(sizes.values()) == total,
        f"三个划分之和必须等于 n_prompts：{sizes} 之和 {sum(sizes.values())} != {total}",
    )
    for name, n in sizes.items():
        _need(n >= 4, f"划分 {name} 太小（{n}）：样本不足时任何率都是噪声")
    _need(int(cfg["world"]["n_candidates"]) >= 2, "每个题面至少要 2 个候选，否则成对标注无从谈起")

    # ---- 真值权重
    lw = cfg["world"]["latent_weights"]
    missing = [k for k in COMPONENTS if k not in lw]
    _need(not missing, f"latent_weights 缺分量：{missing}")
    extra = [k for k in lw if k not in COMPONENTS]
    _need(not extra, f"latent_weights 有多余的键：{extra}（分量表是唯一的，不许自己发明）")
    vals = [float(lw[k]) for k in COMPONENTS]
    _need(all(v >= 0 for v in vals), f"latent_weights 不许为负：{vals}")
    _need(sum(vals) > 0, "latent_weights 之和必须为正")

    # ---- 概率类
    w = cfg["world"]
    for key in ("tie_declare_prob", "label_noise", "safety_violation_prob"):
        v = float(w[key])
        _need(0.0 <= v <= 1.0, f"world.{key} 必须在 [0,1] 内，得到 {v}")
    _need(float(w["residual_sigma"]) > 0, "residual_sigma 必须为正 —— 为 0 时完美 provider 能拿满分，验收会退化成同义反复")
    _need(float(cfg["anchors"]["component_sigma"]) > 0, "anchor 的人工评分也必须有噪声")
    _need(0.0 <= float(w["tie_band"]), "tie_band 不能为负")

    # ---- provider 池
    provs = cfg["providers"]
    _need(len(provs) >= 2, f"至少要 2 个 provider（README：Reward 不应该只有一个模型），得到 {len(provs)}")
    ids = [str(p["id"]) for p in provs]
    _need(len(set(ids)) == len(ids), f"provider id 必须唯一：{ids}")
    kinds = {str(p["kind"]) for p in provs}
    for kind in PROVIDER_KINDS:
        _need(kind in kinds, f"provider 池里必须至少有一个 {kind}（README 的架构图是三路：RM / LLM Judge / Rule）")
    for p in provs:
        pid = str(p["id"])
        _need(str(p["scale"]) in SCALE_FAMILIES, f"{pid} 的 scale={p['scale']!r} 未声明（可选：{SCALE_FAMILIES}）")
        _need(float(p["sigma"]) >= 0, f"{pid} 的 sigma 不能为负")
        _need(float(p.get("blind_noise", 0.0)) >= 0, f"{pid} 的 blind_noise 不能为负")
        _need(
            float(p["sigma"]) + float(p.get("blind_noise", 0.0)) > 0,
            f"{pid} 的 sigma 与 blind_noise 不能同时为 0 —— 那样它就是个完美的神谕，"
            "而「有缺陷的打分器」是这个项目要演示的对象",
        )
        vis = p["vision"]
        miss = [k for k in COMPONENTS if k not in vis]
        _need(not miss, f"{pid} 的 vision 缺分量：{miss}")
        for k in COMPONENTS:
            v = float(vis[k])
            _need(0.0 <= v <= 1.0, f"{pid}.vision[{k}]={v} 必须在 [0,1] 内")
        _need(int(p["declared_direction"]) in (1, -1), f"{pid} 的 declared_direction 只能是 ±1")
        _need(float(p.get("length_bias", 0.0)) >= 0, f"{pid} 的 length_bias 不能为负（方向由 scale 决定）")
        fb = p.get("family_bias", {})
        for fam in fb:
            _need(fam in ("arith", "logic", "writing", "code", "safety"), f"{pid} 的 family_bias 有未知家族：{fam!r}")
    # 至少一个 provider 对某分量全盲，否则「盲区」这条路径永远不被走到
    _need(
        any(float(p["vision"][k]) == 0.0 for p in provs for k in COMPONENTS),
        "至少要有一个 provider 对某个分量全盲 —— 否则盲区处理这条代码路径永远不被执行到",
    )

    # ---- 归一化 / 校准
    nz = cfg["normalize"]
    _need(float(nz["blind_tol"]) > 0, "blind_tol 必须为正")
    _need(float(nz["blind_rho_tol"]) > float(nz["direction_rho_tol"]),
          "blind_rho_tol 必须大于 direction_rho_tol —— 判「没信息」要比判「方向反了」严格，"
          "否则一条噪声列会同时被当成「有信息」和「方向可信」")
    _need(int(nz["min_anchor_for_direction"]) >= 4, "方向检测至少需要 4 个 anchor")
    cal = cfg["calibrate"]
    _need(str(cal["method"]) in CALIBRATION_METHODS, f"calibrate.method={cal['method']!r} 未声明（可选：{CALIBRATION_METHODS}）")
    _need(int(cal["min_anchor"]) >= 4, "校准至少需要 4 个 anchor")
    _need(float(cal["clip_lo"]) < float(cal["clip_hi"]), "clip 区间必须非空")
    _need(float(cal["reliability_slope"]) > 0, "reliability_slope 必须为正")

    # ---- 聚合
    ag = cfg["aggregate"]
    _need(float(ag["max_provider_sigma"]) > 0, "max_provider_sigma 必须为正")

    # ---- 不确定度
    un = cfg["uncertainty"]
    _need(float(un["tie_kappa"]) >= 0, "tie_kappa 不能为负")
    _need(float(un["floor"]) >= 0, "不确定度下限不能为负")
    _need(int(un["ece_bins"]) >= 2, "ECE 至少分 2 桶")

    # ---- 搜索
    se = cfg["search"]
    _need(str(se["space"]) == "simplex", f"只支持 simplex 搜索空间，得到 {se['space']!r}")
    _need(int(se["den"]) >= 2, "网格分母至少为 2")
    _need(int(se["budget"]) >= 8, "搜索预算太小")
    _need(len(se["searchers"]) >= 1, "至少要有一个搜索器")
    for name in se["searchers"]:
        _need(str(name) in SEARCHERS, f"未声明的搜索器：{name!r}（可选：{SEARCHERS}）")
    _need("random" in se["searchers"], "随机搜索是**必需的对照**：没有它，分不出「策略的贡献」与「预算的贡献」")
    for t in se["forward_taus"]:
        _need(0.0 < float(t) <= 1.0, f"forward_taus 里的 {t} 必须在 (0,1] 内")
    for t in se["coordinate_moves"]:
        _need(0.0 < float(t) <= 1.0, f"coordinate_moves 里的 {t} 必须在 (0,1] 内")

    # ---- bootstrap
    bs = cfg["bootstrap"]
    _need(int(bs["n_resamples"]) >= 20, "bootstrap 至少 20 次，否则 CI 本身是噪声")
    _need(0.5 < float(bs["ci_level"]) < 1.0, "ci_level 必须在 (0.5,1) 内")
    _need(str(bs["searcher"]) in SEARCHERS, f"bootstrap 的搜索器未声明：{bs['searcher']!r}")

    # ---- 评测
    bm = cfg["benchmark"]
    _need(int(bm["specific_majority"]) >= 1, "specific_majority 至少为 1")

    # ---- 验收门槛与期望
    acc = cfg["acceptance"]
    for key in (
        "min_mixture_gain_over_best_single",
        "min_weight_ci_coverage",
        "min_calibration_gain",
        "max_mae_after",
        "min_reliability_spread",
        "min_scale_variance_ratio",
        "min_naive_deficit",
        "max_ece",
        "min_uncertainty_lift",
        "min_providers",
        "min_holdout_pairs",
        "min_anchor_items",
        "min_solo_spread",
    ):
        _need(key in acc, f"acceptance 缺门槛：{key}")
    _need(0.0 <= float(acc["min_weight_ci_coverage"]) <= 1.0, "min_weight_ci_coverage 必须在 [0,1] 内")
    _need(float(acc["min_scale_variance_ratio"]) >= 1.0, "min_scale_variance_ratio 至少为 1")
    _need(float(acc["min_uncertainty_lift"]) >= 1.0, "min_uncertainty_lift 至少为 1（否则门槛没有方向）")

    exp = cfg["expectations"]
    _need(len(exp) >= 6, f"声明的期望太少（{len(exp)} 条）：期望必须在跑之前写死，太少就没有可被推翻的空间")
    for name in exp:
        _need(name in EXPECTATION_CLAIMS, f"期望 {name!r} 没有对应的中文说法，补进 EXPECTATION_CLAIMS")


# ------------------------------------------------------------------ 指纹

def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """协议摘要：口径与门槛的指纹。

    **刻意排除 `world.latent_weights`** —— 换一组注入的真值权重，协议没变，
    变的是「这一次注入了什么」。把两者混进同一个指纹，会让「换了注入量」
    看起来像「换了协议」。
    """
    slim = copy.deepcopy(dict(cfg))
    slim.get("world", {}).pop("latent_weights", None)
    return sha256_text(canonical_json(slim))


def version_digest(cfg: Mapping[str, Any]) -> str:
    """版本摘要：协议 + 注入实例。

    与 `protocol_digest` 的分工：**协议摘要回答「规则是什么」，版本摘要回答
    「这一次跑的是哪一份输入」**。产物目录里的 `version.json` 用后者。
    """
    return sha256_text(canonical_json(cfg))
