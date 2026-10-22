"""不确定度：把「这个分数离真值有多远」变成一个有量纲的数。

三项相加（平方和开根），三项来源不同、互不包含：

| 项 | 来源 | 谁在管 |
| --- | --- | --- |
| `u_comp` | 各 provider 在 anchor 上的交叉验证残差，按可靠性加权 | `aggregate.component_table` |
| `u_w` | 权重 `w` 本身的不稳（bootstrap 分布下同一个候选的分数散布） | `weight_uncertainty` |
| `floor` | 下限，防止「残差恰好为 0」让所有对都判胜负 | 配置 |

**`u` 不是一个报告字段，它参与判定**（`aggregate.decide` 用它判平局）。
所以「不确定度估得准不准」是可测的：ECE 小 ⇒ 准；ECE 大 ⇒ 要么太乐观
（永远硬选），要么太保守（永远判平）。

## 一个必须说清的偏保守来源

`u_comp` 的输入是「校准后分量 vs **人类分量评分**」的残差。人类评分自己有
`anchors.component_sigma = 0.08` 的噪声，所以 `u_comp` 里天然含了这 0.08 ——
而它要预测的是「R vs **真值 quality**」的误差，真值里没有这块噪声。

于是 `u` 系统性偏**保守**（高估）。这不是 bug，是**信息论上的必然**：
你手上只有带噪声的标注，误差估计里必然含标注噪声。报告里如实写明
「偏保守」，并给出实测的 `over_estimate_ratio`，比把它调准（那要靠偷看真值）诚实。

## provider 自报的不确定度不参与

`ProviderOutput.self_uncertainty` 是 provider 自己说的，实测它系统性乐观
（`SELF_UNCERTAINTY_DISCOUNT`）。产物里一律用**实测**的残差。
自报值只在报告里出现一次，作为「为什么不能信自报」的证据。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .stats import mean

__all__ = [
    "weight_uncertainty",
    "combine",
    "pair_uncertainty",
    "ece",
    "over_estimate_ratio",
    "uncertainty_report",
]


def weight_uncertainty(
    w_samples: Sequence[Sequence[float]],
    components: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """权重不确定度：`std` over bootstrap 权重下同一候选的分数散布。

    直接对 `R` 采样，而不是对 `w` 求协方差再传播 —— 后者要求 `R` 对 `w` 线性
    且需要完整的协方差矩阵，而这里 `w` 在单纯形上（受约束、边缘分布非正态），
    线性传播会低估。**直接重算 `R` 是最省的，也是唯一不需要额外假设的。**
    """
    if not w_samples:
        return {cid: 0.0 for cid in components}
    out: dict[str, float] = {}
    for cid, comps in components.items():
        vals: list[float] = []
        for w in w_samples:
            s = math.fsum(a * b for a, b in zip(w, comps))
            vals.append(s)
        mu = mean(vals)
        if len(vals) < 2:
            out[cid] = 0.0
            continue
        var = math.fsum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        out[cid] = math.sqrt(var)
    return out


def combine(u_comp: float, u_w: float, floor: float) -> float:
    """分量项 + 权重项（独立 ⇒ 平方相加）+ 下限。"""
    return max(floor, math.sqrt(u_comp * u_comp + u_w * u_w))


def pair_uncertainty(u_a: float, u_b: float) -> float:
    """成对不确定度 = `u_a + u_b`。

    **刻意与 `aggregate.decide` 用同一个口径。** 报告里如果出现第二个口径
    （比如 `sqrt(u_a²+u_b²)`），读者就再也算不出「为什么这一对判成了平局」——
    一份读不出判定过程的报告，等于没有报告。
    """
    return u_a + u_b


def ece(
    uncertainties: Sequence[float],
    errors: Sequence[float],
    *,
    bins: int = 8,
) -> float:
    """Expected Calibration Error：把样本按 `u` 分桶，比较桶内 `mean(u)` 与 `mean(|err|)`。

    加权口径（按桶内样本数加权），不是简单平均 —— 简单平均会让只有 1 个样本的
    极端桶与有 100 个样本的桶等权，而 ECE 想回答的是「**整体上**报得准不准」。

    空桶跳过。`bins < 2` 时退化为「全体一个桶」，此时 ECE = `|mean(u) − mean(err)|`。
    """
    if not uncertainties or len(uncertainties) != len(errors):
        return 0.0
    n = len(uncertainties)
    lo = min(uncertainties)
    hi = max(uncertainties)
    if bins < 2 or hi - lo <= 1e-12:
        return abs(mean(uncertainties) - mean([abs(e) for e in errors]))
    width = (hi - lo) / bins
    total = 0.0
    for b in range(bins):
        lo_b = lo + b * width
        hi_b = lo + (b + 1) * width
        idx = [
            i
            for i, u in enumerate(uncertainties)
            if (u >= lo_b and (u < hi_b or (b == bins - 1 and u <= hi_b)))
        ]
        if not idx:
            continue
        mu = mean([uncertainties[i] for i in idx])
        me = mean([abs(errors[i]) for i in idx])
        total += (len(idx) / n) * abs(mu - me)
    return total


def over_estimate_ratio(uncertainties: Sequence[float], errors: Sequence[float]) -> float:
    """`mean(u) / mean(|err|)`。>1 表示偏保守，<1 表示偏乐观。

    这个数比 ECE 更能说明**方向**：ECE 只说「差多少」，它说「往哪边差」。
    两者都要报 —— 只看 ECE 会把「偏保守」与「偏乐观」当成同一件事，
    而它们的修法完全相反。
    """
    me = mean([abs(e) for e in errors])
    if me <= 1e-12:
        return 0.0
    return mean(uncertainties) / me


def uncertainty_report(
    uncertainties: Mapping[str, float],
    errors: Mapping[str, float],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """把不确定度诊断打包（报告与产物共用一份口径）。"""
    cids = sorted(uncertainties)
    us = [uncertainties[c] for c in cids]
    es = [errors.get(c, 0.0) for c in cids]
    bins = int(cfg["uncertainty"]["ece_bins"])
    return {
        "n": len(cids),
        "mean_uncertainty": mean(us),
        "mean_abs_error": mean([abs(e) for e in es]),
        "ece": ece(us, es, bins=bins),
        "over_estimate_ratio": over_estimate_ratio(us, es),
        "max_uncertainty": max(us) if us else 0.0,
        "min_uncertainty": min(us) if us else 0.0,
    }
