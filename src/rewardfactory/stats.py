"""小统计工具箱。**零依赖**，只用标准库。

放在单独一个模块里的理由：这些函数被 `normalize`（方向检测）、`calibrate`
（分位数）、`benchmark`（相关性 / ECE）三处共用。三处各写一份的后果不是「重复」，
而是**同一份数据算出三个不同的相关系数** —— 那种 bug 不会报错，只会让报告
自相矛盾。

三条约定：

1. **秩相关一律用 Spearman 的「平均秩」版本**。不做平均秩时，大量并列值
   （`ordinal10` 的整数格点、`grid3` 的三档）会让秩相关的结果依赖于排序稳定性，
   而排序稳定性又依赖于输入顺序 —— 于是「打乱候选顺序」这条不变量会莫名其妙地挂。
2. **分位数用线性插值**（numpy 默认口径）。`int(p·n)` 那种写法在 p=1.0 时越界，
   在样本少时偏移一个位置，而校准的分位数映射对这两件事都敏感。
3. **`pearson` 允许标准差为 0 时返回 0.0**，不抛异常：常数序列与任何东西都
   「不相关」，把它算成 NaN 会让下游的 ECE 静默变成 NaN。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "mean",
    "stdev",
    "variance",
    "median",
    "quantile",
    "ranks",
    "pearson",
    "spearman",
    "clamp",
    "sigmoid",
    "norm_cdf",
]


def mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else 0.0


def variance(xs: Sequence[float], *, sample: bool = True) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = mean(xs)
    denom = (n - 1) if sample else n
    return math.fsum((x - mu) ** 2 for x in xs) / denom


def stdev(xs: Sequence[float], *, sample: bool = True) -> float:
    return math.sqrt(variance(xs, sample=sample))


def median(xs: Sequence[float]) -> float:
    return quantile(xs, 0.5)


def quantile(xs: Sequence[float], p: float) -> float:
    """线性插值分位数。`xs` 不必有序（内部排序）。空序列返回 0.0。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    p = clamp(p, 0.0, 1.0)
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def ranks(xs: Sequence[float]) -> list[float]:
    """平均秩（1 基）。并列值取平均 —— 见模块文档第 1 条。"""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            out[order[t]] = avg
        i = j + 1
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError(f"pearson 的两列长度必须相同：{len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """平均秩上的 Pearson。并列值多时比 Pearson 稳健得多。"""
    if len(xs) != len(ys):
        raise ValueError(f"spearman 的两列长度必须相同：{len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def norm_cdf(x: float) -> float:
    """标准正态 CDF，用 `math.erf`（C 实现）。**不查表、不近似** ——
    表会把 `zscore` 校准变成一个离散映射，而离散映射在 anchor 少时
    会退化成阶梯，掩盖掉它本该有的平滑性。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def counts(xs: Iterable[object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        k = str(x)
        out[k] = out.get(k, 0) + 1
    return out
