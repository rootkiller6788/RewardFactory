"""随机搜索：**必需的对照组**，不是凑数的。

它把「策略的贡献」从「预算的贡献」里分出来。四个搜索器跑同一个目标函数、
同一个预算、同一批训练对，差异才只可能来自策略。

提案分布是 `Dirichlet(1,…,1)`（单纯形上的均匀分布）。**不是**「在立方体里均匀
采样再归一化」—— 后者会把质量堆在单纯形的中心附近（中心极限效应），
于是随机搜索会意外地偏爱等权点，对照组的性质就变了。

去重在这里是**真实生效**的：Dirichlet 是连续分布，但量化到 1e-3 之后，
240 次采样里出现重复是常态（5 维单纯形上，中心附近的格点被反复命中）。
`n_evals_cached > 0` 就是这件事的证据，也是「随机搜索有一部分预算被浪费掉」
这个结论的来源。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import SearchResult, SearchTrial
from .base import Objective

__all__ = ["run_random"]


def run_random(obj: Objective, cfg: Mapping[str, Any], rng: Any) -> SearchResult:
    budget = int(cfg["search"]["budget"])
    n = obj.n_components
    obj.reset_counters()
    trace: list[SearchTrial] = []
    best_w = tuple(1.0 / n for _ in range(n))
    best_v = -1.0
    step = 0
    guard = 0
    while obj.n_evals < budget and guard < budget * 20:
        guard += 1
        w = tuple(rng.dirichlet1(n))
        cached = obj.cached(w)
        v = obj(w)
        trace.append(SearchTrial(searcher="random", step=step, w=w, agreement=v, cached=cached))
        step += 1
        if v > best_v:
            best_v = v
            best_w = w
    return SearchResult(
        searcher="random",
        best_w=tuple(best_w),
        best_agreement=best_v,
        n_evals_new=obj.n_evals,
        n_evals_cached=obj.n_cached,
        trace=tuple(trace),
    )
