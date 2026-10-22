"""穷举网格：**真正的全局最优**，作为其它搜索器的上界参考。

分母 `den=8`、5 个分量 ⇒ 495 个格点。这不是「另一个启发式」，是 oracle：
在这个离散化的搜索空间里，它给出的就是最优解。有它在旁边，
「坐标上升用 240 次评估追平了穷举的 495 次」这句话才有意义。

**它不吃 `budget`。** 其它三个搜索器受预算约束，`grid` 不受 —— 因为它的定义
就是「把空间穷举完」。把它也砍到 240 次会得到「穷举的前 240 个点」，那既不是
上界也不是启发式，是第三种没人说得清的东西。

`trace` 是**全部 495 次评估**，不截断。它是「最优点周围的曲面长什么样」的唯一
记录；截断会让报告里那张「网格上的一致性曲面」变成假的。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import SearchResult, SearchTrial
from .base import Objective, simplex_lattice

__all__ = ["run_grid"]


def run_grid(obj: Objective, cfg: Mapping[str, Any], rng: Any) -> SearchResult:
    den = int(cfg["search"]["den"])
    points = simplex_lattice(obj.n_components, den)
    obj.reset_counters()
    trace: list[SearchTrial] = []
    best_w: tuple[float, ...] = points[0]
    best_v = -1.0
    for i, w in enumerate(points):
        v = obj(w)
        trace.append(SearchTrial(searcher="grid", step=i, w=w, agreement=v, cached=False))
        if v > best_v:
            best_v = v
            best_w = w
    return SearchResult(
        searcher="grid",
        best_w=tuple(best_w),
        best_agreement=best_v,
        n_evals_new=obj.n_evals,
        n_evals_cached=obj.n_cached,
        trace=tuple(trace),
    )
