"""坐标上升：一次只动一个分量。

起点是等权（配置 `search.start`）。每一轮遍历全部分量，对每个分量尝试一组
「把质量从别处搬过来」的步长 `t ∈ {0.1, 0.25, 0.5, 0.75, 1.0}`：
`w' = (1−t)·w + t·e_k`。接受**最好的那个改善**；一轮里一个改善都没有就停。

三条刻意的设计：

1. **一次只动一个分量**。同时动几个，一致性涨了也不知道是哪个分量起的
   （这是工作区里 `SyntheticFactory` 「一轮一条变异」的同一条纪律）。
2. **接受最好的那个改善，而不是第一个改善**。第一个改善会让结果依赖分量遍历
   顺序；「最好的那个」把这个自由度消掉，于是「打乱分量顺序结果不变」这条
   不变量更容易成立（虽然严格来说仍有并列平手的可能，那时按分量下标取小）。
3. **平手按分量下标取小**。`>` 而不是 `>=` —— 用 `>=` 会让后访问的分量
   在平手时替换前面的，结果依赖遍历顺序。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..schema import SearchResult, SearchTrial
from .base import Objective

__all__ = ["run_coordinate", "coordinate_ascent"]


def _start_point(obj: Objective, cfg: Mapping[str, Any]) -> tuple[float, ...]:
    n = obj.n_components
    mode = str(cfg["search"].get("start", "uniform"))
    if mode == "uniform":
        return tuple(1.0 / n for _ in range(n))
    if mode == "best-vertex":
        best = max(range(n), key=lambda k: obj(tuple(1.0 if i == k else 0.0 for i in range(n))))
        return tuple(1.0 if i == best else 0.0 for i in range(n))
    raise ValueError(f"未声明的起点：{mode!r}（可选：uniform / best-vertex）")


def coordinate_ascent(
    obj: Objective,
    *,
    moves: Sequence[float],
    budget: int,
    start: Sequence[float],
    searcher: str = "coordinate",
    record: bool = True,
) -> tuple[tuple[float, ...], float, list[SearchTrial]]:
    """坐标上升主体。抽出来是为了让 bootstrap 能用同一段代码、同一个预算口径
    —— 交叉验证里再写一份实现，会让「bootstrap 的结论」与「主搜索的结论」
    来自两份不同的算法，那种不一致没有任何办法发现。
    """
    n = obj.n_components
    w = [float(x) for x in start]
    v = obj(w)
    trace: list[SearchTrial] = []
    step = 0
    if record:
        trace.append(SearchTrial(searcher=searcher, step=step, w=tuple(w), agreement=v, cached=False))
    improved = True
    while improved and obj.n_evals < budget:
        improved = False
        for k in range(n):
            best_w = None
            best_v = v
            for t in moves:
                if obj.n_evals >= budget:
                    break
                cand = [(1.0 - t) * w[i] + (t if i == k else 0.0) for i in range(n)]
                cv = obj(cand)
                step += 1
                if record:
                    trace.append(
                        SearchTrial(searcher=searcher, step=step, w=tuple(cand), agreement=cv, cached=False)
                    )
                # 严格大于：平手保留**先访问**的那个（分量下标小的），
                # 于是结果不依赖遍历顺序之外的东西。
                if cv > best_v + 1e-15:
                    best_v = cv
                    best_w = cand
            if best_w is not None:
                w = best_w
                v = best_v
                improved = True
    return tuple(w), v, trace


def run_coordinate(obj: Objective, cfg: Mapping[str, Any], rng: Any) -> SearchResult:
    moves = [float(t) for t in cfg["search"]["coordinate_moves"]]
    budget = int(cfg["search"]["budget"])
    obj.reset_counters()
    start = _start_point(obj, cfg)
    w, v, trace = coordinate_ascent(obj, moves=moves, budget=budget, start=start, searcher="coordinate")
    return SearchResult(
        searcher="coordinate",
        best_w=w,
        best_agreement=v,
        n_evals_new=obj.n_evals,
        n_evals_cached=obj.n_cached,
        trace=tuple(trace),
    )
