"""前向逐步：产出「单源 → 组合」那条路径。

这是四个搜索器里**唯一一个产物本身就是结论**的：

```
第 0 步：从 5 个顶点里挑最好的一个（= 最好的单源）
第 1 步：再加入一个分量，重新配比
第 2 步：再加一个
…直到加不动为止
```

每一步都记进 `trace`，于是「组合比单源好多少、好到第几步就饱和」这两件事
可以从一条轨迹上直接读出来 —— 而它们正是 README 验收标准一
（「多 Provider 组合相对单 Provider 的一致性提升」）的可视化形态。

三条约定：

1. **第 0 步必须真的评估全部顶点**，不能只评估「看起来最好」的那个。
   顶点之间的一致性差异往往很小（几个百分点），凭先验挑会挑错。
2. **允许停下来**。如果加任何一个分量都让一致性变差，就停 —— 报告里如实写
   「第 k 步饱和」。**「加分量总是更好」是个会被推翻的期望**，代码不许假设它成立。
3. **每步只加一个分量**（一次只动一个变量）。一次加两个，涨了也不知道是谁的功劳，
   而这条轨迹的全部价值就在于「每一步归因到一个分量」。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import SearchResult, SearchTrial
from .base import Objective

__all__ = ["run_forward"]


def run_forward(obj: Objective, cfg: Mapping[str, Any], rng: Any) -> SearchResult:
    n = obj.n_components
    taus = [float(t) for t in cfg["search"]["forward_taus"]]
    budget = int(cfg["search"]["budget"])
    obj.reset_counters()
    trace: list[SearchTrial] = []
    step = 0

    # ---- 第 0 步：全部顶点都真的评估一遍
    best_k = 0
    best_v = -1.0
    best_w: tuple[float, ...] = tuple(1.0 if i == 0 else 0.0 for i in range(n))
    for k in range(n):
        w = tuple(1.0 if i == k else 0.0 for i in range(n))
        v = obj(w)
        trace.append(SearchTrial(searcher="forward", step=step, w=w, agreement=v, cached=False))
        step += 1
        if v > best_v + 1e-15:
            best_v = v
            best_k = k
            best_w = w

    w = list(best_w)
    v = best_v
    support = {best_k}

    # ---- 逐步加入新分量
    while len(support) < n and obj.n_evals < budget:
        pick_w: list[float] | None = None
        pick_v = v
        pick_k = -1
        for k in range(n):
            if k in support:
                continue
            for tau in taus:
                if obj.n_evals >= budget:
                    break
                cand = [(1.0 - tau) * w[i] + (tau if i == k else 0.0) for i in range(n)]
                cv = obj(cand)
                trace.append(
                    SearchTrial(searcher="forward", step=step, w=tuple(cand), agreement=cv, cached=False)
                )
                step += 1
                if cv > pick_v + 1e-15:
                    pick_v = cv
                    pick_w = cand
                    pick_k = k
        if pick_w is None:
            # 加不动了。**如实停下**，不为了「跑满」而接受一个更差的点。
            break
        w = pick_w
        v = pick_v
        support.add(pick_k)

    return SearchResult(
        searcher="forward",
        best_w=tuple(w),
        best_agreement=v,
        n_evals_new=obj.n_evals,
        n_evals_cached=obj.n_cached,
        trace=tuple(trace),
    )
