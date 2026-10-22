"""权重搜索器。

被搜索的对象是**目标函数本身**（分量权重 `w`）—— 这就是 README 说的
「Objective Function Factory」。

```
base.py           协议：Objective（可缓存的目标函数）+ 单纯形格点 + 注册表
grid.py           穷举（oracle / 上界参考）
random_search.py  Dirichlet 随机（**必需的对照**）
coordinate.py     坐标上升（一次只动一个分量）
forward.py        前向逐步（产出「单源 → 组合」那条路径）
```

**本包内不许出现 `latent_weights` / `quality` / `truth`** —— 搜索权重的人
不许看真值权重。真值只由 `benchmark.py`（第三方）使用。有对抗用例读源码断言。
"""

from __future__ import annotations

from .base import (
    Objective,
    PairRow,
    Searcher,
    build_objective,
    make_searcher,
    search_all,
    simplex_lattice,
)

__all__ = [
    "Objective",
    "PairRow",
    "Searcher",
    "build_objective",
    "make_searcher",
    "search_all",
    "simplex_lattice",
]
