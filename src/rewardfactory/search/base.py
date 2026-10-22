"""搜索协议：把「与人类偏好的一致性」包成一个可缓存的目标函数。

被搜索的是**权重向量** `w`（单纯形上）。这是本项目与工作区里其它所有项目的
根本区别：`DataMixtureFactory` 搜的是数据配方，`ArchitectureFactory` 搜的是架构，
而这里搜的是 —— **「什么才算好」本身**。README 的「Objective Function Factory」
说的就是这件事。

## 本模块不许出现 `latent_weights`

搜索器只看得见 `(ΔC, label)` 对，看不见真值权重、也看不见真值 `quality`。
有对抗用例读源码文本断言 `search/` 下没有 `latent_weights` / `quality`。
真值权重只由 `benchmark.py`（第三方）用来算上界与 CI 覆盖率 ——
**搜索权重的人不许看真值权重**，这是本项目的纪律核心。

## 为什么每个搜索器拿**各自**的去重缓存（刻意偏离工作区约定）

`DataMixtureFactory` 的约定是「跨搜索器共享去重集合」，理由是省训练预算。这里**不共享**：

- 本项目的评估是纯内存计算（µs 级），共享省不下任何东西；
- 共享会让**先跑的搜索器把后跑的喂饱** —— `grid` 穷举 495 个点之后，`coordinate`
  的每一次评估都是缓存命中，于是「坐标上升要几次评估才追平穷举」这个问题
  被缓存抹掉了，对照组直接失去意义。

省下的是钱，赔上的是结论。评估便宜的时候，隔离比省更重要。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..aggregate import decide
from ..schema import COMPONENTS, SearchResult, SearchTrial

__all__ = [
    "Objective",
    "PairRow",
    "Searcher",
    "make_searcher",
    "build_objective",
    "search_all",
    "simplex_lattice",
]

#: 去重键的量化精度。网格是 1/8 = 0.125，量化到 1e-3 足够精确区分。
_KEY_SCALE = 1000.0


def simplex_lattice(n: int, den: int) -> list[tuple[float, ...]]:
    """单纯形上分母为 `den` 的全部格点（`Σc = den`，`c ≥ 0` 整数）。

    格点数 = `C(den+n-1, n-1)`：`den=8, n=5` ⇒ **495 个**。这个规模小到可以
    穷举 —— 于是 `grid` 搜索器能给出**真正的全局最优**，其它搜索器才有
    一个可信的上界可以对照。「有个 oracle 在旁边」是判断启发式搜索好坏的唯一办法。

    生成顺序是**字典序**（从全给第一个分量开始），确定，不依赖任何迭代顺序。
    """
    if n <= 0 or den <= 0:
        raise ValueError(f"单纯形格点需要 n>0 且 den>0，得到 n={n}, den={den}")
    if n == 1:
        return [(float(den),)]
    out: list[tuple[float, ...]] = []

    def rec(prefix: list[int], i: int, rem: int) -> None:
        if i == n - 1:
            out.append(tuple((prefix + [rem])[j] / den for j in range(n)))
            return
        for c in range(rem + 1):
            rec(prefix + [c], i + 1, rem - c)

    rec([], 0, den)
    return out


@dataclass(frozen=True)
class PairRow:
    """搜索目标函数的一行：`ΔC`（候选 A 减候选 B 的分量差）、人类标签、成对不确定度。

    三个字段**全部**来自「人类给的」或「我们算的」，没有一个来自真值：
    `ΔC` 来自校准后的分量表，`label` 来自人类标注，`u_pair` 来自 anchor 残差。
    """

    delta_c: tuple[float, ...]
    label: int
    u_pair: float


class Objective:
    """`w → 与人类标注的一致率`。**带缓存**：同一个 `w` 不重复算。

    内层循环用 `math.sumprod`（CPython 3.12 起的 C 实现）。手写
    `sum(a*b for ...)` 在 5 维向量上会因为生成器开销比它慢 2 倍以上 ——
    而这个循环要被跑几百万次（bootstrap 200 次 × 每次几十次评估 × 几百对）。
    """

    __slots__ = ("rows", "kappa", "n_components", "_cache", "n_evals", "n_cached", "n_trials")

    def __init__(self, rows: Sequence[PairRow], *, kappa: float, n_components: int) -> None:
        self.rows = tuple(rows)
        self.kappa = float(kappa)
        self.n_components = int(n_components)
        self._cache: dict[tuple[int, ...], float] = {}
        self.n_evals = 0
        self.n_cached = 0
        self.n_trials = 0

    def key(self, w: Sequence[float]) -> tuple[int, ...]:
        return tuple(int(round(float(x) * _KEY_SCALE)) for x in w)

    def __len__(self) -> int:
        return len(self.rows)

    def __call__(self, w: Sequence[float]) -> float:
        k = self.key(w)
        hit = self._cache.get(k)
        if hit is not None:
            self.n_cached += 1
            return hit
        wl = [float(x) for x in w]
        correct = 0
        sp = math.sumprod
        for row in self.rows:
            s = sp(wl, row.delta_c)
            if abs(s) <= self.kappa * row.u_pair:
                pred = 0
            else:
                pred = 1 if s > 0 else -1
            if pred == row.label:
                correct += 1
        val = correct / len(self.rows) if self.rows else 0.0
        self._cache[k] = val
        self.n_evals += 1
        self.n_trials += 1
        return val

    def cached(self, w: Sequence[float]) -> bool:
        return self.key(w) in self._cache

    def value(self, w: Sequence[float]) -> float:
        """只读查询（不计数、不写缓存）。"""
        k = self.key(w)
        if k in self._cache:
            return self._cache[k]
        return self(w)

    def reset_counters(self) -> None:
        """把「本次搜索用了多少次评估」清零。缓存**保留** —— 同一搜索器内部
        的重复提案仍然算命中，那才是 `n_evals_cached` 该反映的东西。"""
        self.n_evals = 0
        self.n_cached = 0


#: 搜索器签名：`(objective, cfg, rng) -> SearchResult`。
Searcher = Callable[[Objective, Mapping[str, Any], Any], SearchResult]


def build_objective(
    pairs: Sequence[Any],
    components: Mapping[str, Sequence[float]],
    u_pairs: Mapping[str, float],
    *,
    kappa: float,
    n_components: int,
) -> Objective:
    """从成对标注 + 分量表造目标函数。

    **只吃训练集的对。** 目标函数里出现 holdout 的任何一个数，验收标准一
    就变成自评了。
    """
    rows: list[PairRow] = []
    for p in pairs:
        ca = components[p.a]
        cb = components[p.b]
        d = tuple(float(ca[k]) - float(cb[k]) for k in range(n_components))
        rows.append(PairRow(delta_c=d, label=int(p.label), u_pair=float(u_pairs.get(p.id, 0.0))))
    return Objective(rows, kappa=kappa, n_components=n_components)


def _make_trial(searcher: str, step: int, w: Sequence[float], val: float, cached: bool) -> SearchTrial:
    return SearchTrial(
        searcher=searcher, step=step, w=tuple(float(x) for x in w), agreement=float(val), cached=cached
    )


def search_all(
    objective_factory: Callable[[], Objective],
    cfg: Mapping[str, Any],
    rng: Any,
) -> dict[str, SearchResult]:
    """把配置里声明的每个搜索器跑一遍。

    `objective_factory` 是一个**工厂**而不是一个实例：每个搜索器拿一个**全新的**
    目标函数（各自独立的缓存）。理由见模块文档 —— 共享缓存会让先跑的把后跑的喂饱。
    """
    names = [str(s) for s in cfg["search"]["searchers"]]
    out: dict[str, SearchResult] = {}
    for name in names:
        fn = make_searcher(name)
        obj = objective_factory()
        obj.reset_counters()
        out[name] = fn(obj, cfg, rng)
    return out


def make_searcher(name: str) -> Searcher:
    """名字 → 搜索器。**在调用时**构造注册表，不在导入时 ——
    四个搜索器都要 `from .base import ...`，模块导入期构造注册表会成环。"""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"未声明的搜索器：{name!r}（可选：{sorted(reg)}）")
    return reg[name]


def _registry() -> dict[str, Searcher]:
    from .coordinate import run_coordinate
    from .forward import run_forward
    from .grid import run_grid
    from .random_search import run_random

    return {
        "grid": run_grid,
        "random": run_random,
        "coordinate": run_coordinate,
        "forward": run_forward,
    }
