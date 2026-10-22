"""归一化：把 provider 的原始输出变成「方向一致、盲区显式」的可比量。

**这一步不做尺度变换。** 把 logit 拉到 [0,1] 是 `calibrate` 的活。这里只做三件事：

1. **方向对齐**（`-1` 的 provider 把数值翻过来）；
2. **盲区标出**（该分量输出恒等于常数的 provider，标 `None` = 看不见，**不是 0**）；
3. **尺度族记录**（供报告诊断「不校准会输」的机制）。

## 方向必须**测**出来，不许读配置

配置里每个 provider 都有 `declared_direction` —— 那是它的**自我声明**。
`rm_legacy` 自称 `+1`（分数越高越好），实际接反了。真实世界里这种事每周都在发生
（换了个 wrapper、换了个 tokenizer、把 logprob 和 prob 搞混）。

所以方向只能从 **anchor 上的人类标注**测回来。测量用两样东西，都只来自 anchor：

- **人类分量评分** `anchor_ratings`（带噪声的人工打分）—— 逐分量算 Spearman ρ；
- 用 `|ρ|` 加权的**符号投票**决定整体方向（少数分量被噪声翻号时不至于把整体带偏）。

**本模块不许出现 `invert` 这个标识符** —— 有一条对抗用例读源码文本断言这一点。
靠约定不如靠断言：只要 `invert` 在这里出现一次，检测就退化成「抄答案」，
而抄答案的检测器永远是对的，于是这条验收永远绿 —— 那才是最坏的结果。

## 盲区：判据是「没有信息」，不是「输出是常数」

第一版用「输出标准差 ≈ 0」判盲区，那是错的 —— 它把「盲区的**实现方式**」
当成了「盲区的**定义**」。实测抓到的反例很干脆：给全盲分量加一点噪声之后
（加噪声是为了不让 `isotonic` 把一条常数拟合到「平均人类评分」上、变成一条
常数完美预言），标准差不再是 0，检测直接失效。

真正的判据是 **ρ(输出, 人类分量评分) ≈ 0**。而且这个判据顺带澄清了一件事：
`vision=0.15` 的分量其 ρ 仍有 0.9 —— **视力弱是尺度问题（校准能修），
全盲是信息问题（谁也修不了）**。两件事必须分开判。

## 只用 anchor，连 train 都不借

盲区与方向两个信号都需要人类分量评分，而只有 anchor 有。holdout 一个数都不借
—— 一旦开了「就借一点点统计量」的口子，下次就是「借个分位数吧」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .providers import ProviderPool, flip_total
from .schema import COMPONENTS, ProviderScore
from .stats import mean, spearman, stdev
from .world import World

__all__ = [
    "NormalizeResult",
    "ProviderDiagnostic",
    "normalize",
    "detect_direction",
    "detect_blind",
]


@dataclass(frozen=True)
class ProviderDiagnostic:
    """归一化的诊断。**「声明 vs 实测」并排放** —— 报告里最有用的一栏。"""

    provider_id: str
    scale: str
    declared_direction: int
    detected_direction: int
    direction_rho: tuple[float, ...]
    direction_undecidable: bool
    blind: tuple[bool, ...]
    blind_components: tuple[str, ...]
    raw_std: tuple[float, ...]
    n_direction_anchors: int

    @property
    def direction_correct(self) -> bool:
        return self.detected_direction == self.declared_direction

    @property
    def coverage(self) -> float:
        return 1.0 - (sum(self.blind) / len(self.blind)) if self.blind else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "scale": self.scale,
            "declared_direction": self.declared_direction,
            "detected_direction": self.detected_direction,
            "direction_correct": self.direction_correct,
            "direction_undecidable": self.direction_undecidable,
            "direction_rho": {k: round(v, 4) for k, v in zip(COMPONENTS, self.direction_rho)},
            "blind": {k: bool(b) for k, b in zip(COMPONENTS, self.blind)},
            "blind_components": list(self.blind_components),
            "raw_std": {k: round(v, 6) for k, v in zip(COMPONENTS, self.raw_std)},
            "coverage": round(self.coverage, 6),
            "n_direction_anchors": self.n_direction_anchors,
        }


@dataclass(frozen=True)
class NormalizeResult:
    """归一化结果。`scores[pid][cid]` 是 `ProviderScore`。"""

    scores: Mapping[str, Mapping[str, ProviderScore]]
    diagnostics: tuple[ProviderDiagnostic, ...]

    def diagnostic(self, pid: str) -> ProviderDiagnostic:
        for d in self.diagnostics:
            if d.provider_id == pid:
                return d
        raise KeyError(f"没有这个 provider 的诊断：{pid}")

    def components_of(self, pid: str, cid: str) -> tuple[float | None, ...]:
        return self.scores[pid][cid].components

    def total_of(self, pid: str, cid: str) -> float:
        return self.scores[pid][cid].raw_total

    def cids(self) -> tuple[str, ...]:
        first = next(iter(self.scores.values()))
        return tuple(sorted(first))

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "n_scored": sum(len(m) for m in self.scores.values()),
        }


# ------------------------------------------------------------------ 检测

def detect_blind(
    raw_by_cid: Mapping[str, tuple[float | None, ...]],
    *,
    tol: float,
    ratings_by_cid: Mapping[str, tuple[float, ...]] | None = None,
    rho_tol: float = 0.15,
    min_anchor: int = 8,
) -> tuple[tuple[bool, ...], tuple[float, ...], tuple[float, ...]]:
    """逐分量判「全盲」。返回 `(blind, 逐分量标准差, 逐分量 ρ)`。

    **两个信号，任一成立即判盲**：

    1. 输出在参考集上的**标准差 ≈ 0**（head 的偏置被冻住，输出是精确常数）；
    2. 输出与 anchor 上**人工分量评分的 Spearman ρ ≈ 0**（有输出，但没有信息）。

    第 2 条是后加的，而且必须加 —— 这是踩出来的坑。第一版只用了第 1 条，
    于是给全盲分量加了一点 `blind_noise` 之后（那是为了不让 isotonic 把常数
    拟合到「平均人类评分」上、变成一条常数完美预言），盲区的标准差就不再是 0，
    检测直接失效：`rule` 在 style 上 vision=0，却被当成一个有信号的分量。
    **「常数」不是盲区的定义，「没有信息」才是。** ρ 才是那个定义。

    ρ 的量级也说明了这件事：一个 `vision=0.15`、`sigma=0` 的分量，其原始输出与
    人类评分的 ρ 仍然高达 0.9 —— 视力弱是**尺度**问题（校准能修），
    全盲才是**信息**问题（谁也修不了）。这两件事必须分开判。
    """
    n_comp = len(COMPONENTS)
    blind: list[bool] = []
    stds: list[float] = []
    rhos: list[float] = []
    for k in range(n_comp):
        col = [row[k] for row in raw_by_cid.values() if row[k] is not None]
        vals = [float(v) for v in col]
        sd = stdev(vals) if len(vals) >= 2 else 0.0
        stds.append(sd)

        rho = 0.0
        measured = False
        if ratings_by_cid:
            xs: list[float] = []
            ys: list[float] = []
            for cid, row in raw_by_cid.items():
                r = ratings_by_cid.get(cid)
                if row[k] is None or r is None:
                    continue
                xs.append(float(row[k]))
                ys.append(float(r[k]))
            if len(xs) >= min_anchor:
                rho = spearman(xs, ys)
                measured = True
        rhos.append(rho)
        # **只有「真算出了 ρ」时 ρ 判据才生效。** 否则缺评分会被读成 ρ=0，
        # 于是「没有证据」被当成「证据是零」—— 一个不传 `ratings_by_cid` 的
        # 调用方会把所有分量判成全盲（实测踩过：对抗用例全红）。
        # ρ 缺失时退回「标准差」这一个判据，那是它唯一能说的话。
        blind.append(sd < tol or (measured and abs(rho) < rho_tol))
    return tuple(blind), tuple(stds), tuple(rhos)


def detect_direction(
    raw_by_cid: Mapping[str, tuple[float | None, ...]],
    ratings_by_cid: Mapping[str, tuple[float, ...]],
    blind: tuple[bool, ...],
    *,
    rho_tol: float,
    min_anchor: int,
    rhos: tuple[float, ...] | None = None,
) -> tuple[int, tuple[float, ...], bool]:
    """逐分量算 Spearman ρ（provider 原始分量 vs 人类分量评分），再加权投票。

    返回 `(方向, 逐分量 ρ, 是否不可判)`。`rhos` 传入时直接复用（`detect_blind`
    已经算过一遍，重算是白算）。

    - 只统计**非盲**且有标注的分量；样本少于 `min_anchor` 直接报「不可判」。
    - **不可判时报 `+1` 而不是抛异常**：方向不可判意味着这个 provider 对整体
      没有贡献，可靠性加权会把它压到 0。抛异常会让一个「吵但没用」的 provider
      把整条流水线炸掉，那不是我们想要的稳健性。
    - 投票权重用 `|ρ|`：一个 ρ=0.7 的分量比 ρ=0.2 的分量更能说明方向。
    """
    n_comp = len(COMPONENTS)
    out_rhos: list[float] = []
    votes: list[float] = []
    for k in range(n_comp):
        if blind[k]:
            out_rhos.append(0.0)
            continue
        if rhos is not None and k < len(rhos):
            rho = float(rhos[k])
        else:
            xs: list[float] = []
            ys: list[float] = []
            for cid, row in raw_by_cid.items():
                if row[k] is None:
                    continue
                r = ratings_by_cid.get(cid)
                if r is None:
                    continue
                xs.append(float(row[k]))
                ys.append(float(r[k]))
            if len(xs) < min_anchor:
                out_rhos.append(0.0)
                continue
            rho = spearman(xs, ys)
        out_rhos.append(rho)
        if abs(rho) > rho_tol:
            votes.append(rho)  # 权重 = |ρ|，符号 = 方向

    if not votes:
        return 1, tuple(out_rhos), True
    score = mean(votes)
    # 平手（score 恰好为 0）按 +1 处理：默认相信声明，只有在**证据反对**时才翻。
    direction = -1 if score < 0 else 1
    return direction, tuple(out_rhos), False


# ------------------------------------------------------------------ 主入口

def normalize(pool: ProviderPool, world: World, cfg: Mapping[str, Any]) -> NormalizeResult:
    """把原始输出归一化。**只读 anchor 与 train**，不碰 holdout。"""
    nz = cfg["normalize"]
    blind_tol = float(nz["blind_tol"])
    blind_rho_tol = float(nz["blind_rho_tol"])
    rho_tol = float(nz["direction_rho_tol"])
    min_anchor = int(nz["min_anchor_for_direction"])

    # 方向与盲区都只用 anchor（有分量评分的那一批）。holdout 一个数都不借。
    anchor_cids = sorted(c.id for c in world.candidates if c.split == "anchor")
    ratings = {cid: world.anchor_ratings[cid] for cid in anchor_cids if cid in world.anchor_ratings}

    scores: dict[str, dict[str, ProviderScore]] = {}
    diagnostics: list[ProviderDiagnostic] = []

    for spec in pool.specs:
        pid = spec.id
        raw_anchor = {cid: pool.output(pid, cid).components for cid in anchor_cids}

        blind, stds, rhos = detect_blind(
            raw_anchor,
            tol=blind_tol,
            ratings_by_cid=ratings,
            rho_tol=blind_rho_tol,
            min_anchor=min_anchor,
        )
        direction, rhos, undecidable = detect_direction(
            raw_anchor, ratings, blind, rho_tol=rho_tol, min_anchor=min_anchor, rhos=rhos
        )

        per: dict[str, ProviderScore] = {}
        all_cids = sorted(pool.outputs[pid])
        for cid in all_cids:
            out = pool.output(pid, cid)
            comps: list[float | None] = []
            for k, v in enumerate(out.components):
                if blind[k] or v is None:
                    # 盲区：标 `None`（看不见），**不是 0**（很差）。
                    comps.append(None)
                elif direction < 0:
                    comps.append(1.0 - float(v))
                else:
                    comps.append(float(v))
            total = out.raw_total if direction > 0 else flip_total(spec.scale, out.raw_total)
            per[cid] = ProviderScore(
                provider_id=pid,
                candidate_id=cid,
                components=tuple(comps),
                raw_total=float(total),
                scale=spec.scale,
                direction=direction,
                self_uncertainty=out.self_uncertainty,
            )
        scores[pid] = per
        diagnostics.append(
            ProviderDiagnostic(
                provider_id=pid,
                scale=spec.scale,
                declared_direction=spec.declared_direction,
                detected_direction=direction,
                direction_rho=rhos,
                direction_undecidable=undecidable,
                blind=blind,
                blind_components=tuple(k for k, b in zip(COMPONENTS, blind) if b),
                raw_std=stds,
                n_direction_anchors=len(anchor_cids),
            )
        )

    return NormalizeResult(scores=scores, diagnostics=tuple(diagnostics))
