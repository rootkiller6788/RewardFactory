"""校准：把「provider 自己的尺度」映射到「与人类标注同尺度」。

**这一步是「不校准就平均」会输的根本原因，也是本项目的主结论来源。**

五个 provider 的原始分住在五个不同的尺度上：`logit` 无界、`saturated` 被压在
(0.17, 0.83)、`ordinal10` 是 1..10 的整数、`grid3` 只有三档。等权平均它们，
方差最大的那个（`rm_small` 的 logit）单方面决定结果 —— 于是「多模型组合」
反而**比最好的单源还差**。这不是理论推演，是本项目 holdout 上的实测。

## 五种方法最终都落成同一张分段线性表

`identity` / `minmax` / `zscore` / `isotonic` / `quantile` 的数学完全不同，
但**产物形态统一**：一张 `[(x, y), …]` 表 + 端点常数外推。理由：

- 产物 `calibration.json` 必须**自足**。拿到它的人不该需要再实现一遍 PAVA
  才能应用它。一个「方法名 + 参数」的产物，等价于要求读者重写我们的代码。
- 一张表让「校准映射单调」这条不变量可以被**逐点断言**（有对抗用例）。

代价是 `zscore` / `quantile` 用 33 点采样近似，引入约 1e-4 量级的插值误差 ——
比 anchor 噪声（σ=0.08）小两个数量级，可以忽略。

## 可靠性必须交叉验证，否则 `isotonic` 会假装无敌

`isotonic` 能把 anchor 上的残差拟合到接近 0（保序回归的自由度就是 anchor 个数）。
用拟合残差算可靠性，`isotonic` 会拿到接近 1 的 ρ，然后在 holdout 上被打回原形。
所以 `mae_after` 一律是 **4 折交叉验证**（按 x 分层切，不是随机切 —— 随机切会让
某一折的 x 范围与拟合集不重叠，PAVA 只能靠端点外推，误差虚高）。

**`mae_fit` 与 `mae_after` 都报**。两者之差就是「这个方法有多容易过拟合 anchor」，
它本身就是一条结论。

## 本模块不许出现 `invert` 这个标识符

方向已经在 `normalize` 里对齐了。这里再看一眼原始方向，就等于把「方向检测」
退化成「读配置」。有对抗用例读源码文本断言这一点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .normalize import NormalizeResult
from .schema import (
    CALIBRATION_METHODS,
    CALIBRATION_NOTES,
    COMPONENTS,
    CalibratedScore,
    CalibrationMap,
)
from .stats import clamp, mean, norm_cdf, quantile, stdev
from .world import World

__all__ = [
    "CalibrateResult",
    "calibrate",
    "fit_map",
    "apply_points",
    "pava",
    "cv_mae",
    "SCALAR_COMPONENT",
]

#: 标量映射在 `maps` 里的键名。它服务的是基线 `calibrated_scalar_mean`
#: —— 「只校准标量、不做分量分解」到底差多少。
SCALAR_COMPONENT = "*"

#: `zscore` / `quantile` 的采样点数。33 点 → 插值误差 ~1e-4。
_GRID = 33

#: 交叉验证折数。4 折在「anchor 只有 80 个」时每折 20 个，够用且不慢。
_FOLDS = 4


def apply_points(points: Sequence[tuple[float, float]], x: float) -> float:
    """分段线性插值 + 端点**常数外推**。与 `CalibrationMap.apply` 同一套语义。

    独立成函数是为了让 `cv_mae` 能用「还没打包成 dataclass 的裸断点」做折内拟合
    —— 让交叉验证走另一条代码路径，是「CV 与实际应用不一致」这类 bug 的温床。
    """
    if not points:
        return 0.5
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid][0] <= x:
            lo = mid
        else:
            hi = mid
    x0, y0 = points[lo]
    x1, y1 = points[hi]
    if x1 <= x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def pava(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float]]:
    """保序回归（Pool-Adjacent-Violators）→ 分段线性表。

    算法：按 x 排序，从左往右扫，一旦发现前一个块的均值大于当前块，就把两块合并
    重算均值，继续往前检查。合并后的块均值单调不减 —— 这就是最优保序拟合。

    输出：每个块贡献两个点 `(x_lo, ȳ)` 与 `(x_hi, ȳ)`；块之间的过渡由
    分段线性插值补成斜坡。于是映射**连续且单调不减**，且在每个块内取常值
    （这正是保序回归解的形状）。

    边界：`n < 2` 或全同 x 时退化成常数映射 `[(x, ȳ)]` —— **不抛异常**。
    「样本不足」是一条正常的代码路径，不是错误。
    """
    if not xs:
        return [(0.0, 0.5), (1.0, 0.5)]
    order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    x = [float(xs[i]) for i in order]
    y = [float(ys[i]) for i in order]
    # 块 = [x_lo, x_hi, 均值, 计数]
    blocks: list[list[float]] = []
    for xi, yi in zip(x, y):
        blocks.append([xi, xi, yi, 1.0])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2] + 1e-15:
            b2 = blocks.pop()
            b1 = blocks.pop()
            n = b1[3] + b2[3]
            blocks.append([b1[0], b2[1], (b1[2] * b1[3] + b2[2] * b2[3]) / n, n])
    pts: list[tuple[float, float]] = []
    for b in blocks:
        pts.append((b[0], b[2]))
        pts.append((b[1], b[2]))
    # 去掉重复 x 上的后一个点（保持 x 非降且不出现同 x 两个 y）
    dedup: list[tuple[float, float]] = []
    for px, py in pts:
        if dedup and px <= dedup[-1][0]:
            if abs(py - dedup[-1][1]) <= 1e-12:
                continue
            dedup[-1] = (dedup[-1][0], py)
            continue
        dedup.append((px, py))
    if len(dedup) == 1:
        dedup.append((dedup[0][0], dedup[0][1]))
    return dedup


def _identity_points(lo: float, hi: float) -> list[tuple[float, float]]:
    return [(lo, lo), (hi, hi)]


def _minmax_points(xs: Sequence[float]) -> list[tuple[float, float]]:
    lo, hi = min(xs), max(xs)
    if hi - lo <= 1e-12:
        # 常数输入：给一个常数映射。**绝不除零** —— 除零会抛 ZeroDivisionError，
        # 而「这个 provider 在这一列上没信号」是个正常的、必须能跑过去的状态。
        return [(lo, 0.5)]
    return [(lo, 0.0), (hi, 1.0)]


def _zscore_points(xs: Sequence[float]) -> list[tuple[float, float]]:
    mu = mean(xs)
    sd = stdev(xs)
    if sd <= 1e-12:
        return [(mu, 0.5)]
    lo = mu - 3.0 * sd
    hi = mu + 3.0 * sd
    return [
        (lo + (hi - lo) * i / (_GRID - 1), norm_cdf((lo + (hi - lo) * i / (_GRID - 1) - mu) / sd))
        for i in range(_GRID)
    ]


def _quantile_points(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float]]:
    """经验分位数映射：`g(x) = Q_y(F_x(x))`。

    与 `isotonic` 的区别值得说清：**分位数映射匹配的是分布**（校准后 provider 的
    分数分布与人类评分的分布逐分位对齐），**保序回归匹配的是条件均值**。
    在「provider 与人类评分的秩一致但尺度非线性」时两者都对；
    在「秩本身不一致」时两者都救不了 —— 那时只能靠组合别的 provider。
    """
    pts: list[tuple[float, float]] = []
    for i in range(_GRID):
        p = i / (_GRID - 1)
        pts.append((quantile(xs, p), quantile(ys, p)))
    dedup: list[tuple[float, float]] = []
    for px, py in pts:
        if dedup and px <= dedup[-1][0]:
            dedup[-1] = (dedup[-1][0], max(dedup[-1][1], py))
            continue
        dedup.append((px, py))
    if len(dedup) == 1:
        dedup.append((dedup[0][0], dedup[0][1]))
    return dedup


def fit_map(
    xs: Sequence[float],
    ys: Sequence[float],
    method: str,
    *,
    clip_lo: float,
    clip_hi: float,
) -> list[tuple[float, float]]:
    """按方法拟合一张分段线性表。**不做交叉验证、不算可靠性** —— 那是 `cv_mae` 的活。"""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"未声明的校准法：{method!r}（可选：{CALIBRATION_METHODS}）")
    if len(xs) < 2:
        return _identity_points(clip_lo, clip_hi)
    if method == "identity":
        return _identity_points(clip_lo, clip_hi)
    if method == "minmax":
        return _minmax_points(xs)
    if method == "zscore":
        return _zscore_points(xs)
    if method == "isotonic":
        return pava(xs, ys)
    return _quantile_points(xs, ys)


def _mae(points: Sequence[tuple[float, float]], xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return mean([abs(apply_points(points, float(x)) - float(y)) for x, y in zip(xs, ys)])


def cv_mae(
    xs: Sequence[float],
    ys: Sequence[float],
    method: str,
    *,
    clip_lo: float,
    clip_hi: float,
    folds: int = _FOLDS,
) -> tuple[float, int]:
    """按 x **分层**切折的交叉验证 MAE。

    分层（而不是随机）切：随机切会让某一折的 x 范围落在拟合集的 x 范围之外，
    于是 `isotonic` 只能靠端点常数外推，CV 误差被系统性抬高，
    `isotonic` 会被冤判成「不如 minmax」。**分层切保证每一折的 x 覆盖是完整的。**

    返回 `(MAE, 实际折数)`。样本不足以切折时返回全量 MAE 且折数为 1。
    """
    n = len(xs)
    if n < folds * 2:
        return _mae(fit_map(xs, ys, method, clip_lo=clip_lo, clip_hi=clip_hi), xs, ys), 1
    order = sorted(range(n), key=lambda i: xs[i])
    errs: list[float] = []
    used = 0
    for f in range(folds):
        hold = {order[i] for i in range(f, n, folds)}
        tr_x = [float(xs[i]) for i in range(n) if i not in hold]
        tr_y = [float(ys[i]) for i in range(n) if i not in hold]
        te = [i for i in order if i in hold]
        if not tr_x or not te:
            continue
        pts = fit_map(tr_x, tr_y, method, clip_lo=clip_lo, clip_hi=clip_hi)
        errs.extend(abs(apply_points(pts, float(xs[i])) - float(ys[i])) for i in te)
        used += 1
    if not errs:
        return _mae(fit_map(xs, ys, method, clip_lo=clip_lo, clip_hi=clip_hi), xs, ys), 1
    return mean(errs), used


@dataclass(frozen=True)
class CalibrateResult:
    """校准结果。

    `maps[(pid, component)]` 与 `maps[(pid, "*")]`（标量映射）。
    `scores[pid][cid]` 是校准后的分量分 + 该 provider 的逐分量可靠性。
    """

    method: str
    maps: Mapping[tuple[str, str], CalibrationMap]
    scores: Mapping[str, Mapping[str, CalibratedScore]]
    reliability: Mapping[str, tuple[float, ...]]
    anchor_scalar: Mapping[str, float]

    def map_of(self, pid: str, component: str) -> CalibrationMap:
        return self.maps[(pid, component)]

    def reliability_of(self, pid: str) -> tuple[float, ...]:
        return self.reliability[pid]

    def components_of(self, pid: str, cid: str) -> tuple[float | None, ...]:
        return self.scores[pid][cid].components

    def cal_total_of(self, pid: str, cid: str) -> float:
        return self.scores[pid][cid].cal_total

    def cids(self) -> tuple[str, ...]:
        first = next(iter(self.scores.values()))
        return tuple(sorted(first))

    @property
    def mean_mae_before(self) -> float:
        ms = [m.mae_before for m in self.maps.values() if not m.blind]
        return mean(ms) if ms else 0.0

    @property
    def mean_mae_after(self) -> float:
        ms = [m.mae_after for m in self.maps.values() if not m.blind]
        return mean(ms) if ms else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "note": CALIBRATION_NOTES.get(self.method, ""),
            "mean_mae_before": round(self.mean_mae_before, 6),
            "mean_mae_after": round(self.mean_mae_after, 6),
            "maps": [
                m.to_dict()
                for _k, m in sorted(self.maps.items(), key=lambda kv: (kv[0][0], kv[0][1]))
            ],
            "reliability": {
                pid: {k: round(v, 4) for k, v in zip(COMPONENTS, rel)}
                for pid, rel in sorted(self.reliability.items())
            },
        }


def calibrate(
    world: World,
    nz: NormalizeResult,
    cfg: Mapping[str, Any],
    *,
    method: str | None = None,
) -> CalibrateResult:
    """在 anchor 上拟合校准映射，再把**全部**候选校准一遍。

    `method` 可以覆盖配置（消融用）。**注意消融只改方法，不改方向检测与盲区检测**
    —— 「不校准」与「不检测方向」是两件事，把它们绑在一起消融会分不清是谁的功劳。
    """
    cal = cfg["calibrate"]
    m = str(method or cal["method"])
    clip_lo, clip_hi = float(cal["clip_lo"]), float(cal["clip_hi"])
    min_anchor = int(cal["min_anchor"])
    slope = float(cal["reliability_slope"])
    r_floor, r_ceil = float(cal["reliability_floor"]), float(cal["reliability_ceiling"])

    anchor_cids = sorted(cid for cid in world.anchor_ratings)
    # anchor 上的「人类标量」= 分量评分的等权均值。**它是人类给的**，
    # 不是我们算出来的真值 —— 标量基线必须靠它，否则就是偷看真值。
    anchor_scalar = {
        cid: mean(list(world.anchor_ratings[cid])) for cid in anchor_cids
    }

    maps: dict[tuple[str, str], CalibrationMap] = {}
    scores: dict[str, dict[str, CalibratedScore]] = {}
    reliability: dict[str, tuple[float, ...]] = {}

    for diag in nz.diagnostics:
        pid = diag.provider_id
        rel: list[float] = []
        per: dict[str, CalibratedScore] = {}
        cal_by_cid: dict[str, list[float | None]] = {cid: [] for cid in sorted(nz.scores[pid])}

        for k, comp in enumerate(COMPONENTS):
            blind = diag.blind[k]
            xs: list[float] = []
            ys: list[float] = []
            if not blind:
                for cid in anchor_cids:
                    v = nz.scores[pid][cid].components[k]
                    if v is None:
                        continue
                    xs.append(float(v))
                    ys.append(float(world.anchor_ratings[cid][k]))

            if blind or len(xs) < min_anchor:
                # 盲区 / 样本不足：给一张常数映射。**不抛异常** ——
                # 「这个 (provider, 分量) 不可用」是一条必须能走下去的路径，
                # 而 `reliability=0` 会让它在聚合时自动出局。
                pts = [(0.0, 0.5), (1.0, 0.5)]
                cm = CalibrationMap(
                    provider_id=pid,
                    component=comp,
                    method=m,
                    points=tuple(pts),
                    n_anchor=len(xs),
                    mae_before=0.5,
                    mae_after=0.5,
                    reliability=0.0,
                    mae_fit=0.5,
                    n_folds=1,
                    blind=True,
                )
                maps[(pid, comp)] = cm
                rel.append(0.0)
                continue

            pts = fit_map(xs, ys, m, clip_lo=clip_lo, clip_hi=clip_hi)
            mae_fit = _mae(pts, xs, ys)
            mae_cv, folds = cv_mae(xs, ys, m, clip_lo=clip_lo, clip_hi=clip_hi)
            # 「不校准」的对照：identity 的 MAE 是无参数的，直接在全量 anchor 上算。
            mae_before = _mae(_identity_points(clip_lo, clip_hi), xs, ys)
            rho = clamp(1.0 - slope * mae_cv, r_floor, r_ceil)
            maps[(pid, comp)] = CalibrationMap(
                provider_id=pid,
                component=comp,
                method=m,
                points=tuple(pts),
                n_anchor=len(xs),
                mae_before=mae_before,
                mae_after=mae_cv,
                reliability=rho,
                mae_fit=mae_fit,
                n_folds=folds,
            )
            rel.append(rho)

        # ---- 标量映射（服务 `calibrated_scalar_mean` 基线）
        sx: list[float] = []
        sy: list[float] = []
        for cid in anchor_cids:
            sx.append(nz.scores[pid][cid].raw_total)
            sy.append(anchor_scalar[cid])
        spts = fit_map(sx, sy, m, clip_lo=clip_lo, clip_hi=clip_hi)
        smae_cv, sfolds = cv_mae(sx, sy, m, clip_lo=clip_lo, clip_hi=clip_hi)
        maps[(pid, SCALAR_COMPONENT)] = CalibrationMap(
            provider_id=pid,
            component=SCALAR_COMPONENT,
            method=m,
            points=tuple(spts),
            n_anchor=len(sx),
            mae_before=_mae(_identity_points(clip_lo, clip_hi), sx, sy),
            mae_after=smae_cv,
            reliability=clamp(1.0 - slope * smae_cv, r_floor, r_ceil),
            mae_fit=_mae(spts, sx, sy),
            n_folds=sfolds,
        )

        # ---- 应用到全部候选
        for cid in sorted(nz.scores[pid]):
            src = nz.scores[pid][cid].components
            vals: list[float | None] = []
            for k, comp in enumerate(COMPONENTS):
                if src[k] is None or maps[(pid, comp)].blind:
                    vals.append(None)
                else:
                    vals.append(clamp(maps[(pid, comp)].apply(float(src[k])), clip_lo, clip_hi))
            vis = [v for v in vals if v is not None]
            per[cid] = CalibratedScore(
                provider_id=pid,
                candidate_id=cid,
                components=tuple(vals),
                cal_total=mean(vis) if vis else 0.5,
                reliability=tuple(rel),
            )
            cal_by_cid[cid] = vals
        scores[pid] = per
        reliability[pid] = tuple(rel)

    return CalibrateResult(
        method=m,
        maps=maps,
        scores=scores,
        reliability=reliability,
        anchor_scalar=anchor_scalar,
    )
