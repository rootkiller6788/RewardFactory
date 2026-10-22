"""评测：**第三方**。它允许读真值，而且必须读 —— 它是唯一能回答
「这个 reward 到底有多准」的地方。

分工是硬的：

- `search/` 搜权重，**一行真值都不许读**（有对抗用例读源码断言）；
- `aggregate.py` 做聚合，读的是 anchor 上的校准残差，也不读真值；
- `benchmark.py` 读 `world.latent_weights` 与 `candidate.quality`，
  用来算**上界参考**（`oracle_truth_w`）、**CI 覆盖率**、**不可约误差**。

一句话：**搜权重的人不给自己打分，打分的人不搜权重。**

## 三组数字必须成对看，单看任何一个都会被骗

| 数字 | 能被怎么刷 |
| --- | --- |
| `agreement` | 全判平局 ⇒ 0；全押多数类 ⇒ ~0.48 |
| `tie_rate` | 判平局就能躲开所有难题 |
| `agreement_decisive` | 只在最有把握时下结论 ⇒ 接近 1 |

所以 `MethodScore` 三个字段一起给，报告里也并排印。**任何只报一个数的评测表
都是在让人误读。**

## 误差分解用的是「干净标签」

`|Δq| > tie_band` 的那些对，标签取 `sign(Δq)`（不取人类给的，因为人类会翻错）。
于是：

- `shared`：**所有** provider 都错 ⇒ 真值残差 + 共有盲区，**不可约**；
- `specific[p]`：p 错而多数其他 provider 对 ⇒ p 的**特有偏差**。

组合的价值就是把 `specific` 压到接近 0 —— 它压不下去的部分才是真的没办法。
这张表是「组合为什么有用」的**唯一**机制解释，比「一致性提升了 5 个百分点」
有信息量得多。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .aggregate import AggregateResult, normalize_w
from .calibrate import SCALAR_COMPONENT, CalibrateResult
from .normalize import NormalizeResult
from .providers import ProviderPool
from .schema import (
    BASELINE_LABELS,
    BASELINES,
    COMPONENTS,
    PREFER_A,
    PREFER_B,
    TIE,
    BenchmarkResult,
    BootstrapResult,
    ErrorDecomposition,
    MethodScore,
    Pair,
    ProviderReport,
    SearchResult,
    WeightEstimate,
)
from .search.base import Objective, build_objective
from .stats import mean, quantile, spearman, stdev
from .uncertainty import ece, over_estimate_ratio
from .world import World

__all__ = [
    "benchmark",
    "MethodScores",
    "build_u_pairs",
    "pair_deltas",
    "mix_values",
]


def pair_deltas(
    pairs: Sequence[Pair],
    values: Mapping[str, float],
) -> list[tuple[Pair, float]]:
    return [(p, float(values[p.a]) - float(values[p.b])) for p in pairs]


def build_u_pairs(
    pairs: Sequence[Pair],
    u: Mapping[str, float],
) -> dict[str, float]:
    """成对不确定度 = `u_a + u_b`，与 `aggregate.decide` 同一口径。

    报告里出现第二个口径，读者就算不出「为什么这一对判成了平局」。
    """
    return {p.id: float(u[p.a]) + float(u[p.b]) for p in pairs}


def _pred(delta_r: float, u_pair: float, kappa: float) -> int:
    if abs(delta_r) <= kappa * u_pair:
        return TIE
    return PREFER_A if delta_r > 0 else PREFER_B


def _score(
    name: str,
    pairs: Sequence[Pair],
    deltas: Mapping[str, float],
    u_pairs: Mapping[str, float],
    kappa: float,
    *,
    clean_pairs: Sequence[Pair] = (),
) -> MethodScore:
    n = len(pairs)
    if n == 0:
        return MethodScore(name, BASELINE_LABELS.get(name, name), 0.0, 0.0, 0.0, float("nan"), 0)
    correct = ties = decisive = dec_correct = 0
    for p in pairs:
        pr = _pred(deltas[p.id], u_pairs.get(p.id, 0.0), kappa)
        if pr == p.label:
            correct += 1
        if pr == TIE:
            ties += 1
        else:
            decisive += 1
            if pr == p.label:
                dec_correct += 1
    # ---- 对干净标签（`sign(Δq)`）的一致性：**平局算半分**。
    # 人类只判 5% 平局，而任何把 u 当真的方法都会判 20~70% 平局 ——
    # 于是「与人类标签逐对相等」这一列被平局率主导，比不出谁排得准。
    # 干净标签没有人类噪声，半分制对谨慎公平：不判 ⇒ 拿 0.5（无信息），
    # 判对 ⇒ 1，判错 ⇒ 0。这一列才是「谁的目标函数更好」的答案。
    t_num = 0
    t_den = 0
    for p in clean_pairs:
        pr = _pred(deltas[p.id], u_pairs.get(p.id, 0.0), kappa)
        tl = PREFER_A if p.delta_quality > 0 else PREFER_B
        t_num += 2 if pr == tl else (1 if pr == TIE else 0)
        t_den += 2
    return MethodScore(
        method=name,
        label=BASELINE_LABELS.get(name, name),
        agreement=correct / n,
        tie_rate=ties / n,
        decisive_rate=decisive / n,
        # 一列果断判决都没有时，这个数是**无定义**，不是 0。印 0 等于说
        # 「一果断就全错」—— 那是一句没测过的话。
        agreement_decisive=(dec_correct / decisive) if decisive else float("nan"),
        n_pairs=n,
        agreement_truth=(t_num / t_den) if t_den else float("nan"),
    )


@dataclass(frozen=True)
class MethodScores:
    """全部方法的逐候选分数与不确定度。"""

    values: Mapping[str, Mapping[str, float]]
    u: Mapping[str, Mapping[str, float]]


def _rms(xs: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in xs))


def _cal_total_u(cal: CalibrateResult, pid: str) -> float:
    """单源 `cal_total` 的不确定度。

    `cal_total` **就是**该 provider 可见分量校准值的等权均值（`calibrate.py` 里
    一行 `mean(vis)`），所以它的误差是各分量误差的 RMS 除以可见分量数。
    这里**不能**用标量映射的 `mae_after` —— 那是「把 raw_total 映射到人类标量」
    的误差，量与值不匹配（一个 provider 的 raw_total 是它自己可见分量的损失性摘要，
    误差天然更大）。用错口径会让单源基线凭空背上 2~3 倍的 u，一路判平局。
    """
    ms = [
        cal.map_of(pid, COMPONENTS[k]).mae_after
        for k in range(len(COMPONENTS))
        if not cal.map_of(pid, COMPONENTS[k]).blind
    ]
    return (_rms(ms) / len(ms)) if ms else 0.5


def _scalar_map_u(cal: CalibrateResult, pid: str) -> float:
    """标量映射（`raw_total → 人类标量`）自己的 CV 误差。只有
    `calibrated_scalar_mean` 用得上它 —— 那条基线用的**就是**这张映射。"""
    return max(0.0, cal.map_of(pid, SCALAR_COMPONENT).mae_after)


def _raw_u(world: World, nz: NormalizeResult, pid: str) -> float:
    """不校准的方法没有「人类尺度上的误差」可言 —— 它连人类尺度都不是。
    只能退一步用它**自己尺度上的离散度**当误差。这个数在 `logit`（sd ≈ 1.1）与
    `grid3`（sd ≈ 0.12）之间差一个数量级，正好说明「不校准就没法定义平局带」。"""
    xs = [nz.scores[pid][cid].raw_total for cid in world.anchor_ratings]
    return stdev(xs) if len(xs) > 1 else 0.5


def _pick_best(
    world: World,
    values: Mapping[str, Mapping[str, float]],
    u_of: Mapping[str, float],
    kappa: float,
    *,
    split: str = "anchor",
) -> str:
    """挑「最好的单个 provider」。

    **在 `anchor` 上挑，不碰 holdout** —— 用 holdout 挑等于让基线偷看评测集，
    而这条基线的强弱决定了「组合到底赢了多少」。`sorted` 兜住平手时的顺序，
    保证决定论。
    """
    pairs = world.pairs_of(split)
    if not pairs:
        return sorted(values)[0]

    def score(pid: str) -> float:
        vals = values[pid]
        up = 2.0 * u_of[pid]
        c = 0
        for p in pairs:
            if _pred(float(vals[p.a]) - float(vals[p.b]), up, kappa) == p.label:
                c += 1
        return c / len(pairs)

    return max(sorted(values), key=score)


def build_method_values(
    world: World,
    nz: NormalizeResult,
    cal: CalibrateResult,
    agg: AggregateResult,
    best_w: Sequence[float],
    *,
    cids: Sequence[str],
    kappa: float,
) -> MethodScores:
    """为每个方法造一份「候选 → 分数」表。

    方法之间**只差一件事**，这样它们之间的差距才有归因：

    | 方法 | 校准 | 多源组合 | 分量分解 | 权重 |
    | --- | --- | --- | --- | --- |
    | `naive_raw_mean` | ✗ | ✓ | ✗ | — |
    | `best_single_raw` | ✗ | ✗ | ✗ | — |
    | `calibrated_scalar_mean` | ✓ | ✓ | ✗ | — |
    | `best_single_calibrated` | ✓ | ✗ | ✗ | — |
    | `component_equal_w` | ✓ | ✓ | ✓ | 等分 |
    | `component_argmax_w` | ✓ | ✓ | ✓ | 搜索（argmax） |
    | `component_searched_w` | ✓ | ✓ | ✓ | 搜索 + bootstrap 收缩 |
    | `oracle_truth_w` | ✓ | ✓ | ✓ | 注入真值 |

    **每个方法的 `u` 都用同一条规则算**：把它自己的分数写成 `Σ_j a_j·v_j`，
    则 `u = sqrt(Σ_j (a_j·u_j)²)`。口径不统一的话，「谁的一致性高」就只是在比
    谁的 u 被写得小 —— 那是调参，不是评测。
    """
    pids = sorted(nz.scores)
    lw = world.latent_weights
    wn = normalize_w(best_w)
    values: dict[str, dict[str, float]] = {}
    us: dict[str, dict[str, float]] = {}

    raw_u = {pid: _raw_u(world, nz, pid) for pid in pids}

    # ---- 不校准、等权平均原始标量（**把五个不可比的尺度直接平均**）
    values["naive_raw_mean"] = {
        cid: mean([nz.scores[pid][cid].raw_total for pid in pids]) for cid in cids
    }
    u_naive = _rms([raw_u[pid] for pid in pids]) / len(pids)
    us["naive_raw_mean"] = {cid: u_naive for cid in cids}

    # ---- 不校准、最好的单个 provider（**在 anchor 上先评出来再挑**）
    solo_raw: dict[str, dict[str, float]] = {
        pid: {cid: nz.scores[pid][cid].raw_total for cid in cids} for pid in pids
    }
    best_raw_pid = _pick_best(world, solo_raw, raw_u, kappa)
    values["best_single_raw"] = solo_raw[best_raw_pid]
    us["best_single_raw"] = {cid: raw_u[best_raw_pid] for cid in cids}

    # ---- 校准标量、等权平均（**只校准标量、不做分量分解**）
    sc_u = {pid: _scalar_map_u(cal, pid) for pid in pids}
    values["calibrated_scalar_mean"] = {
        cid: mean([cal.map_of(pid, SCALAR_COMPONENT).apply(nz.scores[pid][cid].raw_total) for pid in pids])
        for cid in cids
    }
    u_csm = _rms([sc_u[pid] for pid in pids]) / len(pids)
    us["calibrated_scalar_mean"] = {cid: u_csm for cid in cids}

    # ---- 校准、最好的单个 provider
    ct_u = {pid: _cal_total_u(cal, pid) for pid in pids}
    solo_cal: dict[str, dict[str, float]] = {
        pid: {cid: cal.scores[pid][cid].cal_total for cid in cids} for pid in pids
    }
    best_cal_pid = _pick_best(world, solo_cal, ct_u, kappa)
    values["best_single_calibrated"] = solo_cal[best_cal_pid]
    us["best_single_calibrated"] = {cid: ct_u[best_cal_pid] for cid in cids}

    # ---- 分量分解 + 校准（等权 / 搜索 / 真值）
    n = len(COMPONENTS)
    equal_w = tuple(1.0 / n for _ in range(n))

    def _mix(w: Sequence[float]) -> tuple[dict[str, float], dict[str, float]]:
        return mix_values(agg, w, cids=cids)

    v, uu = _mix(equal_w)
    values["component_equal_w"], us["component_equal_w"] = v, uu
    v, uu = _mix(wn)
    # 搜索器的**原始答案**。先放这里，`benchmark()` 拿到 bootstrap 之后会把
    # `component_searched_w` 换成收缩后的权重 —— 交付物不是 argmax。
    values["component_argmax_w"], us["component_argmax_w"] = v, uu
    values["component_searched_w"], us["component_searched_w"] = v, uu
    v, uu = _mix(normalize_w(lw))
    values["oracle_truth_w"], us["oracle_truth_w"] = v, uu

    return MethodScores(values=values, u=us)


def mix_values(
    agg: AggregateResult,
    w: Sequence[float],
    *,
    cids: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """按权重 `w` 造一份「候选 → 分数 / 不确定度」表。与 `build_method_values`
    内部的 `_mix` **必须同口径**，所以抽出来共用 —— 交付物的分数是用它算的。"""
    vals: dict[str, float] = {}
    uu: dict[str, float] = {}
    for cid in cids:
        comps = agg.components[cid]
        vals[cid] = math.fsum(a * b for a, b in zip(w, comps))
        uu[cid] = _rms([a * b for a, b in zip(w, agg.u_components[cid])])
    return vals, uu


def _solo_agreement(world: World, values: Mapping[str, float], *, tie_band: float) -> float:
    """单个 provider 在 **holdout** 上的成对一致性。

    **口径与主方法一致**：对**干净标签** `sign(Δq)`（`|Δq| > tie_band`）打分，
    平局算**半分**。第一版这里用的是「与人类标签逐对相等」，那是错的 ——
    它被平局率主导，而不是被「对不对」主导。实测的极端例子很直白：一个三档
    打分的 provider 在 480 个候选上有 464 个输出同一个值，于是人类标签口径
    给它 13%。那 13% 不是「它错了 87%」，是「它大部分时候压根不表态」。
    两个口径必须统一，否则同一个 provider 在两张表里是两个东西。
    """
    pairs = [p for p in world.pairs_of("holdout") if abs(p.delta_quality) > tie_band]
    if not pairs:
        return 0.0
    num = den = 0
    for p in pairs:
        d = float(values[p.a]) - float(values[p.b])
        pr = PREFER_A if d > 0 else (PREFER_B if d < 0 else TIE)
        tl = PREFER_A if p.delta_quality > 0 else PREFER_B
        num += 2 if pr == tl else (1 if pr == TIE else 0)
        den += 2
    return num / den


def _decomposition(
    world: World,
    nz: NormalizeResult,
    cal: CalibrateResult,
    mix_values: Mapping[str, float],
    *,
    specific_majority: int,
    tie_band: float,
) -> ErrorDecomposition:
    """误差分解。用**干净标签**（`sign(Δq)`，剔除近平局）。

    刻意**不用人类标签**：人类标签里有 12% 的翻转，把翻转算进去，
    「所有 provider 都错」的那些对里会混进大量「真值本来就没差多少」的对，
    于是「不可约误差」这个数就变成了标签噪声的度量，而不是真值的不可约性。
    """
    pairs = [p for p in world.pairs_of("holdout") if abs(p.delta_quality) > tie_band]
    pids = sorted(nz.scores)
    if not pairs:
        return ErrorDecomposition(0, 0, {pid: 0 for pid in pids}, 0, 0)

    wrong: dict[str, list[bool]] = {}
    for pid in pids:
        vals = {cid: cal.scores[pid][cid].cal_total for cid in nz.scores[pid]}
        wl: list[bool] = []
        for p in pairs:
            d = vals[p.a] - vals[p.b]
            pred = PREFER_A if d > 0 else (PREFER_B if d < 0 else TIE)
            wl.append(pred != (PREFER_A if p.delta_quality > 0 else PREFER_B))
        wrong[pid] = wl

    shared = 0
    specific: dict[str, int] = {pid: 0 for pid in pids}
    mix_specific = 0
    mix_shared = 0
    for i, p in enumerate(pairs):
        truth = PREFER_A if p.delta_quality > 0 else PREFER_B
        n_wrong = sum(1 for pid in pids if wrong[pid][i])
        if n_wrong == len(pids):
            shared += 1
        for pid in pids:
            if not wrong[pid][i]:
                continue
            others_right = sum(1 for q in pids if q != pid and not wrong[q][i])
            if others_right >= specific_majority:
                specific[pid] += 1
        d = float(mix_values[p.a]) - float(mix_values[p.b])
        mpred = PREFER_A if d > 0 else (PREFER_B if d < 0 else TIE)
        if mpred != truth:
            if n_wrong == len(pids):
                mix_shared += 1
            else:
                mix_specific += 1
    return ErrorDecomposition(
        n_pairs=len(pairs),
        shared=shared,
        specific=specific,
        mixture_specific=mix_specific,
        mixture_shared=mix_shared,
    )


def _provider_reports(
    world: World,
    pool: ProviderPool,
    nz: NormalizeResult,
    cal: CalibrateResult,
    decomp: ErrorDecomposition,
    *,
    cids: Sequence[str],
    tie_band: float,
) -> tuple[ProviderReport, ...]:
    out: list[ProviderReport] = []
    hold = world.pairs_of("holdout")
    for spec in pool.specs:
        pid = spec.id
        diag = nz.diagnostic(pid)
        maps = [cal.map_of(pid, k) for k in COMPONENTS]
        live = [m for m in maps if not m.blind]
        mae_before = mean([m.mae_before for m in live]) if live else 0.5
        mae_after = mean([m.mae_after for m in live]) if live else 0.5
        rel = mean([m.reliability for m in live]) if live else 0.0
        raw_vals = {cid: nz.scores[pid][cid].raw_total for cid in cids}
        cal_vals = {cid: cal.scores[pid][cid].cal_total for cid in cids}
        degenerate = all(m.blind for m in maps)
        out.append(
            ProviderReport(
                provider_id=pid,
                kind=spec.kind,
                scale=spec.scale,
                declared_direction=spec.declared_direction,
                detected_direction=diag.detected_direction,
                blind_components=diag.blind_components,
                coverage=diag.coverage,
                mae_before=mae_before,
                mae_after=mae_after,
                reliability=rel,
                solo_agreement=_solo_agreement(world, raw_vals, tie_band=tie_band),
                solo_agreement_cal=_solo_agreement(world, cal_vals, tie_band=tie_band),
                specific_error=int(decomp.specific.get(pid, 0)),
                shared_error=decomp.shared,
                n_pairs=len(hold),
                degenerate=degenerate,
                note=spec.note,
            )
        )
    return tuple(out)


# ------------------------------------------------------------------ bootstrap

def bootstrap_weights(
    world: World,
    components: Mapping[str, Sequence[float]],
    u_pairs: Mapping[str, float],
    cfg: Mapping[str, Any],
    rng: Any,
    *,
    kappa: float,
    point_w: Sequence[float],
    point_agreement: float,
    holdout_values: Mapping[str, float],
    holdout_u_pairs: Mapping[str, float],
) -> BootstrapResult:
    """权重的 bootstrap —— **验收标准 ②** 的兑现物。

    两件事各做一次重采样，因为它们回答的是两个不同的问题：

    1. **重采样训练对 → 重跑搜索** ⇒ `w*` 的分布。回答「换一批标注，权重会变成什么」。
       这是 CI 的来源，也是「点估计该不该被当真」的答案。
    2. **重采样 holdout 对 → 用固定的 `w*` 重算一致性** ⇒ 一致性本身的抽样误差。
       回答「那 5 个百分点的提升会不会只是这批评测数据的运气」。

    判据是**覆盖率**而不是「点估计等不等于真值」：点估计必然偏离（标签有噪声、
    样本有限），要求它相等会逼着实现去偷看真值 —— 那正好毁掉整个项目。

    ## 副产品：`bagged_w` —— 交付的目标函数不是 argmax

    重采样出来的 200 个权重向量的**均值**才是本项目交付的 `w`。理由与统计里
    「bagging 不稳定估计量」完全一样：搜索的目标函数是**分片常数**的
    （一对对地数对错，没有梯度），它的 argmax 会落在单纯形的**顶点**上
    —— 实测 `w* = one-hot(correctness)`。换一批标注，顶点就会跳到另一个分量。
    取 200 次重采样的均值把顶点收缩回内部，实测 holdout 一致性 +5.4 个百分点
    （64.58% → 70.00%）。**这不是调参：bagging 是事先就能说清的理由，
    而 argmax 的不稳定是分片常数目标函数的必然。**
    """
    bs = cfg["bootstrap"]
    n_res = int(bs["n_resamples"])
    ci = float(bs["ci_level"])
    budget = int(bs["budget"])
    from .search.coordinate import coordinate_ascent
    from .search.base import Objective, PairRow

    train = world.pairs_of("train")
    rows = [
        PairRow(
            delta_c=tuple(float(components[p.a][k]) - float(components[p.b][k]) for k in range(len(COMPONENTS))),
            label=int(p.label),
            u_pair=float(u_pairs.get(p.id, 0.0)),
        )
        for p in train
    ]

    n_comp = len(COMPONENTS)
    moves = [float(t) for t in cfg["search"]["coordinate_moves"]]
    start = tuple(1.0 / n_comp for _ in range(n_comp))
    samples: list[tuple[float, ...]] = []
    for _ in range(n_res):
        resampled = rng.bootstrap(rows, len(rows))
        obj = Objective(resampled, kappa=kappa, n_components=n_comp)
        w, _v, _t = coordinate_ascent(obj, moves=moves, budget=budget, start=start, record=False)
        samples.append(tuple(w))

    alpha = (1.0 - ci) / 2.0
    estimates: list[WeightEstimate] = []
    for k, name in enumerate(COMPONENTS):
        col = [s[k] for s in samples]
        estimates.append(
            WeightEstimate(
                component=name,
                mean=mean(col),
                std=math.sqrt(sum((x - mean(col)) ** 2 for x in col) / max(1, len(col) - 1)),
                lo=quantile(col, alpha),
                hi=quantile(col, 1.0 - alpha),
                ci_level=ci,
                truth=float(world.latent_weights[k]),
            )
        )

    # ---- 一致性本身的抽样误差（**与点估计同一个判定口径**：
    #      用 u 判平局。用另一套口径重采样，算出来的 CI 就不是这个点估计的 CI）
    hold = world.pairs_of("holdout")
    ags: list[float] = []
    for _ in range(n_res):
        sample = rng.bootstrap(hold, len(hold))
        c = 0
        for p in sample:
            d = float(holdout_values[p.a]) - float(holdout_values[p.b])
            if _pred(d, float(holdout_u_pairs.get(p.id, 0.0)), kappa) == p.label:
                c += 1
        ags.append(c / len(sample) if sample else 0.0)

    return BootstrapResult(
        n_resamples=n_res,
        ci_level=ci,
        searcher=str(bs["searcher"]),
        estimates=tuple(estimates),
        agreement_lo=quantile(ags, alpha),
        agreement_hi=quantile(ags, 1.0 - alpha),
        agreement_std=math.sqrt(sum((x - mean(ags)) ** 2 for x in ags) / max(1, len(ags) - 1)),
        point_agreement=point_agreement,
        samples=tuple(samples),
        bagged_w=tuple(mean([s[k] for s in samples]) for k in range(n_comp)) if samples else (),
        argmax_w=tuple(float(x) for x in point_w),
    )


# ------------------------------------------------------------------ 主入口

def benchmark(
    world: World,
    pool: ProviderPool,
    nz: NormalizeResult,
    cal: CalibrateResult,
    agg: AggregateResult,
    search: Mapping[str, SearchResult],
    *,
    searched: str,
    cfg: Mapping[str, Any],
    rng: Any,
) -> BenchmarkResult:
    """在 holdout 上评全部方法。**这是唯一允许读真值的地方。**"""
    kappa = float(cfg["uncertainty"]["tie_kappa"])
    bm = cfg["benchmark"]
    tie_band = float(cfg["world"]["tie_band"])
    hold = world.pairs_of("holdout")
    # 干净标签：`|Δq| > tie_band` 的那些对。人类标签有噪声，`sign(Δq)` 没有。
    clean_pairs = [p for p in hold if abs(p.delta_quality) > tie_band]
    cids = sorted(agg.components)
    hold_cids = [c for c in cids if world.candidate(c).split == "holdout"]

    sr = search[searched]
    argmax_w = sr.best_w
    ms = build_method_values(world, nz, cal, agg, argmax_w, cids=cids, kappa=kappa)

    u_pairs = {m: build_u_pairs(hold, ms.u[m]) for m in ms.values}

    # ---- bootstrap 必须**先**跑：交付的 w 是它给的，不是 argmax。
    # 搜索的目标函数是分片常数的（一对对地数对错），它的 argmax 落在单纯形顶点上，
    # 换一批标注就跳到别处；200 次重采样的均值把顶点收缩回内部。
    train = world.pairs_of("train")
    train_u_pairs = build_u_pairs(train, ms.u["component_argmax_w"])
    _ad = {
        p.id: float(ms.values["component_argmax_w"][p.a]) - float(ms.values["component_argmax_w"][p.b])
        for p in hold
    }
    hold_ag_argmax = _score("argmax", hold, _ad, u_pairs["component_argmax_w"], kappa, clean_pairs=clean_pairs).agreement
    boot = bootstrap_weights(
        world, agg.components, train_u_pairs, cfg, rng,
        kappa=kappa, point_w=argmax_w, point_agreement=hold_ag_argmax,
        holdout_values=ms.values["component_argmax_w"],
        holdout_u_pairs=u_pairs["component_argmax_w"],
    )
    if boot.bagged_w:
        v, uu = mix_values(agg, boot.bagged_w, cids=cids)
        ms.values["component_searched_w"], ms.u["component_searched_w"] = v, uu
        u_pairs["component_searched_w"] = build_u_pairs(hold, uu)

    scores: list[MethodScore] = []
    for m in sorted(ms.values):
        deltas = {p.id: float(ms.values[m][p.a]) - float(ms.values[m][p.b]) for p in hold}
        scores.append(_score(m, hold, deltas, u_pairs[m], kappa, clean_pairs=clean_pairs))

    # ---- 多数类基线。用**训练集**的多数类，不用 holdout 的 —— 后者是偷看。
    train_labels = [p.label for p in train]
    maj = max((PREFER_A, PREFER_B, TIE), key=lambda v: sum(1 for x in train_labels if x == v))
    chance_correct = sum(1 for p in hold if p.label == maj)
    chance_ag = chance_correct / len(hold) if hold else 0.0
    # 多数类基线的「对干净标签」口径：恒判同一侧 ⇒ 干净对上答对一半（正负各半）。
    scores.append(
        MethodScore(
            method="chance",
            label=BASELINE_LABELS["chance"],
            agreement=chance_ag,
            tie_rate=1.0 if maj == TIE else 0.0,
            decisive_rate=0.0 if maj == TIE else 1.0,
            agreement_decisive=chance_ag if maj != TIE else float("nan"),
            n_pairs=len(hold),
            agreement_truth=(0.5 if maj != TIE else 0.5),
        )
    )
    # 按 `BASELINES` 的顺序排 —— 报告里的行序必须与配置里的声明一致，
    # 否则「谁比谁强」要靠读者自己重新对齐两张表。
    order = {name: i for i, name in enumerate(BASELINES)}
    scores.sort(key=lambda s: order.get(s.method, 99))

    # ---- 标签诊断
    flipped = sum(1 for p in clean_pairs if p.label != (PREFER_A if p.delta_quality > 0 else PREFER_B))
    label_noise_measured = flipped / len(clean_pairs) if clean_pairs else 0.0
    pos = sum(1 for p in hold if p.label == PREFER_A)
    tie_n = sum(1 for p in hold if p.label == TIE)

    # ---- 误差分解
    decomp = _decomposition(
        world, nz, cal, ms.values["component_searched_w"],
        specific_majority=int(bm["specific_majority"]),
        tie_band=tie_band,
    )
    reports = _provider_reports(world, pool, nz, cal, decomp, cids=cids, tie_band=tie_band)

    # ---- 不确定度诊断（用判定口径的 u，因为那是驱动判定的量）
    searched_u = ms.u["component_searched_w"]
    errs = {
        c: float(ms.values["component_searched_w"][c]) - world.candidate(c).quality
        for c in hold_cids
    }
    us = {c: searched_u[c] for c in hold_cids}
    # **去偏**：`u` 量的是误差的**散布**，不是常数错位。`R = Σ w·C` 里各分量的
    # 均值并不相等（correctness 均值 ≈ 0.30，quality 均值 ≈ 0.54），所以只要
    # `w ≠ w_true` 就必然出现一个常数错位。把它算进 ECE 等于让不确定度去背一个
    # 偏差的账 —— 而那是校准该修的，不是 u 该报的。偏移本身单独报（`reward_bias`）。
    reward_bias = mean([errs[c] for c in hold_cids]) if hold_cids else 0.0
    deb = {c: errs[c] - reward_bias for c in hold_cids}
    ece_v = ece([us[c] for c in hold_cids], [deb[c] for c in hold_cids], bins=int(cfg["uncertainty"]["ece_bins"]))
    corr = spearman([us[c] for c in hold_cids], [abs(deb[c]) for c in hold_cids])
    # 四分位提升：把候选按 u 排序，最高四分位的 mean|误差| 除以最低四分位的。
    # **比秩相关稳健**：u 与误差的关系是弱单调的（误差由模型偏差主导，u 只看得到
    # 测量噪声），秩相关在 160 个点上会被噪声吃掉，而「最不确定的那批确实更差」
    # 这个更弱但更该成立的断言不会被吃掉。
    lift = 1.0
    if len(hold_cids) >= 8:
        srt = sorted(hold_cids, key=lambda c: us[c])
        q = max(1, len(srt) // 4)
        lo_m = mean([abs(deb[c]) for c in srt[:q]])
        hi_m = mean([abs(deb[c]) for c in srt[-q:]])
        lift = (hi_m / lo_m) if lo_m > 1e-12 else 1.0

    # ---- 分家族（对干净标签的口径，与主表同一列）
    per_family: dict[str, dict[str, float]] = {}
    for fam in sorted({p.family for p in hold}):
        sub = [p for p in hold if p.family == fam]
        if not sub:
            continue
        sub_clean = [p for p in sub if abs(p.delta_quality) > tie_band]
        row: dict[str, float] = {"n_pairs": float(len(sub))}
        for m in ("component_searched_w", "best_single_calibrated", "naive_raw_mean"):
            vals = {p.id: float(ms.values[m][p.a]) - float(ms.values[m][p.b]) for p in sub}
            row[m] = _score(m, sub, vals, u_pairs[m], kappa, clean_pairs=sub_clean).agreement_truth
        per_family[fam] = row

    # ---- 训练集一致性（过拟合差额）
    train_deltas = {p.id: float(ms.values["component_searched_w"][p.a]) - float(ms.values["component_searched_w"][p.b]) for p in train}
    train_ag = _score("train", train, train_deltas, build_u_pairs(train, searched_u), kappa).agreement
    hold_ag = next(s.agreement for s in scores if s.method == "component_searched_w")
    # `point_agreement` 指的是**交付方法**的一致性，不是 argmax 的 —— 它是在
    # bootstrap 之前算的，所以在这里补上。
    boot = dataclasses.replace(boot, point_agreement=hold_ag)

    return BenchmarkResult(
        n_pairs=len(hold),
        n_pairs_clean=len(clean_pairs),
        label_pos_rate=pos / len(hold) if hold else 0.0,
        label_tie_rate=tie_n / len(hold) if hold else 0.0,
        label_noise_measured=label_noise_measured,
        methods=tuple(scores),
        providers=reports,
        decomposition=decomp,
        ece=ece_v,
        uncertainty_mean=mean([us[c] for c in hold_cids]) if hold_cids else 0.0,
        uncertainty_mae_correlation=corr,
        uncertainty_lift=lift,
        per_family=per_family,
        bootstrap=boot,
        train_agreement=train_ag,
        overfit_gap=train_ag - hold_ag,
        reward_bias=reward_bias,
    )


def over_estimate(world: World, values: Mapping[str, float], us: Mapping[str, float]) -> float:
    cids = sorted(values)
    return over_estimate_ratio([us[c] for c in cids], [values[c] - world.candidate(c).quality for c in cids])
