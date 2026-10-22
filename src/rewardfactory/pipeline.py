"""编排：把九个模块串成一条流水线，并把结果落成产物。

```
config ─┬─ world     真值层（latent 分量 + 注入权重 + 带噪声的人类标注）
        ├─ pool      provider 池（五个不可比的打分器）
        ├─ normalize 方向 + 盲区（**只用 anchor 测，不读配置**）
        ├─ calibrate 五种校准法 → 分段线性表 + 交叉验证可靠性
        ├─ aggregate 可靠性加权 → C[k] → u[k]
        ├─ search    四个搜索器搜权重 w（**不读真值**）
        ├─ benchmark 第三方评测（读真值，算上界与覆盖率）
        ├─ lineage   阶段级血缘 DAG
        └─ products  17 个产物 → out/
```

## 产物从**字典**渲染，不从对象渲染

`build_products` 一次性把全部产物造成**纯字典**，然后 `write_outputs` 落盘、
`report.render_report` / `visualize.render_html` 从同一批字典渲染。
于是「跑完直接看报告」与「从产物重建报告」走**同一条代码路径** ——
两条路径会漂，而漂了之后「报告里的数字」与「产物里的数字」哪个对，
没有任何办法判断。

## 决定论

除 `version.json` 的 `created_at` 之外，**每个产物逐位可复现**。
所以报告正文里不许印生成时间 —— 印了它，两份相同输入的报告就不逐位相同。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate import AggregateResult, aggregate, normalize_w
from .benchmark import benchmark
from .calibrate import CalibrateResult, calibrate
from .config import protocol_digest, version_digest
from .lineage import (
    ACCEPT_NODE,
    BENCH_NODE,
    EXPECT_NODE,
    MIX_NODE,
    POOL_NODE,
    REWARD_NODE,
    TRUTH_NODE,
    WEIGHT_NODE,
    build_lineage,
)
from .normalize import NormalizeResult, normalize
from .providers import ProviderPool, build_pool
from .rng import Rng, derive_seed, digest_of
from .schema import (
    COMPONENTS,
    PREFER_A,
    PREFER_B,
    TIE,
    RewardEvidence,
    SearchResult,
    VerifyResult,
    label_name,
)
from .search import build_objective, search_all, simplex_lattice
from .stats import mean
from .uncertainty import combine, weight_uncertainty
from .world import World, build_world, clip

__all__ = [
    "OUT_FILES",
    "EXTRA_FILES",
    "Version",
    "PipelineResult",
    "run_pipeline",
    "build_products",
    "write_outputs",
    "load_result",
    "LoadedResult",
    "read_products",
]

#: 全部产物（不含 `report.html`，它由 `visualize` 命令单独写）。
OUT_FILES: tuple[str, ...] = (
    "world.json",
    "providers.json",
    "calibration.json",
    "aggregate.json",
    "weights.json",
    "search.json",
    "benchmark.json",
    "pairs.csv",
    "scores.csv",
    "evidence.json",
    "expectations.json",
    "acceptance.json",
    "lineage.json",
    "lineage_reward.dot",
    "report.md",
    "version.json",
)

EXTRA_FILES: tuple[str, ...] = ("report.html",)

#: `scores.csv` 的列。顺序钉死 —— 列序会进指纹。
_SCORE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "prompt_id",
    "family",
    "split",
    "n_tokens",
    "quality",
    "total",
    "uncertainty",
    "u_weight",
    "u_total",
    *[f"C_{k}" for k in COMPONENTS],
    *[f"u_{k}" for k in COMPONENTS],
    *[f"contrib_{k}" for k in COMPONENTS],
)

_PAIR_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "split",
    "family",
    "a",
    "b",
    "label",
    "label_name",
    "delta_quality",
    "noisy",
    *[f"dC_{k}" for k in COMPONENTS],
    "delta_r",
    "u_pair",
    "pred",
    "correct",
    "decisive",
)


@dataclass(frozen=True)
class Version:
    """内容寻址版本。**`created_at` 是唯一允许每次不同的字段。**"""

    id: str
    digest: str
    protocol_digest: str
    created_at: str
    n_candidates: int
    n_pairs: int
    n_providers: int
    n_components: int
    searched: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "digest": self.digest,
            "protocol_digest": self.protocol_digest,
            "created_at": self.created_at,
            "n_candidates": self.n_candidates,
            "n_pairs": self.n_pairs,
            "n_providers": self.n_providers,
            "n_components": self.n_components,
            "searched": self.searched,
            "note": "除 created_at 外全部字段逐位可复现；报告正文不印生成时间。",
        }


@dataclass
class PipelineResult:
    """一次完整运行的全部中间物。**只在内存里活着。**"""

    cfg: Mapping[str, Any]
    world: World
    pool: ProviderPool
    normalize: NormalizeResult
    calibrate: CalibrateResult
    aggregate: AggregateResult
    search: Mapping[str, SearchResult]
    searched: str
    benchmark: Any
    mixes: Mapping[str, Any]
    evidence: Mapping[str, RewardEvidence]
    ablations: Mapping[str, Any]
    lineage: Any
    version: Version

    # ---- 便捷查询

    @property
    def weights(self) -> tuple[float, ...]:
        """**交付的目标函数**的权重 = bootstrap 均值（bagged），不是 argmax。"""
        return tuple(self.benchmark.bootstrap.bagged_w) or tuple(self.search[self.searched].best_w)

    @property
    def weights_argmax(self) -> tuple[float, ...]:
        """搜索器直接给出的答案（argmax）。**会落在单纯形顶点上**，只作对照。"""
        return tuple(self.search[self.searched].best_w)

    @property
    def weights_point_error(self) -> dict[str, float]:
        """`|w*_k − w_true_k|`。**这是评测侧的读数**（要真值），不是搜索侧的。
        这里量的是**交付的 bagged 权重**。"""
        return {
            k: abs(a - b)
            for k, a, b in zip(COMPONENTS, self.weights, self.world.latent_weights)
        }

    def mix(self, cid: str) -> Any:
        return self.mixes[cid]

    def reward(self, cid: str) -> float:
        return self.mixes[cid].total

    def method(self, name: str) -> Any:
        return self.benchmark.method(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version.to_dict(),
            "searched": self.searched,
            "weights": {k: round(v, 6) for k, v in zip(COMPONENTS, self.weights)},
        }


# ------------------------------------------------------------------ 产物构建

def _fmt(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """手写 CSV。**不引 `csv` 模块**：它的行尾与引号策略随平台变
    （Windows 上默认 `\\r\\n`），而产物必须逐位可复现。

    空值一律写成空串，`None` 与 0 必须能区分 —— 「看不见」和「打了 0 分」
    是两件完全不同的事。
    """

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            if math.isnan(v):
                return "nan"
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return f"{v:.6f}"
        s = str(v)
        return '"' + s.replace('"', '""') + '"' if ("," in s or '"' in s or "\n" in s) else s

    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(cell(r.get(c)) for c in columns))
    return "\n".join(lines) + "\n"


def build_products(res: PipelineResult) -> dict[str, Any]:
    """把结果造成全部产物的**纯字典**。

    **报告从这批字典渲染，不从对象渲染。** 于是「跑完直接看报告」与
    「从产物重建报告」走同一条路径 —— 两条路径会漂，而漂了之后
    「报告里的数字」与「产物里的数字」哪个对，没有任何办法判断。
    """
    cfg = res.cfg
    w = res.weights
    nz, cal, agg, bench = res.normalize, res.calibrate, res.aggregate, res.benchmark

    # ---- providers.json（含归一化诊断 —— 它属于「provider 的档案」）
    prov_doc = res.pool.to_dict()
    prov_doc["normalize"] = nz.to_dict()
    prov_doc["reports"] = [p.to_dict() for p in bench.providers]
    prov_doc["self_uncertainty_note"] = (
        "provider 自报的 self_uncertainty 系统性乐观（配置里 ×0.5），"
        "产物里一律用 anchor 上实测的 mae_after。自报值只在报告里出现一次作为反例。"
    )

    # ---- weights.json
    weights_doc = {
        "searched": res.searched,
        "w": {k: _fmt(v) for k, v in zip(COMPONENTS, w)},
        "w_truth": {k: _fmt(v) for k, v in zip(COMPONENTS, res.world.latent_weights)},
        "point_error": {
            k: _fmt(abs(a - b)) for k, a, b in zip(COMPONENTS, w, res.world.latent_weights)
        },
        "support": [COMPONENTS[i] for i in res.search[res.searched].support],
        "bootstrap": bench.bootstrap.to_dict(),
        "note": (
            "w_truth 是注入的真值权重，只有评测（第三方）能读。"
            "点估计偏离它是正常的（标签有噪声、样本有限）；"
            "判据是 bootstrap CI 的覆盖率，不是点估计相等。"
        ),
    }

    # ---- search.json
    search_doc = {
        "budget": int(cfg["search"]["budget"]),
        "den": int(cfg["search"]["den"]),
        "n_lattice": len(
            simplex_lattice(len(COMPONENTS), int(cfg["search"]["den"]))
        ),
        "shared_cache": False,
        "shared_cache_note": (
            "刻意不共享去重缓存：本项目评估是纯内存计算，共享省不下钱，"
            "却会让先跑的搜索器把后跑的喂饱，对照组失去意义。"
        ),
        "searchers": {name: sr.to_dict() for name, sr in sorted(res.search.items())},
        "forward_path": _forward_path(res.search.get("forward")),
    }

    # ---- benchmark.json
    bench_doc = bench.to_dict()
    bench_doc["label_counts"] = {
        "a": sum(1 for p in res.world.pairs_of("holdout") if p.label == PREFER_A),
        "b": sum(1 for p in res.world.pairs_of("holdout") if p.label == PREFER_B),
        "tie": sum(1 for p in res.world.pairs_of("holdout") if p.label == TIE),
    }
    bench_doc["ablations"] = res.ablations

    # ---- scores.csv
    score_rows: list[dict[str, Any]] = []
    for cid in sorted(res.mixes):
        cand = res.world.candidate(cid)
        m = res.mixes[cid]
        row: dict[str, Any] = {
            "candidate_id": cid,
            "prompt_id": cand.prompt_id,
            "family": cand.family,
            "split": cand.split,
            "n_tokens": cand.n_tokens,
            "quality": cand.quality,
            "total": m.total,
            "uncertainty": m.uncertainty,
            "u_weight": m.u_weight,
            "u_total": m.u_total,
        }
        for k, name in enumerate(COMPONENTS):
            row[f"C_{name}"] = m.components[k]
            row[f"u_{name}"] = m.u_components[k]
            row[f"contrib_{name}"] = m.provider_contrib.get(name, 0.0)
        score_rows.append(row)

    # ---- pairs.csv
    hold = res.world.pairs_of("holdout")
    kappa = float(cfg["uncertainty"]["tie_kappa"])
    pair_rows: list[dict[str, Any]] = []
    for p in hold:
        ma, mb = res.mixes[p.a], res.mixes[p.b]
        dr = ma.total - mb.total
        up = ma.uncertainty + mb.uncertainty
        pred = TIE if abs(dr) <= kappa * up else (PREFER_A if dr > 0 else PREFER_B)
        pair_rows.append(
            {
                "pair_id": p.id,
                "split": p.split,
                "family": p.family,
                "a": p.a,
                "b": p.b,
                "label": p.label,
                "label_name": label_name(p.label),
                "delta_quality": p.delta_quality,
                "noisy": p.noisy,
                **{
                    f"dC_{k}": agg.components[p.a][i] - agg.components[p.b][i]
                    for i, k in enumerate(COMPONENTS)
                },
                "delta_r": dr,
                "u_pair": up,
                "pred": pred,
                "correct": pred == p.label,
                "decisive": pred != TIE,
            }
        )

    products: dict[str, Any] = {
        "world.json": res.world.to_dict(),
        "providers.json": prov_doc,
        "calibration.json": cal.to_dict(),
        "aggregate.json": agg.to_dict(),
        "weights.json": weights_doc,
        "search.json": search_doc,
        "benchmark.json": bench_doc,
        "pairs.csv": _csv(_PAIR_COLUMNS, pair_rows),
        "scores.csv": _csv(_SCORE_COLUMNS, score_rows),
        "evidence.json": {
            "interface": "RewardProvider.score(prompt, response) -> RewardEvidence",
            "n": len(res.evidence),
            "evidences": [e.to_dict() for e in res.evidence.values()],
        },
        "lineage.json": res.lineage.to_dict(),
        "lineage_reward.dot": res.lineage.to_dot(),
        "version.json": res.version.to_dict(),
    }
    return products


def _forward_path(sr: SearchResult | None) -> list[dict[str, Any]]:
    """把 `forward` 的轨迹压成「每一步的支持集 + 权重 + 一致性」。

    这条路径是**产物里最重要的一张表**：它把 README 验收标准一
    （组合相对单源的提升）从一句话变成一个可以逐行读的过程。
    """
    if sr is None:
        return []
    out: list[dict[str, Any]] = []
    best = -1.0
    for t in sr.trace:
        if t.agreement > best + 1e-15:
            best = t.agreement
            out.append(
                {
                    "step": t.step,
                    "support": [COMPONENTS[i] for i, x in enumerate(t.w) if x > 1e-9],
                    "w": {k: _fmt(v) for k, v in zip(COMPONENTS, t.w)},
                    "agreement": _fmt(t.agreement, 6),
                }
            )
    return out


def _evidence(
    res_cfg: Mapping[str, Any],
    world: World,
    agg: AggregateResult,
    cal: CalibrateResult,
    mixes: Mapping[str, Any],
    cids: Sequence[str],
) -> dict[str, RewardEvidence]:
    """造 `RewardEvidence` —— README 第四节那个统一接口的返回体。

    它是 M0「跑通单一 RM 打分一次」的兑现物：**任何来源都返回同一个结构，
    带分量、带不确定度，而不是一个裸标量。**
    """
    kappa = float(res_cfg["uncertainty"]["tie_kappa"])
    out: dict[str, RewardEvidence] = {}
    pids = sorted(cal.scores)
    for cid in cids:
        m = mixes[cid]
        cand = world.candidate(cid)
        comps = {k: _fmt(v) for k, v in zip(COMPONENTS, m.components)}
        lines = [
            f"分量 C[k]: " + " / ".join(f"{k}={comps[k]:.4f}" for k in COMPONENTS),
            f"总不确定度 u={m.uncertainty:.4f}（u_total={m.u_total:.4f}，其中权重项 u_w={m.u_weight:.4f}）",
            f"判定尺度: |ΔR| ≤ {kappa:g}·(u_a+u_b) 判平局",
            "provider 贡献: "
            + " / ".join(f"{p}={m.provider_contrib.get(p, 0.0):.4f}" for p in pids),
            "可靠来源: "
            + " / ".join(
                f"{p}ρ={mean([x for x in cal.reliability[p]]):.3f}" for p in pids
            ),
        ]
        out[cid] = RewardEvidence(
            candidate_id=cid,
            prompt_id=cand.prompt_id,
            total_score=m.total,
            components=comps,
            uncertainty=m.uncertainty,
            providers={p: m.provider_contrib.get(p, 0.0) for p in pids},
            evidence=tuple(lines),
        )
    return out


# ------------------------------------------------------------------ 消融

def _force_positive(nz: NormalizeResult, pool: ProviderPool) -> NormalizeResult:
    """把方向检测**关掉**（全部按 +1 处理）重建一份归一化结果。

    刻意重建 `ProviderScore` 而不是改 `NormalizeResult` 的字段：后者是 frozen
    的 dataclass，`dataclasses.replace` 换不掉嵌套的 `Mapping` 内容，
    会得到一个「看起来改了、其实没改」的对象 —— 那种 bug 的产物与主运行完全一样，
    消融结论却写成「方向检测没有价值」。
    """
    from .schema import ProviderScore

    scores: dict[str, dict[str, ProviderScore]] = {}
    for spec in pool.specs:
        pid = spec.id
        diag = nz.diagnostic(pid)
        per: dict[str, ProviderScore] = {}
        for cid in sorted(pool.outputs[pid]):
            out = pool.output(pid, cid)
            comps = tuple(
                None if diag.blind[k] else float(v) for k, v in enumerate(out.components)
            )
            per[cid] = ProviderScore(
                provider_id=pid,
                candidate_id=cid,
                components=comps,
                raw_total=float(out.raw_total),
                scale=spec.scale,
                direction=1,
                self_uncertainty=out.self_uncertainty,
            )
        scores[pid] = per
    return NormalizeResult(scores=scores, diagnostics=nz.diagnostics)


def _direction_ablation(
    world: World,
    pool: ProviderPool,
    nz: NormalizeResult,
    cfg: Mapping[str, Any],
    *,
    main_agreement: float,
    cal_main: CalibrateResult,
) -> dict[str, Any]:
    """方向检测的消融：**真的重跑整条下游**，只把方向固定成 +1。

    两条独立的结论要分开看（实测它们确实不同）：

    - `agreement`：一致性变了多少 —— 可能几乎不变，因为可靠性加权会**自己**
      把一个方向反了的 provider 压到 0（它的校准残差 ≈ 0.5 ⇒ ρ ≈ 0）；
    - `dead_provider_reliability`：那个反号的 provider 的可靠性 —— 这个一定变，
      因为它从「有信息但用反了」变成「纯噪声」。
    """
    from .providers import ProviderPool as _P

    forced = _force_positive(nz, pool)
    cal2 = calibrate(world, forced, cfg)
    agg2 = aggregate(cal2, forced, cfg)
    rng = Rng(derive_seed(cfg["seed"], "ablation", "direction"))
    hold = world.pairs_of("holdout")
    train = world.pairs_of("train")
    u_pairs = {p.id: float(agg2.u_components[p.a][0] + agg2.u_components[p.b][0]) for p in hold}
    kappa = float(cfg["uncertainty"]["tie_kappa"])
    obj = build_objective(
        train, agg2.components,
        {p.id: float(agg2.u_components[p.a][0] + agg2.u_components[p.b][0]) for p in train},
        kappa=kappa, n_components=len(COMPONENTS),
    )
    from .search.coordinate import coordinate_ascent

    w, _v, _t = coordinate_ascent(
        obj,
        moves=[float(t) for t in cfg["search"]["coordinate_moves"]],
        budget=int(cfg["search"]["budget"]),
        start=tuple(1.0 / len(COMPONENTS) for _ in COMPONENTS),
        record=False,
    )
    correct = 0
    for p in hold:
        da = math.fsum(a * b for a, b in zip(w, agg2.components[p.a]))
        db = math.fsum(a * b for a, b in zip(w, agg2.components[p.b]))
        dr = da - db
        up = agg2.u_components[p.a][0] + agg2.u_components[p.b][0]
        pred = TIE if abs(dr) <= kappa * up else (PREFER_A if dr > 0 else PREFER_B)
        if pred == p.label:
            correct += 1
    ag = correct / len(hold) if hold else 0.0

    inverted = [str(p["id"]) for p in cfg["providers"] if bool(p.get("invert", False))]
    rel_now = {pid: mean(list(cal_main.reliability[pid])) for pid in cal_main.reliability}
    rel_forced = {pid: mean(list(cal2.reliability[pid])) for pid in cal2.reliability}
    return {
        "forced_direction": 1,
        "agreement": _fmt(ag, 6),
        "main_agreement": _fmt(main_agreement, 6),
        "agreement_delta": _fmt(main_agreement - ag, 6),
        "inverted_providers": inverted,
        "reliability_main": {pid: _fmt(v, 6) for pid, v in rel_now.items()},
        "reliability_forced": {pid: _fmt(v, 6) for pid, v in rel_forced.items()},
        "detected_direction": {
            d.provider_id: d.detected_direction for d in nz.diagnostics
        },
        "note": (
            "forced=+1 时方向检测被关掉。`agreement_delta` 可能接近 0 —— "
            "因为可靠性加权会把方向反了的 provider 自动压到 0（两条机制在互相兜底）；"
            "但 `reliability_main` 与 `reliability_forced` 的差一定显著。"
        ),
    }


# ------------------------------------------------------------------ 主流程

def _reproducibility_check(
    cfg: Mapping[str, Any],
    search: Mapping[str, SearchResult],
    searched: str,
    factory: Any,
) -> dict[str, Any]:
    """**真的再搜一遍**，逐位比对。

    不是「读一遍代码觉得它确定」—— 那种信心一文不值。真的重跑一遍，成本是
    一次搜索（约 1 秒），换来的是一个可写进产物的布尔值。

    这一条同时守住三个前提：`Objective` 的缓存不改变结果、搜索器的平手规则
    是确定性的、`Rng` 的派生路径与调用顺序无关。
    """
    from .search import search_all as _sa

    rng2 = Rng(derive_seed(cfg["seed"], "search"))
    again = _sa(factory, cfg, rng2)
    a = search[searched].best_w
    b = again[searched].best_w
    same = len(a) == len(b) and all(abs(x - y) <= 1e-12 for x, y in zip(a, b))
    same_ag = abs(search[searched].best_agreement - again[searched].best_agreement) <= 1e-12
    return {
        "identical": bool(same and same_ag),
        "digest": digest_of([round(x, 9) for x in a]),
        "second_digest": digest_of([round(x, 9) for x in b]),
        "n_searchers": len(search),
    }


def run_pipeline(cfg: Mapping[str, Any]) -> PipelineResult:
    """端到端。**唯一的写盘入口是 `write_outputs`，本函数不碰磁盘。**"""
    world = build_world(cfg)
    pool = build_pool(world, cfg)
    nz = normalize(pool, world, cfg)
    cal = calibrate(world, nz, cfg)
    agg = aggregate(cal, nz, cfg)

    kappa = float(cfg["uncertainty"]["tie_kappa"])
    train = world.pairs_of("train")

    # 搜索目标函数里的 u_pair：用**逐分量平均**（不是 w 加权）—— 目标函数必须在
    # 搜索开始之前就固定下来，否则「评估一个 w」会依赖于「当前 w」，
    # 目标函数就不是一个函数了。
    train_u = {p.id: float(mean(agg.u_components[p.a]) + mean(agg.u_components[p.b])) for p in train}

    def _factory():
        return build_objective(
            train, agg.components, train_u, kappa=kappa, n_components=len(COMPONENTS)
        )

    srng = Rng(derive_seed(cfg["seed"], "search"))
    search = search_all(_factory, cfg, srng)
    searched = "coordinate" if "coordinate" in search else sorted(search)[0]

    brng = Rng(derive_seed(cfg["seed"], "benchmark"))
    bench = benchmark(
        world, pool, nz, cal, agg, search,
        searched=searched, cfg=cfg, rng=brng,
    )

    # ---- 权重项不确定度（bootstrap 样本 → 每个候选一个 std）。只报不判。
    include_w = bool(cfg["uncertainty"]["include_weight_term"])
    u_w = weight_uncertainty(bench.bootstrap.samples, agg.components) if include_w else {}
    prompt_of = {c.id: c.prompt_id for c in world.candidates}
    # **交付的目标函数用 bootstrap 均值，不用 argmax。** 搜索目标是分片常数的
    # （一对对地数对错），argmax 会落在单纯形顶点上，换一批标注就跳到别处；
    # 200 次重采样的均值把顶点收缩回内部。见 `benchmark.bootstrap_weights`。
    delivered_w = tuple(bench.bootstrap.bagged_w) or tuple(search[searched].best_w)
    mixes = agg.mix_all(
        delivered_w, kappa,
        u_w=(u_w if include_w else None),
        prompt_of=prompt_of,
    )

    ablations = {
        "direction": _direction_ablation(
            world, pool, nz, cfg,
            main_agreement=bench.method("component_searched_w").agreement,
            cal_main=cal,
        ),
        "reproducibility": _reproducibility_check(cfg, search, searched, _factory),
    }

    evidence = _evidence(cfg, world, agg, cal, mixes, sorted(mixes))

    lin = build_lineage(
        providers=[s.id for s in pool.specs],
        searchers=sorted(search),
        searched=searched,
        n_candidates=len(world.candidates),
        n_pairs={s: len(world.pairs_of(s)) for s in ("anchor", "train", "holdout")},
        mix_inputs=[s.id for s in pool.specs],
        weight_inputs=sorted(search),
        extra_meta={
            "holdout_pairs": bench.n_pairs,
            "searched_agreement": _fmt(bench.method("component_searched_w").agreement, 6),
        },
    )

    vdig = version_digest(cfg)
    version = Version(
        id=f"v1-{vdig[:12]}",
        digest=vdig,
        protocol_digest=protocol_digest(cfg),
        created_at=_utcnow(),
        n_candidates=len(world.candidates),
        n_pairs=len(world.pairs),
        n_providers=len(pool.specs),
        n_components=len(COMPONENTS),
        searched=searched,
    )

    return PipelineResult(
        cfg=cfg,
        world=world,
        pool=pool,
        normalize=nz,
        calibrate=cal,
        aggregate=agg,
        search=search,
        searched=searched,
        benchmark=bench,
        mixes=mixes,
        evidence=evidence,
        ablations=ablations,
        lineage=lin,
        version=version,
    )


def _utcnow() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_all_products(res: PipelineResult) -> dict[str, Any]:
    """核心产物 + 期望 + 验收 + 报告。**落盘与内存两条路都走这里。**

    期望与验收需要跨产物计算（要 `world` 的真值与 `bench` 的成绩），所以它们
    不能在 `build_products` 里算 —— 那个函数只吃「已经是产物形态」的东西。
    于是分两层：`build_products`（纯产物）→ 本函数（补上需要回看结果的三个）。
    """
    from .acceptance import verify_all
    from .report import expectation_rows, render_report

    products = build_products(res)
    products["expectations.json"] = {"rows": [r.to_dict() for r in expectation_rows(res)]}
    products["acceptance.json"] = verify_all(res).to_dict()
    products["report.md"] = render_report(products)
    return products


def write_outputs(res: PipelineResult, out_dir: Path | str) -> list[Path]:
    """落盘。**唯一写盘入口。**"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    products = build_all_products(res)

    written: list[Path] = []
    for name in OUT_FILES:
        payload = products.get(name)
        if payload is None:
            raise KeyError(f"产物 {name} 没有构造出来 —— OUT_FILES 与实际产物必须一一对应")
        p = out / name
        if name.endswith((".csv", ".dot", ".md")):
            p.write_text(str(payload), encoding="utf-8", newline="\n")
        else:
            _write_json(p, payload)
        written.append(p)
    return written


# ------------------------------------------------------------------ 回读

@dataclass(frozen=True)
class LoadedResult:
    """从产物重建的只读视图。**字段就是产物字典本身** —— 报告从它渲染，
    与「刚跑完」那条路径共用同一批字典，所以两条路径不可能漂。"""

    out_dir: Path
    products: Mapping[str, Any]

    def get(self, name: str) -> Any:
        if name not in self.products:
            raise KeyError(f"产物里没有 {name}（已载入：{sorted(self.products)}）")
        return self.products[name]

    @property
    def version(self) -> Mapping[str, Any]:
        return self.get("version.json")

    @property
    def benchmark(self) -> Mapping[str, Any]:
        return self.get("benchmark.json")

    @property
    def weights(self) -> Mapping[str, Any]:
        return self.get("weights.json")

    def methods(self) -> list[Mapping[str, Any]]:
        return list(self.benchmark["methods"])

    def method(self, name: str) -> Mapping[str, Any] | None:
        for m in self.methods():
            if m["method"] == name:
                return m
        return None

    def providers(self) -> list[Mapping[str, Any]]:
        return list(self.get("providers.json")["reports"])

    def score_rows(self) -> list[dict[str, str]]:
        return _read_csv(str(self.get("scores.csv")))

    def pair_rows(self) -> list[dict[str, str]]:
        return _read_csv(str(self.get("pairs.csv")))

    def evidence_rows(self) -> list[Mapping[str, Any]]:
        return list(self.get("evidence.json")["evidences"])

    def evidence(self, cid: str) -> Mapping[str, Any] | None:
        for e in self.evidence_rows():
            if e["candidate_id"] == cid:
                return e
        return None


def _read_csv(text: str) -> list[dict[str, str]]:
    """回读自己写出去的 CSV。**只支持自己写的那个格式**（无引号内换行），
    这不是通用 CSV 解析器，也不该是 —— 通用解析器会掩盖「我们写了什么」。"""
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return []
    head = lines[0].split(",")
    out: list[dict[str, str]] = []
    for ln in lines[1:]:
        cells = ln.split(",")
        out.append({h: (cells[i] if i < len(cells) else "") for i, h in enumerate(head)})
    return out


def read_products(out_dir: Path | str) -> LoadedResult | None:
    """读全部产物。**缺任何一个就返回 `None`** —— 半套产物比没有产物更危险：
    它会让人拿一份不完整的报告下结论。"""
    out = Path(out_dir)
    if not out.is_dir():
        return None
    products: dict[str, Any] = {}
    for name in OUT_FILES:
        p = out / name
        if not p.is_file():
            return None
        if name.endswith(".csv") or name.endswith(".dot") or name.endswith(".md"):
            products[name] = p.read_text(encoding="utf-8")
        else:
            try:
                products[name] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return LoadedResult(out_dir=out, products=products)


#: 兼容别名：读命令统一调 `load_result`。
load_result = read_products
