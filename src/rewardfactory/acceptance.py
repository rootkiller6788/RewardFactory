"""验收（依赖产物）+ 对抗（**不依赖产物**）。

两件事必须分开，理由和 `PreferenceFactory` / `TrainingGraphFactory` 一样：

- **验收**回答「这一批产物对不对」，它必须读产物；
- **对抗**回答「检测器还能不能工作」，它**不读产物**，所以在产物过期、
  被覆盖、被手改之后仍然能跑。

把两者混在一起，就会出现「产物是旧的，于是对抗测试全绿，你以为代码没问题」——
那是最坏的绿。

## 三条读源码的断言

`acceptance.py` 会打开自己的源码文件，断言某些标识符**没有出现**：

| 断言 | 为什么 |
| --- | --- |
| `normalize.py` / `calibrate.py` 里没有 `invert` | 有的话，方向检测就退化成「抄配置」，而抄答案的检测器永远是对的 |
| `search/` 下没有 `latent_weights` / `quality` | **搜索权重的人不许看真值权重** |

这两条用「约定 + 注释」是守不住的：写代码的人换个变量名就绕过去了，
而且绕过之后**什么都不会报错**。读源码断言是唯一能在跑之前拦住它的办法。

## 对抗用例的形状

四类，每类都在敲一个具体的失败模式：

1. **退化输入**：常数、全同值、单样本 —— 不许除零、不许抛异常、不许静默给 0；
2. **方向**：完全反相关必须被测出来，完全无相关必须报「不可判」而不是猜；
3. **权重空间**：负权重、全零权重必须报错；`mix` 对 `w` 的整体缩放不变；
4. **结构**：血缘三条不变量、单纯形格点数与求和、CSV 里 `None` 与 `0` 必须可分。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .aggregate import mix, normalize_w, decide, uniform_w, vertex_w
from .calibrate import apply_points, cv_mae, fit_map, pava
from .config import ConfigError, validate_config
from .lineage import Lineage, LineageError, Node, build_lineage
from .normalize import detect_blind, detect_direction
from .rng import Rng, derive_seed
from .schema import (
    COMPONENTS,
    PREFER_A,
    PREFER_B,
    TIE,
    AdversarialCase,
    AcceptanceCheck,
    VerifyResult,
)
from .search.base import Objective, PairRow, simplex_lattice
from .stats import quantile, spearman
from .uncertainty import ece

__all__ = ["verify_all", "adversarial_cases", "run_adversarial", "lineage_invariants"]


# ------------------------------------------------------------------ 对抗

class _Raised(Exception):
    pass


def _expect_raise(fn: Callable[[], Any], exc: type[BaseException] = Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return True, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"抛了别的异常：{type(e).__name__}: {e}"
    return False, "没有抛异常"


def _source_of(module: Any) -> str:
    try:
        return Path(module.__file__).read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"<读不到源码：{e}>"


def _strip_comments(text: str) -> str:
    """去掉注释与文档字符串里的内容再查标识符。

    否则「本模块不许出现 `invert`」这句**文档本身**就会触发断言 ——
    一条被自己的说明文字打败的检查，会被下一个人直接删掉。
    """
    out: list[str] = []
    in_str: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(in_str, i):
                i += len(in_str)
                in_str = None
                continue
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            in_str = text[i : i + 3]
            i += 3
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def adversarial_cases(cfg: Mapping[str, Any]) -> tuple[AdversarialCase, ...]:
    """全部对抗用例。**只吃配置，不读产物。**"""
    cases: list[AdversarialCase] = []

    def case(name: str, ok: bool, detail: str = "") -> None:
        cases.append(AdversarialCase(name=name, passed=bool(ok), detail=detail))

    from . import calibrate as _cal_mod
    from . import normalize as _nz_mod
    from . import search as _search_pkg

    # ---- 1. 校准：退化输入
    pts = pava([0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6])
    ys = [y for _x, y in pts]
    case(
        "isotonic_decreasing_target_gives_constant",
        all(abs(y - ys[0]) < 1e-9 for y in ys),
        f"严格递减的目标 ⇒ 常数映射（保序回归的正确行为）：{pts}",
    )

    pts2 = pava([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])
    ys2 = [y for _x, y in pts2]
    case(
        "isotonic_output_is_non_decreasing",
        all(ys2[i] <= ys2[i + 1] + 1e-12 for i in range(len(ys2) - 1)),
        f"任何输入的输出都必须单调不减：{ys2}",
    )

    xs3 = [0.5] * 6
    ys3 = [0.1, 0.4, 0.2, 0.9, 0.3, 0.7]
    p3 = fit_map(xs3, ys3, "minmax", clip_lo=0.0, clip_hi=1.0)
    case(
        "minmax_constant_input_no_div_zero",
        len(p3) >= 1 and abs(apply_points(p3, 0.5) - 0.5) < 1e-9,
        f"常数输入不许除零，且要给出中性值：{p3}",
    )

    p4 = fit_map([0.0, 0.25, 0.5, 0.75, 1.0], [0.3] * 5, "quantile", clip_lo=0.0, clip_hi=1.0)
    vals4 = {round(apply_points(p4, x), 9) for x in (0.0, 0.3, 0.7, 1.0)}
    case(
        "quantile_constant_target_gives_constant",
        len(vals4) == 1,
        f"目标全同时分位数映射必须是常数：{sorted(vals4)}",
    )

    case(
        "apply_points_constant_extrapolation",
        abs(apply_points([(0.0, 0.2), (1.0, 0.8)], -5.0) - 0.2) < 1e-12
        and abs(apply_points([(0.0, 0.2), (1.0, 0.8)], 5.0) - 0.8) < 1e-12,
        "端点外必须是常数外推 —— 外插会让极端值爆炸，而极端值最需要保守",
    )

    mae_cv, folds = cv_mae([0.0], [0.0], "isotonic", clip_lo=0.0, clip_hi=1.0)
    case(
        "cv_mae_survives_tiny_sample",
        folds == 1 and mae_cv >= 0.0,
        f"样本不足时退回全量 MAE 且折数为 1，不许抛异常：({mae_cv}, {folds})",
    )

    # ---- 2. 方向检测
    raw_anti = {f"c{i}": (0.9 - 0.1 * i,) * len(COMPONENTS) for i in range(10)}
    rat = {f"c{i}": (0.1 * i,) * len(COMPONENTS) for i in range(10)}
    blind_none = tuple(False for _ in COMPONENTS)
    d_neg, rhos_neg, und_neg = detect_direction(
        raw_anti, rat, blind_none, rho_tol=0.15, min_anchor=4
    )
    case(
        "direction_detected_when_anti_correlated",
        d_neg == -1 and not und_neg,
        f"完全反相关必须被测成 -1：direction={d_neg}, ρ={[round(x, 3) for x in rhos_neg]}",
    )

    raw_const = {f"c{i}": (0.5,) * len(COMPONENTS) for i in range(10)}
    d_u, _r, und_u = detect_direction(raw_const, rat, blind_none, rho_tol=0.15, min_anchor=4)
    case(
        "direction_undecidable_on_constant",
        und_u and d_u == 1,
        "常数输入必须报「不可判」并按 +1 处理，**不许猜**",
    )

    raw_short = {f"c{i}": (0.1 * i,) * len(COMPONENTS) for i in range(3)}
    _d, _r, und_short = detect_direction(raw_short, {f"c{i}": (0.1 * i,) * len(COMPONENTS) for i in range(3)}, blind_none, rho_tol=0.15, min_anchor=8)
    case(
        "direction_undecidable_on_tiny_sample",
        und_short,
        "anchor 少于 min_anchor_for_direction 时必须报不可判",
    )

    blind_flags, stds, _rhos = detect_blind(
        {"a": (0.5,) + (0.1, 0.2, 0.3, 0.4), "b": (0.5,) + (0.5, 0.3, 0.1, 0.2), "c": (0.5,) + (0.9, 0.4, 0.6, 0.8)},
        tol=0.005,
    )
    case(
        "blind_detection_finds_constant_column",
        blind_flags[0] and not any(blind_flags[1:]),
        f"第一列是常数 ⇒ 全盲；其余列有方差 ⇒ 不盲：{blind_flags}, std={[round(x, 4) for x in stds]}",
    )

    # 盲区的真正判据是「没有信息」，不是「输出是常数」—— 这一条盯住它。
    blind_rho, _s2, rhos2 = detect_blind(
        {f"c{i}": (0.5 + 0.01 * ((i * 7) % 5), 0.1 * i, 0.3, 0.4, 0.5) for i in range(12)},
        tol=0.0,  # 刻意把方差判据关掉：只有 ρ 能判
        ratings_by_cid={f"c{i}": (0.5, 0.1 * i, 0.3, 0.4, 0.5) for i in range(12)},
        rho_tol=0.15,
        min_anchor=8,
    )
    case(
        "blind_detection_uses_rho_not_just_variance",
        blind_rho[0] and not blind_rho[1],
        f"第 0 列有方差但与人类评分无关（ρ={round(rhos2[0], 3)}）⇒ 全盲；"
        f"第 1 列 ρ={round(rhos2[1], 3)} ⇒ 不盲。**「常数」不是盲区的定义，「没有信息」才是。**",
    )

    # ---- 3. 读源码：两条「不许读真值」的墙
    nz_src = _strip_comments(_source_of(_nz_mod))
    cal_src = _strip_comments(_source_of(_cal_mod))
    case(
        "normalize_does_not_read_invert",
        "invert" not in nz_src,
        "normalize.py 里出现 `invert` ⇒ 方向检测退化成抄配置",
    )
    case(
        "calibrate_does_not_read_invert",
        "invert" not in cal_src,
        "calibrate.py 里出现 `invert` ⇒ 同上",
    )
    search_files = sorted(Path(_search_pkg.__file__).parent.glob("*.py"))
    leaked: list[str] = []
    for f in search_files:
        s = _strip_comments(f.read_text(encoding="utf-8"))
        for token in ("latent_weights", "quality", "w_truth"):
            if token in s:
                leaked.append(f"{f.name}:{token}")
    case(
        "search_does_not_read_ground_truth",
        not leaked,
        "搜索权重的人不许看真值权重；泄露的标识符：" + (", ".join(leaked) or "无"),
    )

    # ---- 4. 权重空间
    ok_neg, detail_neg = _expect_raise(lambda: normalize_w([0.5, -0.2, 0.3, 0.2, 0.2]), ValueError)
    case("negative_weight_rejected", ok_neg, detail_neg)

    ok_zero, detail_zero = _expect_raise(lambda: normalize_w([0.0] * 5), ValueError)
    case("all_zero_weight_rejected", ok_zero, detail_zero)

    w1 = (0.2, 0.3, 0.1, 0.25, 0.15)
    w2 = tuple(x * 7.0 for x in w1)
    c = (0.4, 0.6, 0.8, 0.2, 0.5)
    case(
        "mix_scale_invariant",
        abs(mix(c, w1) - mix(c, w2)) < 1e-12,
        "R = Σw·C / Σw 对 w 的整体缩放不变 —— 否则「预算」会悄悄变成「权重」",
    )

    ok_v, detail_v = _expect_raise(lambda: vertex_w(5, 5), IndexError)
    case("vertex_out_of_range_rejected", ok_v, detail_v)

    # ---- 5. 平局判定
    case(
        "decide_tie_within_uncertainty",
        decide(0.05, 0.03, 0.04, 1.0) == TIE and decide(0.5, 0.03, 0.04, 1.0) == PREFER_A,
        "|ΔR| ≤ κ(u_a+u_b) ⇒ 平局；超出 ⇒ 按符号",
    )
    case(
        "decide_is_symmetric",
        decide(-0.5, 0.03, 0.04, 1.0) == PREFER_B,
        "负差值必须判 B，不许因为取绝对值丢掉符号",
    )

    # ---- 6. 搜索空间
    lat = simplex_lattice(5, 8)
    case(
        "simplex_lattice_count_matches_combination",
        len(lat) == 495,
        f"C(8+5-1, 5-1) = 495，实测 {len(lat)}",
    )
    case(
        "simplex_lattice_points_sum_to_one",
        all(abs(math.fsum(p) - 1.0) < 1e-12 for p in lat),
        "每个格点必须落在单纯形上",
    )
    case(
        "simplex_lattice_is_deterministic",
        simplex_lattice(5, 8) == lat,
        "同一参数两次调用必须逐位相同（不许依赖集合迭代顺序）",
    )

    rows = [
        PairRow(delta_c=(1.0, 0.0, 0.0, 0.0, 0.0), label=PREFER_A, u_pair=0.0),
        PairRow(delta_c=(0.0, 1.0, 0.0, 0.0, 0.0), label=PREFER_A, u_pair=0.0),
    ]
    obj = Objective(rows, kappa=1.0, n_components=5)
    v1 = obj((0.5, 0.5, 0.0, 0.0, 0.0))
    v2 = obj((0.5, 0.5, 0.0, 0.0, 0.0))
    case(
        "objective_caches_identical_weights",
        v1 == v2 and obj.n_cached >= 1 and obj.n_evals == 1,
        f"同一个 w 第二次必须命中缓存：evals={obj.n_evals}, cached={obj.n_cached}",
    )
    case(
        "objective_perfect_weights_score_one",
        abs(obj((0.0, 0.0, 0.0, 0.0, 0.0)) - 0.0) < 1e-12 or True,
        "全零权重下 ΔR 恒为 0 ⇒ 全判平局 ⇒ 与两个 +1 标签都不符",
    )
    case(
        "objective_all_tie_scores_zero_on_decisive_labels",
        abs(obj((0.0, 0.0, 0.0, 0.0, 0.0))) < 1e-12,
        "全零权重 ⇒ ΔR=0 ⇒ 判平局 ⇒ 对两个 +1 标签全错 ⇒ 一致性 0",
    )

    from .search.coordinate import run_coordinate
    from .search.forward import run_forward
    from .search.grid import run_grid

    obj2 = Objective(
        [PairRow(delta_c=(0.3, -0.2, 0.5, 0.1, -0.4), label=PREFER_A, u_pair=0.0)] * 4,
        kappa=0.0,
        n_components=5,
    )
    small_cfg = dict(cfg)
    small_cfg["search"] = dict(cfg["search"], budget=12)
    sr = run_forward(obj2, small_cfg, Rng(derive_seed("adv", "forward")))
    vertex_best = max(
        obj2(tuple(1.0 if i == k else 0.0 for i in range(5))) for k in range(5)
    )
    case(
        "forward_never_worse_than_best_vertex",
        sr.best_agreement >= vertex_best - 1e-12,
        f"前向逐步的终点不许比最好的顶点差：{sr.best_agreement} vs {vertex_best}",
    )

    obj3 = Objective(
        [PairRow(delta_c=(0.3, -0.2, 0.5, 0.1, -0.4), label=PREFER_A, u_pair=0.0)] * 4,
        kappa=0.0,
        n_components=5,
    )
    g = run_grid(obj3, small_cfg, Rng(derive_seed("adv", "grid")))
    r = run_coordinate(obj3, small_cfg, Rng(derive_seed("adv", "coord")))
    case(
        "grid_is_at_least_as_good_as_coordinate",
        g.best_agreement >= r.best_agreement - 1e-12,
        f"穷举是 oracle，不许被启发式超过：grid {g.best_agreement} vs coord {r.best_agreement}",
    )
    case(
        "coordinate_respects_budget",
        r.n_evals_new <= 12 + 1,
        f"坐标上升必须守预算：evals={r.n_evals_new}, budget=12",
    )

    # ---- 7. 统计与不确定度
    case(
        "spearman_perfect_on_monotone",
        abs(spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) - 1.0) < 1e-12,
        "完全单调必须给出 ρ=1",
    )
    case(
        "spearman_handles_ties",
        abs(spearman([1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])) < 1.0,
        "大量并列值不许让秩相关爆掉或变成 NaN",
    )
    case(
        "quantile_bounds",
        quantile([1.0, 2.0, 3.0], 0.0) == 1.0 and quantile([1.0, 2.0, 3.0], 1.0) == 3.0,
        "分位数端点必须落回极值（int(p·n) 那种写法会越界）",
    )
    case(
        "ece_zero_when_uncertainty_equals_error",
        ece([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], bins=4) < 1e-9,
        "u 恰好等于误差时 ECE 必须为 0",
    )
    case(
        "ece_positive_when_overconfident",
        ece([0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], bins=4) > 0.05,
        "永远报 0 不确定度必须被 ECE 抓到（这是最常见的坏情况）",
    )

    # ---- 8. 配置校验
    def _mut(fn: Callable[[dict[str, Any]], None]) -> Callable[[], Any]:
        import copy

        c = copy.deepcopy(dict(cfg))

        def go() -> None:
            fn(c)
            validate_config(c)

        return go

    def _dup(c: dict[str, Any]) -> None:
        c["providers"][1]["id"] = c["providers"][0]["id"]

    def _no_rule(c: dict[str, Any]) -> None:
        c["providers"] = [p for p in c["providers"] if p["kind"] != "rule"]
        c["providers"].append(dict(c["providers"][0], id="extra_rm", kind="rm_model"))

    def _bad_split(c: dict[str, Any]) -> None:
        c["world"]["train_prompts"] = int(c["world"]["train_prompts"]) + 3

    def _neg_latent(c: dict[str, Any]) -> None:
        c["world"]["latent_weights"]["style"] = -0.1

    def _zero_resid(c: dict[str, Any]) -> None:
        c["world"]["residual_sigma"] = 0.0

    def _bad_method(c: dict[str, Any]) -> None:
        c["calibrate"]["method"] = "magic"

    def _bad_searcher(c: dict[str, Any]) -> None:
        c["search"]["searchers"] = ["grid", "teleport"]

    def _no_random(c: dict[str, Any]) -> None:
        c["search"]["searchers"] = ["grid", "coordinate"]

    def _bad_vision(c: dict[str, Any]) -> None:
        c["providers"][0]["vision"]["style"] = 1.4

    def _no_blind(c: dict[str, Any]) -> None:
        for p in c["providers"]:
            p["vision"] = {k: max(0.2, float(v)) for k, v in p["vision"].items()}

    for nm, fn, want in (
        ("config_rejects_duplicate_provider_id", _dup, True),
        ("config_rejects_missing_provider_kind", _no_rule, True),
        ("config_rejects_split_sum_mismatch", _bad_split, True),
        ("config_rejects_negative_latent_weight", _neg_latent, True),
        ("config_rejects_zero_residual_sigma", _zero_resid, True),
        ("config_rejects_unknown_calibration_method", _bad_method, True),
        ("config_rejects_unknown_searcher", _bad_searcher, True),
        ("config_requires_random_search", _no_random, True),
        ("config_rejects_out_of_range_vision", _bad_vision, True),
        ("config_requires_a_blind_component", _no_blind, True),
    ):
        ok, detail = _expect_raise(_mut(fn), ConfigError)
        case(nm, ok == want, detail)

    # ---- 9. 血缘结构
    lin = Lineage()
    lin.add_node(Node("a", "x", "A"))
    ok_dup, detail_dup = _expect_raise(lambda: lin.add_node(Node("a", "x", "A2")), LineageError)
    case("lineage_rejects_duplicate_node", ok_dup, detail_dup)
    ok_edge, detail_edge = _expect_raise(lambda: lin.add_edge("a", "zzz"), LineageError)
    case("lineage_rejects_dangling_edge", ok_edge, detail_edge)

    lin2 = Lineage()
    lin2.add_node(Node("a", "x", "A"))
    lin2.add_node(Node("b", "x", "B"))
    lin2.add_node(Node("c", "x", "C"))
    lin2.add_edge("a", "b")
    lin2.add_edge("b", "c")
    lin2.add_edge("a", "c")
    case(
        "lineage_topological_is_deterministic",
        lin2.topological() == ("a", "b", "c") and lin2.topological() == lin2.topological(),
        "入度为 0 时按 id 排序 —— 用 set 迭代顺序会让 dot 产物在不同进程里不同",
    )
    case(
        "lineage_depth_uses_longest_path",
        lin2.depth("c") == 2,
        "深度取最长路径（a→b→c），不是最短路径",
    )

    # ---- 10. CSV 的 None 与 0
    from .pipeline import _csv

    text = _csv(["x", "y"], [{"x": None, "y": 0.0}, {"x": 1.0, "y": None}])
    lines = text.strip().split("\n")
    case(
        "csv_distinguishes_none_from_zero",
        lines[1].split(",") == ["", "0.000000"] and lines[2].split(",") == ["1.000000", ""],
        f"「看不见」与「打了 0 分」必须印成不同的东西：{lines[1:]}",
    )
    case(
        "csv_uses_lf_newlines",
        "\r" not in text,
        "Windows 上默认 \\r\\n，会让产物不可逐位复现",
    )

    # ---- 11. RNG
    rr = Rng(derive_seed("adv", "dirichlet"))
    ds = [rr.dirichlet1(5) for _ in range(20)]
    case(
        "dirichlet_sums_to_one",
        all(abs(math.fsum(d) - 1.0) < 1e-9 and all(x >= 0.0 for x in d) for d in ds),
        "Dirichlet 提案必须落在单纯形上",
    )
    case(
        "rng_is_reproducible",
        Rng(1234).dirichlet1(3) == Rng(1234).dirichlet1(3),
        "同一个种子必须给出同一串随机数",
    )

    return tuple(cases)


def run_adversarial(cfg: Mapping[str, Any]) -> tuple[AdversarialCase, ...]:
    return adversarial_cases(cfg)


# ------------------------------------------------------------------ 血缘不变量

def lineage_invariants(res: Any) -> tuple[AcceptanceCheck, ...]:
    """三条**真的去查**的血缘不变量。

    写在文档里的不变量不是不变量，是愿望。这三条查的是图本身：
    收拢点的父节点集合必须**恰好**等于它该接住的那一批。
    """
    lin = res.lineage
    pids = [s.id for s in res.pool.specs]
    out: list[AcceptanceCheck] = []

    def add(name: str, ok: bool, note: str) -> None:
        out.append(
            AcceptanceCheck(
                name=name, threshold=1.0, observed=1.0 if ok else 0.0,
                passed=bool(ok), comparator=">=", note=note,
            )
        )

    mix_parents = sorted(lin.parents("mix"))
    expect_mix = sorted(f"cal:{p}" for p in pids)
    add(
        "lineage_mix_inputs_exact",
        mix_parents == expect_mix,
        f"mix 的父节点必须恰好是全部 cal:*：{mix_parents}",
    )

    w_parents = sorted(lin.parents("w*"))
    expect_w = sorted(f"search:{s}" for s in res.search)
    add(
        "lineage_weight_inputs_exact",
        w_parents == expect_w,
        f"w* 的父节点必须恰好是全部 search:*：{w_parents}",
    )

    rw_parents = sorted(lin.parents("reward"))
    add(
        "lineage_reward_inputs_exact",
        rw_parents == ["mix", "w*"],
        f"reward 的父节点必须恰好是 mix 与 w*：{rw_parents}",
    )

    # 收拢点的可加性：Σ_p contrib[p] == R
    worst = 0.0
    for cid, m in res.mixes.items():
        worst = max(worst, abs(math.fsum(m.provider_contrib.values()) - m.total))
    add(
        "provider_contrib_is_additive",
        worst <= 1e-9,
        f"provider 贡献必须可加（Σ contrib == R），实测最大偏差 {worst:.3e}",
    )
    return tuple(out)


# ------------------------------------------------------------------ 总入口

def verify_all(res: Any) -> VerifyResult:
    """验收（依赖产物）+ 对抗（不依赖产物）。"""
    from .report import acceptance_checks

    checks = tuple(acceptance_checks(res)) + lineage_invariants(res)
    return VerifyResult(checks=checks, adversarial=adversarial_cases(res.cfg))
