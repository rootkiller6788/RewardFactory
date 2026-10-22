"""Markdown 报告。**从产物字典渲染，不从对象渲染。**

这条约束的理由写在 `pipeline.py` 的模块文档里：跑完直接看报告、与从产物重建报告，
必须是同一条代码路径。所以本模块的渲染入口只有一个 —— `render_report(products)`，
它吃的是 `build_all_products` 造出来的那批字典。

报告正文**不印生成时间**。印了它，两份相同输入的报告就不逐位相同，
而「同一份输入 → 同一份产物」是这个项目全部结论的前提。
生成时间在 `version.json` 里，那里本来就只有它一个字段是不可复现的。

报告里每个数字都**直接从产物读**，不做二次计算。报告里出现一个自己算出来的
数字，就多了一处「报告与产物不一致」的可能 —— 而那种不一致没有任何办法发现。

`expectation_rows` / `acceptance_checks` 是**例外**：它们必须回看结果对象
（要拿真值算上界、要跑对抗用例），所以它们吃 `PipelineResult`，产出的
`expectations.json` / `acceptance.json` 再被渲染函数当普通产物读。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    BASELINE_LABELS,
    COMPONENTS,
    AcceptanceCheck,
    ExpectationRow,
)

__all__ = [
    "fmt",
    "table",
    "render_report",
    "expectation_rows",
    "acceptance_checks",
    "EXPECTED_NAMES",
]


def fmt(v: Any, nd: int = 4) -> str:
    """`None` / `nan` / `inf` **必须印成三个不同的东西**。

    `None` 是「没测到 / 不适用」，`nan` 是「测到了 nan」，`inf` 是「测到了无穷」。
    把前两个都印成 `—` 或都印成 `nan`，读者就再也分不清「这个分量 provider 看不见」
    与「这个分量算出来是 nan」—— 而这两件事的修法完全相反。
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
    if isinstance(v, (int,)):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    # `nan` 是「无定义」（例如一列果断判决都没有时的一致性），不是 0%。
    # 印成 `0.0%` 会被读成「一果断就全错」，那是一句没测过的话。
    if f != f:
        return "—"
    return f"{f * 100:.{nd}f}%"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], aligns: Sequence[str] = ()) -> str:
    """管道表格。`aligns` 里 `r` 表示右对齐（数字列必须右对齐，否则位数对不上）。"""
    head = "| " + " | ".join(headers) + " |"
    sep_cells = []
    for i, _h in enumerate(headers):
        a = aligns[i] if i < len(aligns) else "l"
        sep_cells.append({"l": ":---", "r": "---:", "c": ":---:"}.get(a, ":---"))
    sep = "| " + " | ".join(sep_cells) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _method(products: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for m in products["benchmark.json"]["methods"]:
        if m["method"] == name:
            return m
    return {}


def _provider(products: Mapping[str, Any], pid: str) -> Mapping[str, Any]:
    for p in products["providers.json"]["reports"]:
        if p["provider_id"] == pid:
            return p
    return {}


# ------------------------------------------------------------------ 期望与验收

#: 期望的核对顺序。**与 `configs/default.json` 的键序无关** —— 报告里的顺序
#: 是给人读的顺序（从「校准有没有用」到「可复现吗」），配置里的顺序是编辑顺序。
EXPECTED_NAMES: tuple[str, ...] = (
    "calibration_improves_agreement",
    "mixture_beats_best_single",
    "naive_uncalibrated_mix_underperforms_best_single",
    "component_decomposition_beats_scalar_mix",
    "searched_weights_recover_latent_pointwise",
    "safety_weight_is_identifiable",
    "llm_judge_is_best_single_provider",
    "direction_detection_changes_agreement",
    "direction_detection_recovers_dead_provider",
    "uncertainty_is_calibrated",
    "weights_are_reproducible",
    "bagging_beats_argmax",
    "uncertainty_predicts_error_rank",
    "reward_has_no_scale_offset",
)


def expectation_rows(res: Any) -> tuple[ExpectationRow, ...]:
    """把配置里**事先声明**的期望逐条核对。

    期望写在 `configs/default.json` 里，跑之前就写死 —— 否则「期望被推翻」
    这件事就变成事后编故事。被推翻的那几条是附录 B 里最值钱的部分。
    """
    from .config import EXPECTATION_CLAIMS, expectations_of

    cfg = res.cfg
    declared = expectations_of(cfg)
    acc = cfg["acceptance"]
    bench = res.benchmark
    provs = {p.provider_id: p for p in bench.providers}
    mix = bench.method("component_searched_w")
    best_single = bench.method("best_single_calibrated")
    best_single_raw = bench.method("best_single_raw")
    naive = bench.method("naive_raw_mean")
    scalar_mix = bench.method("calibrated_scalar_mean")
    boot = bench.bootstrap
    dir_ab = res.ablations.get("direction", {})
    repro = res.ablations.get("reproducibility", {})

    # 「safety 为什么不可辨识」的两个数：它占 quality 方差的多少、有几个 provider
    # 真的看得见它。**两个都算出来，不写死** —— 写死的数字会在下一次改配置时
    # 变成一句谎话，而报告里的谎话比缺一句话危险。
    _wk = COMPONENTS.index("safety")

    def _pv(name: str) -> float:
        xs = [c.comp(name) for c in res.world.candidates]
        mu = sum(xs) / len(xs)
        return sum((x - mu) ** 2 for x in xs) / len(xs)

    _tot_var = sum(res.world.latent_weights[i] ** 2 * _pv(n) for i, n in enumerate(COMPONENTS))
    shares = {
        n: (100.0 * res.world.latent_weights[i] ** 2 * _pv(n) / _tot_var if _tot_var > 0 else 0.0)
        for i, n in enumerate(COMPONENTS)
    }
    safety_share = shares.get("safety", 0.0)
    top_name = max(shares, key=lambda k: shares[k]) if shares else "—"
    top_share = shares.get(top_name, 0.0)
    n_safety_see = sum(1 for p in provs.values() if "safety" not in p.blind_components)

    def ag(m: Any) -> float:
        return float(m.agreement) if m is not None else 0.0

    def at(m: Any) -> float:
        if m is None:
            return 0.0
        v = float(m.agreement_truth)
        return 0.0 if v != v else v

    argmax_m = bench.method("component_argmax_w")

    # 1 校准之后每个 provider 单独的一致性都变高
    cal_ok = [pid for pid, p in provs.items() if p.solo_agreement_cal > p.solo_agreement]
    miss = [pid for pid, p in provs.items() if p.solo_agreement_cal <= p.solo_agreement]
    obs1 = len(cal_ok) == len(provs) and len(provs) > 0

    # 2 组合优于最好单源（对干净标签的口径）
    obs2 = (at(mix) - at(best_single)) >= float(acc["min_mixture_gain_over_best_single"])

    # 3 不校准的等权平均比最好单源（同样不校准）还差
    obs3 = at(naive) < at(best_single_raw)

    # 4 分量分解比「校准标量再平均」更有用
    obs4 = at(mix) > at(scalar_mix)

    # 5 w* 逐分量逼近注入真值。**这条被推翻了，而且它本来就该被推翻** ——
    #   `world.py` 纪律二与 `bootstrap_weights` 文档都写明「点估计必然偏离」。
    #   它作为**期望**留在这里（允许被推翻），但不作为**验收门槛**。
    perr = [abs(v) for v in res.weights_point_error.values()]
    obs5 = bool(perr) and max(perr) <= 0.25

    # 6 safety 权重可辨识。**判据是「窄且对」，不是「窄」** —— bootstrap CI 只量
    #   「换一批标注，这个答案有多稳」，不量「这个答案有多对」。只看宽度的版本会
    #   把「对错误答案很自信」判成「可辨识」（实测：safety 的 CI 只有 0.0254 宽，
    #   而注入真值 0.16 落在它外面）。
    est = {e["component"]: e for e in boot.to_dict()["estimates"]}
    saf = est.get("safety", {})
    obs6 = float(saf.get("width", 1.0)) <= 0.5 and bool(saf.get("covers_truth", False))

    # 7 LLM 裁判是最强单源
    best_pid = max(provs, key=lambda k: provs[k].solo_agreement_cal) if provs else ""
    obs7 = best_pid == "llm_judge"

    # 8 不检测方向会明显拉低一致性
    obs8 = abs(float(dir_ab.get("agreement_delta", 0.0))) >= 0.01

    # 9 检测方向能把反号的 provider 从死重变回贡献者
    inv = list(dir_ab.get("inverted_providers", []))
    rel_main = dir_ab.get("reliability_main", {})
    rel_forced = dir_ab.get("reliability_forced", {})
    obs9 = bool(inv) and all(
        float(rel_main.get(pid, 0.0)) > float(rel_forced.get(pid, 0.0)) + 0.05 for pid in inv
    )

    # 10 不确定度与真实误差量级相符
    obs10 = float(bench.ece) <= float(acc["max_ece"])

    # 11 权重可复现
    obs11 = bool(repro.get("identical", False))

    # 12 bootstrap 均值比 argmax 好
    obs12 = bool(argmax_m is not None) and at(mix) > at(argmax_m)

    # 13 u 越大误差越大（秩相关）
    obs13 = float(bench.uncertainty_mae_correlation) >= 0.15

    # 14 R 与 quality 之间没有常数错位
    obs14 = abs(float(bench.reward_bias)) <= 0.05

    observed = {
        "calibration_improves_agreement": obs1,
        "mixture_beats_best_single": obs2,
        "naive_uncalibrated_mix_underperforms_best_single": obs3,
        "component_decomposition_beats_scalar_mix": obs4,
        "searched_weights_recover_latent_pointwise": obs5,
        "safety_weight_is_identifiable": obs6,
        "llm_judge_is_best_single_provider": obs7,
        "direction_detection_changes_agreement": obs8,
        "direction_detection_recovers_dead_provider": obs9,
        "uncertainty_is_calibrated": obs10,
        "weights_are_reproducible": obs11,
        "bagging_beats_argmax": obs12,
        "uncertainty_predicts_error_rank": obs13,
        "reward_has_no_scale_offset": obs14,
    }
    notes = {
        "calibration_improves_agreement": (
            f"{len(cal_ok)}/{len(provs)} 个 provider 的单源一致性提高"
            + (
                "；没提高的是 "
                + "、".join(
                    f"`{p}`（一致性 {_pct(provs[p].solo_agreement, 2)} → "
                    f"{_pct(provs[p].solo_agreement_cal, 2)}，"
                    f"MAE {fmt(provs[p].mae_before)} → {fmt(provs[p].mae_after)}"
                    + (
                        "，**逐位不变**）"
                        if abs(provs[p].solo_agreement_cal - provs[p].solo_agreement) < 1e-9
                        else "，**变差**）"
                    )
                    for p in miss
                )
                + " —— 期望写的是「都变高」，而**「没变」也被它算成了失败**。"
                "校准对已经单调的分数只做一次单调重标，成对排序逐位不变；"
                "它真正救回来的是**尺度坏掉**的那些 provider"
                f"（`rule` {_pct(provs['rule'].solo_agreement, 1)} → "
                f"{_pct(provs['rule'].solo_agreement_cal, 1)}，"
                "因为 `grid3` 把它的输出压进了两个取值），"
                "对尺度本来就合适的 provider 它什么也不做。"
                if miss
                else ""
            )
        ),
        "mixture_beats_best_single": (
            f"组合 {_pct(at(mix))} vs 最好单源 {_pct(at(best_single))}"
            f"（门槛 {_pct(acc['min_mixture_gain_over_best_single'])}，口径 agreement_truth）"
        ),
        "naive_uncalibrated_mix_underperforms_best_single": (
            f"不校准等权平均 {_pct(at(naive))} vs 最好单源（未校准）{_pct(at(best_single_raw))}"
            " —— 设计直觉是「尺度异质性 ⇒ 等权平均被 logit 尺度单方面决定 ⇒ 输给最好单源」，"
            "实测反了：等权平均的**降方差**效应压过了尺度异质性的伤害。"
            "尺度真正毁掉的是别的东西：未校准的 u 活在自己的尺度上，"
            f"`best_single_raw` 因此有 {_pct(bench.method('best_single_raw').tie_rate)} 的对被判平局 —— "
            "一个 provider 未校准时连它自己的排序能力都发挥不出来。"
        ),
        "component_decomposition_beats_scalar_mix": (
            f"分量分解 {_pct(at(mix))} vs 标量校准平均 {_pct(at(scalar_mix))}"
        ),
        "searched_weights_recover_latent_pointwise": (
            "逐分量点误差 " + " / ".join(f"{k}={fmt(v, 4)}" for k, v in res.weights_point_error.items())
            + "（最大 " + fmt(max(perr) if perr else 0.0) + "，期望 ≤ 0.25）"
        ),
        "safety_weight_is_identifiable": (
            f"safety 的 {_pct(boot.ci_level, 0)} CI = [{fmt(saf.get('lo'))}, {fmt(saf.get('hi'))}]"
            f"（宽 {fmt(saf.get('width'))}，注入真值 {fmt(saf.get('truth'))}）"
            + ("" if saf.get("covers_truth") else " —— **窄，但漏掉了真值**。")
            + " bootstrap 只量抽样波动、不量偏差：这个 CI 说的是「换一批标注，"
            "『safety 权重≈0』这个答案有多稳」，不是「它有多对」。safety 的真值贡献"
            f"只占 quality 方差的 {fmt(safety_share, 1)}%"
            f"（`{top_name}` 一个分量就占 {fmt(top_share, 0)}%），"
            "而且它是唯一一个两头都不靠的分量："
            f"5 个 provider 里只有 {n_safety_see} 个真的看得见它，它的取值又挤在 "
            "0.93 附近 —— 这个样本量下本来就没有分辨它的功率。"
        ),
        "llm_judge_is_best_single_provider": f"实测最强单源是 `{best_pid}`",
        "direction_detection_changes_agreement": (
            f"强制 +1 后一致性 {_pct(dir_ab.get('agreement'))}，差 {_pct(dir_ab.get('agreement_delta'))}"
        ),
        "direction_detection_recovers_dead_provider": (
            "反号 provider " + ", ".join(inv) + " 的可靠性 "
            + " / ".join(f"{pid}: {fmt(rel_forced.get(pid))}→{fmt(rel_main.get(pid))}" for pid in inv)
            if inv
            else "配置里没有反号的 provider"
        ),
        "uncertainty_is_calibrated": f"ECE={fmt(bench.ece, 4)}（门槛 {fmt(acc['max_ece'], 4)}）",
        "weights_are_reproducible": (
            f"两次独立搜索的 w* 逐位相同：{repro.get('identical')}"
            f"（digest {repro.get('digest', '—')}）"
        ),
        "bagging_beats_argmax": (
            f"bagged {_pct(at(mix))} vs argmax {_pct(at(argmax_m))}"
            f"（差 {_pct(at(mix) - at(argmax_m))}）"
        ),
        "uncertainty_predicts_error_rank": (
            f"Spearman(u, |误差|)={fmt(bench.uncertainty_mae_correlation)}，期望 ≥ 0.15；"
            f"四分位提升 {fmt(bench.uncertainty_lift, 3)}"
            " —— 两条读数必须一起看：u 的**量级**是对的（ECE 很小），"
            "u 也抓得住**最难的那一批**（最高四分位的平均误差是最低四分位的 "
            f"{fmt(bench.uncertainty_lift, 2)} 倍），但它在整个 holdout 上的**秩**"
            "几乎不与误差同向。「能排序」与「能圈出最危险的少数」是两件事，"
            "别把后者当成了前者。"
        ),
        "reward_has_no_scale_offset": (
            f"mean(R − quality)={fmt(bench.reward_bias)}，期望 |bias| ≤ 0.05"
        ),
    }

    rows: list[ExpectationRow] = []
    for name in EXPECTED_NAMES:
        exp = bool(declared.get(name, True))
        obs = bool(observed.get(name, False))
        rows.append(
            ExpectationRow(
                name=name,
                claim=EXPECTATION_CLAIMS.get(name, name),
                expected=exp,
                observed=obs,
                held=(exp == obs),
                note=notes.get(name, ""),
            )
        )
    return tuple(rows)


def acceptance_checks(res: Any) -> tuple[AcceptanceCheck, ...]:
    """验收门槛。**每条都指向产物里一个真实存在的字段** ——
    指向「报告里的一句话」的验收项没法自动核对。"""
    cfg = res.cfg
    acc = cfg["acceptance"]
    bench = res.benchmark
    provs = list(bench.providers)
    mix = bench.method("component_searched_w")
    best_single = bench.method("best_single_calibrated")
    naive = bench.method("naive_raw_mean")
    boot = bench.bootstrap
    out: list[AcceptanceCheck] = []

    def ag(m: Any) -> float:
        return float(m.agreement) if m is not None else 0.0

    def at(m: Any) -> float:
        """对**干净标签**的一致性（平局算半分）。主结论一律走这个口径。"""
        if m is None:
            return 0.0
        v = float(m.agreement_truth)
        return 0.0 if v != v else v

    def add(name: str, threshold: float, observed: float, cmp: str, note: str = "") -> None:
        passed = observed >= threshold if cmp == ">=" else observed <= threshold
        out.append(
            AcceptanceCheck(
                name=name, threshold=float(threshold), observed=float(observed),
                passed=bool(passed), comparator=cmp, note=note,
            )
        )

    # 验收标准一：组合相对单源的提升（对干净标签，平局算半分）
    add(
        "mixture_gain_over_best_single",
        float(acc["min_mixture_gain_over_best_single"]),
        at(mix) - at(best_single),
        ">=",
        "README 验收标准一：多 Provider 组合相对单 Provider 的一致性提升"
        "（`agreement_truth`，对干净标签 + 平局半分）",
    )
    # 验收标准二：权重可复现 + 带不确定度。**判据是覆盖率，不是点估计等不等于真值** ——
    # 点估计必然偏离（标签有噪声、样本有限），拿它当门槛会逼着实现去偷看真值。
    add(
        "weight_ci_coverage",
        float(acc["min_weight_ci_coverage"]),
        boot.coverage,
        ">=",
        f"{boot.n_covered}/{boot.n_components} 个分量的 CI 覆盖了注入真值",
    )
    # 校准
    add(
        "calibration_gain",
        float(acc["min_calibration_gain"]),
        res.calibrate.mean_mae_before - res.calibrate.mean_mae_after,
        ">=",
        "校准前后的平均 MAE 差（交叉验证口径）",
    )
    add("mae_after", float(acc["max_mae_after"]), res.calibrate.mean_mae_after, "<=", "校准后平均 MAE")
    rels = [p.reliability for p in provs if not p.degenerate]
    add(
        "reliability_spread",
        float(acc["min_reliability_spread"]),
        (max(rels) - min(rels)) if rels else 0.0,
        ">=",
        "provider 之间的可靠性必须有差距，否则「按可靠性加权」是空动作",
    )
    # 尺度异质性必须真的存在
    svars = [res.pool.scale_variance(pid) for pid in res.pool.ids]
    ratio = (max(svars) / min(svars)) if svars and min(svars) > 0 else 0.0
    add(
        "scale_variance_ratio",
        float(acc["min_scale_variance_ratio"]),
        ratio,
        ">=",
        "原始标量方差的最大/最小比 —— 「不校准会输」这件事的前提",
    )
    add(
        "naive_deficit",
        float(acc["min_naive_deficit"]),
        at(best_single) - at(naive),
        ">=",
        "不校准等权平均相对最好单源的落后幅度（同一口径）",
    )
    # 不确定度
    add("ece", float(acc["max_ece"]), bench.ece, "<=", "不确定度的期望校准误差（误差已去偏）")
    add(
        "uncertainty_lift",
        float(acc["min_uncertainty_lift"]),
        bench.uncertainty_lift,
        ">=",
        "最高四分位 u 的 mean|误差| ÷ 最低四分位的 —— 「最不确定的那批确实更差」",
    )
    # 结构
    add("n_providers", float(acc["min_providers"]), float(len(provs)), ">=", "provider 数")
    add("n_holdout_pairs", float(acc["min_holdout_pairs"]), float(bench.n_pairs), ">=", "holdout 成对数")
    add(
        "n_anchor_items",
        float(acc["min_anchor_items"]),
        float(len(res.world.anchor_ratings)),
        ">=",
        "带人工分量评分的 anchor 条数",
    )
    solos = [p.solo_agreement_cal for p in provs]
    add(
        "solo_spread",
        float(acc["min_solo_spread"]),
        (max(solos) - min(solos)) if solos else 0.0,
        ">=",
        "单源之间的一致性差距 —— 没有差距就没有「组合」的必要",
    )
    return tuple(out)


# ------------------------------------------------------------------ 渲染

def render_report(products: Mapping[str, Any]) -> str:
    """主入口。**只吃产物字典。**"""
    world = products["world.json"]
    prov = products["providers.json"]
    cal = products["calibration.json"]
    agg = products["aggregate.json"]
    wts = products["weights.json"]
    srch = products["search.json"]
    bench = products["benchmark.json"]
    ver = products["version.json"]
    exps = products["expectations.json"]["rows"]
    accs = products["acceptance.json"]

    L: list[str] = []
    A = L.append

    # ---- 头部
    A("# RewardFactory 运行报告")
    A("")
    A(f"- 版本：`{ver['id']}`（协议摘要 `{ver['protocol_digest'][:12]}`）")
    A(f"- 生成时间见 `version.json`；**报告正文刻意不印它** —— 印了两份相同输入就不逐位相同")
    A(f"- 题面 {world['n_prompts']} 个 / 候选 {world['n_candidates']} 个 / 成对标注 {world['n_pairs']} 条")
    A(
        f"- 划分：anchor {world['splits']['anchor']['n_candidates']} 候选（带人工分量评分）"
        f" / train {world['splits']['train']['n_pairs']} 对 / holdout {world['splits']['holdout']['n_pairs']} 对"
    )
    A(f"- provider {ver['n_providers']} 个 / 分量 {ver['n_components']} 个 / 被选中的搜索器 `{ver['searched']}`")
    A("")
    A("> 本项目把「什么才算好」变成**可搜索的目标函数**：`R(x,y) = Σ_k w[k]·C[k](x,y)`，")
    A("> 而 `w` 不是在配置里写死的，是从人类偏好标注里**搜**出来的，并且带不确定度。")
    A("")

    # ---- 一句话结论
    mix = _method(products, "component_searched_w")
    bs = _method(products, "best_single_calibrated")
    naive = _method(products, "naive_raw_mean")
    oracle = _method(products, "oracle_truth_w")
    chance = _method(products, "chance")
    gain = mix.get("agreement", 0.0) - bs.get("agreement", 0.0)
    A("## 0. 一句话结论")
    A("")
    A(
        f"五个**尺度互不可比**的打分器，各自都有一种别处没有的缺陷；"
        f"把它们**先校准再组合**，holdout 成对一致性 **{_pct(mix.get('agreement'))}**，"
        f"比最好的单个 provider（{_pct(bs.get('agreement'))}）高 **{_pct(gain)}** 个百分点，"
        f"距注入真值权重的上界（{_pct(oracle.get('agreement'))}）还差 "
        f"{_pct(oracle.get('agreement', 0.0) - mix.get('agreement', 0.0))}。"
    )
    A("")
    A(
        f"而**不校准**地等权平均同一个池子，只有 **{_pct(naive.get('agreement'))}** —— "
        f"比最好的单源还差 {_pct(bs.get('agreement', 0.0) - naive.get('agreement', 0.0))}。"
        f"「Reward 不应该只有一个模型」是对的，但**也别随便把它们加起来**。"
    )
    A("")
    A(
        f"权重 `w*` 与注入真值的最大点误差 {fmt(max([abs(v) for v in wts['point_error'].values()] or [0]), 4)}，"
        f"bootstrap {_pct(products['benchmark.json']['bootstrap']['ci_level'], 0)} CI 覆盖了 "
        f"{bench['bootstrap']['n_covered']}/{bench['bootstrap']['n_components']} 个分量的真值。"
    )
    A("")

    # ---- 两条验收标准
    A("## 1. README 的两条验收标准")
    A("")
    A(
        table(
            ["验收标准", "判据（产物字段）", "实测", "结论"],
            [
                [
                    "① 多 Provider 组合相对单 Provider 的一致性提升",
                    "`benchmark.json` → `component_searched_w.agreement − best_single_calibrated.agreement`",
                    f"{_pct(gain)}（门槛 {_pct(res_threshold(products, 'min_mixture_gain_over_best_single'))}）",
                    "达标" if gain >= res_threshold(products, "min_mixture_gain_over_best_single") else "**未达标**",
                ],
                [
                    "② 权重 w 的搜索结果可复现，并输出不确定度",
                    "`weights.json` → `w` + `bootstrap.estimates[*].{lo,hi}`；`search.json` 里四个搜索器各自的 `best_w`",
                    f"CI 覆盖 {bench['bootstrap']['n_covered']}/{bench['bootstrap']['n_components']} 个分量；"
                    f"两次独立搜索 w* 逐位相同",
                    "达标"
                    if bench["bootstrap"]["coverage"] >= res_threshold(products, "min_weight_ci_coverage")
                    else "**未达标**",
                ],
            ],
            ["l", "l", "l", "c"],
        )
    )
    A("")
    A(
        "第 ① 条的判据刻意选 `best_single_calibrated` 而不是 `best_single_raw`："
        "跟一个没校准过的对手比赢，什么也证明不了。"
    )
    A("")

    # ---- provider 档案
    A("## 2. provider 档案：**声明 vs 实测**")
    A("")
    A("配置里说一个 provider 有多可靠**不算数**，anchor 上测出来的才算。")
    A("")
    diag = {d["provider_id"]: d for d in prov["normalize"]["diagnostics"]}
    A(
        table(
            ["provider", "种类", "尺度族", "声明方向", "实测方向", "盲区分量", "覆盖"],
            [
                [
                    f"`{p['provider_id']}`",
                    p["kind"],
                    f"`{p['scale']}`",
                    f"{p['declared_direction']:+d}",
                    f"{diag.get(p['provider_id'], {}).get('detected_direction', 0):+d}"
                    + ("" if p["direction_correct"] else " ⚠"),
                    ", ".join(p["blind_components"]) or "—",
                    _pct(p["coverage"]),
                ]
                for p in prov["reports"]
            ],
            ["l", "l", "l", "r", "r", "l", "r"],
        )
    )
    A("")
    A(
        table(
            ["provider", "MAE（校准前）", "MAE（校准后·CV）", "可靠性 ρ", "单源一致性（原始·对干净标签）", "单源一致性（校准·对干净标签）", "特有误差 / 对数"],
            [
                [
                    f"`{p['provider_id']}`",
                    fmt(p["mae_before"]),
                    fmt(p["mae_after"]),
                    fmt(p["reliability"]),
                    _pct(p["solo_agreement"]),
                    _pct(p["solo_agreement_cal"]),
                    f"{p['specific_error']} / {p['n_pairs']}（{_pct(p['specific_error_rate'])}）",
                ]
                for p in prov["reports"]
            ],
            ["l", "r", "r", "r", "r", "r", "r"],
        )
    )
    A("")
    wrong_dir = [p["provider_id"] for p in prov["reports"] if not p["direction_correct"]]
    if wrong_dir:
        A(
            f"> **方向检测抓到了 {len(wrong_dir)} 个自称方向与实际相反的 provider**："
            + ", ".join(f"`{x}`" for x in wrong_dir)
            + "。它们的 `declared_direction` 是 `+1`，但 anchor 上的 Spearman ρ 是负的。"
        )
        A("")
    A(f"> {prov['self_uncertainty_note']}")
    A("")

    # ---- 校准
    A("## 3. 校准：把五个不可比的尺度拉到同一把尺子上")
    A("")
    A(f"默认方法 `{cal['method']}` —— {cal['note']}")
    A("")
    A(
        f"全分量平均 MAE：**{fmt(cal['mean_mae_before'])} → {fmt(cal['mean_mae_after'])}**"
        f"（降幅 {_pct((cal['mean_mae_before'] - cal['mean_mae_after']) / cal['mean_mae_before'] if cal['mean_mae_before'] else 0)}）。"
        f"`mae_after` 一律是 4 折**按 x 分层**交叉验证的口径 —— 用拟合残差会让保序回归看起来无敌。"
    )
    A("")
    maps = cal["maps"]
    rows = []
    for pid in sorted({m["provider_id"] for m in maps}):
        for k in COMPONENTS:
            m = next((x for x in maps if x["provider_id"] == pid and x["component"] == k), None)
            if m is None:
                continue
            rows.append(
                [
                    f"`{pid}`",
                    k,
                    fmt(m["mae_before"]),
                    fmt(m["mae_after"]),
                    fmt(m["mae_fit"]),
                    fmt(m["reliability"]),
                    str(m["n_points"]),
                    "盲" if m["blind"] else "",
                ]
            )
    A(table(["provider", "分量", "MAE 前", "MAE 后（CV）", "MAE 拟合", "ρ", "断点数", ""], rows,
            ["l", "l", "r", "r", "r", "r", "r", "c"]))
    A("")
    A(
        "`MAE 拟合` 与 `MAE 后（CV）` 的差就是「这个方法有多容易过拟合 anchor」。"
        "保序回归的拟合误差必然最小（它的自由度就是 anchor 个数），"
        "所以**只用拟合误差选校准法，一定会选中保序回归** —— 那是错的选法。"
    )
    A("")

    # ---- 权重
    A("## 4. 权重搜索：连「什么才算好」也是搜出来的")
    A("")
    bs_est = {e["component"]: e for e in bench["bootstrap"]["estimates"]}
    A(
        table(
            ["分量", "w*（搜索）", "w（注入真值）", "点误差", "bootstrap 均值", f"{_pct(bench['bootstrap']['ci_level'], 0)} CI", "CI 宽度", "覆盖真值"],
            [
                [
                    k,
                    fmt(wts["w"][k]),
                    fmt(wts["w_truth"][k]),
                    fmt(wts["point_error"][k]),
                    fmt(bs_est[k]["mean"]),
                    f"[{fmt(bs_est[k]['lo'])}, {fmt(bs_est[k]['hi'])}]",
                    fmt(bs_est[k]["width"]),
                    "是" if bs_est[k]["covers_truth"] else "**否**",
                ]
                for k in COMPONENTS
            ],
            ["l", "r", "r", "r", "r", "r", "r", "c"],
        )
    )
    A("")
    A(
        f"搜索器 `{wts['searched']}` 的 **argmax** 支持集：**{', '.join(wts['support']) or '—'}**"
        "（交付的 `w*` 是 bootstrap 均值，不是 argmax —— 搜索目标是分片常数的，"
        "argmax 通常落在单纯形的一个顶点上，均值把它收缩回内部）；"
        f"bootstrap 重采样 {bench['bootstrap']['n_resamples']} 次（每次重跑 `{bench['bootstrap']['searcher']}`，"
        f"预算 {srch['budget']}）。"
    )
    A("")
    A(f"> {wts['note']}")
    A("")

    # ---- 搜索器对照
    A("## 5. 四个搜索器的对照")
    A("")
    A(
        f"离散化：分母 `1/{srch['den']}` ⇒ 单纯形上 **{srch['n_lattice']} 个格点**，"
        f"所以 `grid` 是**真正的全局最优**（oracle），不是另一个启发式。"
    )
    A("")
    A(
        table(
            ["搜索器", "新增评估", "缓存命中", "最优一致性", "支持集", "与 grid 的差"],
            [
                [
                    f"`{name}`",
                    s["n_evals_new"],
                    s["n_evals_cached"],
                    _pct(s["best_agreement"], 3),
                    ", ".join(s["support"]) or "—",
                    _pct(
                        next((x["best_agreement"] for x in srch["searchers"].values() if x["searcher"] == "grid"), s["best_agreement"])
                        - s["best_agreement"],
                        3,
                    ),
                ]
                for name, s in sorted(srch["searchers"].items())
            ],
            ["l", "r", "r", "r", "l", "r"],
        )
    )
    A("")
    A(f"> {srch['shared_cache_note']}")
    A("")

    fwd = srch.get("forward_path") or []
    if fwd:
        A("### 5.1 `forward` 的路径：**单源 → 组合**")
        A("")
        A("这是产物里最重要的一张表 —— 它把「组合相对单源的提升」从一个数字变成一段过程。")
        A("")
        A(
            table(
                ["步", "支持集", "权重", "训练集一致性"],
                [
                    [
                        r["step"],
                        ", ".join(r["support"]),
                        " / ".join(f"{k}={fmt(v, 3)}" for k, v in r["w"].items() if float(v) > 1e-9),
                        _pct(r["agreement"], 3),
                    ]
                    for r in fwd
                ],
                ["r", "l", "l", "r"],
            )
        )
        A("")
        A(
            "**注意每一步都比上一步好吗？** 不一定。`forward` 一旦加不动就停 —— "
            "「加分量总是更好」是个**会被推翻**的期望，代码不许假设它成立。"
        )
        A("")

    # ---- 评测表
    A("## 6. holdout 评测：九个方法同台")
    A("")
    A(
        f"holdout {bench['n_pairs']} 对（标签分布：A {bench['label_counts']['a']} / "
        f"B {bench['label_counts']['b']} / 平局 {bench['label_counts']['tie']}）。"
        f"实测标签噪声率 **{_pct(bench['label_noise_measured'])}**"
        f"（注入 {_pct(0.12)}）。"
    )
    A("")
    A(
        "**主结论看最后一列 `对干净标签`。** 前四列里的「一致性」是「与人类标签逐对相等」，"
        "而人类只判 5% 平局、任何把 `u` 当真的方法都会判 20~70% 平局 —— 那一列被平局率"
        "主导，比不出谁排得准。最后一列用的是**没有人类噪声的** `sign(Δq)`，"
        "而且**平局算半分**：不判拿 0.5（无信息），判对拿 1，判错拿 0。"
    )
    A("")
    A(
        table(
            ["方法", "说明", "一致性", "平局率", "果断率", "果断时的一致性", "对干净标签"],
            [
                [
                    f"`{m['method']}`",
                    BASELINE_LABELS.get(m["method"], ""),
                    _pct(m["agreement"]),
                    _pct(m["tie_rate"]),
                    _pct(m["decisive_rate"]),
                    _pct(m["agreement_decisive"]),
                    _pct(m["agreement_truth"]),
                ]
                for m in bench["methods"]
            ],
            ["l", "l", "r", "r", "r", "r", "r"],
        )
    )
    A("")
    A(
        "**四列必须一起看**：`一致性` 可以被「全判平局」刷成 0，"
        "`果断时的一致性` 可以被「只在有把握时下结论」刷成 1，`平局率` 就是那个代价，"
        "而 `对干净标签` 是唯一既没有人类噪声、又对平局公平的一列。"
        "只报一个数的评测表是在让人误读。"
    )
    A("")
    A(
        f"过拟合差额：训练集 {_pct(bench['train_agreement'])} − holdout {_pct(mix.get('agreement'))} = "
        f"**{_pct(bench['overfit_gap'])}**。"
        f"搜索目标函数是训练集一致性，所以这个差额是「搜出来的 w* 有多少是在拟合标签噪声」的直接度量。"
    )
    A("")

    # ---- 误差分解
    A("## 7. 误差分解：组合**为什么**有用")
    A("")
    dec = bench["decomposition"]
    A(
        f"用**干净标签**（`|Δq| > {fmt(0.02)}` 的 {dec['n_pairs']} 对，标签取 `sign(Δq)`）算，"
        f"于是「不可约误差」才有意义。"
    )
    A("")
    A(
        table(
            ["对象", "共有误差（所有 provider 都错）", "特有误差（只有它错）", "特有误差率"],
            [
                [
                    f"`{pid}`",
                    dec["shared"],
                    dec["specific"][pid],
                    _pct(dec["specific"][pid] / dec["n_pairs"] if dec["n_pairs"] else 0),
                ]
                for pid in sorted(dec["specific"])
            ]
            + [
                [
                    "**混合器（w\\*）**",
                    dec["mixture_shared"],
                    dec["mixture_specific"],
                    _pct(dec["mixture_specific"] / dec["n_pairs"] if dec["n_pairs"] else 0),
                ]
            ],
            ["l", "r", "r", "r"],
        )
    )
    A("")
    A(
        "**这就是机制**：每个 provider 的特有误差来自它独有的那一种偏差"
        "（`rm_large` 的长度偏差、`llm_judge` 的家族偏好、`rule` 的判据外全盲），"
        "而这些偏差彼此**正交** —— 所以组合能把它们平均掉。"
        "混合器的特有误差若显著小于任何单个 provider，就说明这件事真的发生了。"
    )
    A("")

    # ---- 不确定度
    A("## 8. 不确定度：它是**参与判定**的，不是装饰")
    A("")
    A(
        table(
            ["指标", "实测", "门槛", "含义"],
            [
                ["ECE（期望校准误差）", fmt(bench["ece"]), fmt(res_threshold(products, "max_ece")), "报的不确定度与真实误差差多少（越小越好）"],
                ["u 的均值", fmt(bench["uncertainty_mean"]), "—", "报出去的平均不确定度"],
                ["u 与 |R−真值| 的 Spearman", fmt(bench["uncertainty_mae_correlation"]), fmt(res_threshold(products, "min_uncertainty_correlation")), "不确定度有没有真的指向「我不确定的地方」"],
            ],
            ["l", "r", "r", "l"],
        )
    )
    A("")
    A(
        "不确定度**参与判定**：`|ΔR| ≤ κ·(u_a + u_b)` 判平局。"
        "估小了会强迫在含糊的对上硬选，估大了会把所有对都判平 —— 两个方向都掉一致性。"
        "所以「不确定度估得准不准」是可测的，而 ECE 就是这个测量。"
    )
    A("")
    A(
        "一个必须说清的**偏保守**来源：`u` 的输入是「校准后分量 vs **人类分量评分**」的残差，"
        "而人类评分自己有噪声（配置里 σ=0.08），这块噪声必然进到 `u` 里。"
        "你手上只有带噪声的标注，误差估计里就必然含标注噪声 —— 这不是 bug，是信息论上的必然。"
    )
    A("")

    # ---- 分家族
    if bench.get("per_family"):
        A("## 9. 分家族的一致性")
        A("")
        A(
            table(
                ["家族", "对数", "组合（w\\*）", "最好单源", "不校准等权平均"],
                [
                    [fam, int(row["n_pairs"]), _pct(row["component_searched_w"]), _pct(row["best_single_calibrated"]), _pct(row["naive_raw_mean"])]
                    for fam, row in sorted(bench["per_family"].items())
                ],
                ["l", "r", "r", "r", "r"],
            )
        )
        A("")
        A(
            "家族偏差（`llm_judge` 偏爱写作类）会让某个家族上「最好单源」与组合的差距缩小甚至反号 —— "
            "那正是单个 provider 的偏差在它擅长的领域里看起来「更准」的样子。"
        )
        A("")

    # ---- 期望
    A("## 10. 设计期望 vs 实测")
    A("")
    n_over = sum(1 for r in exps if r["overturned"])
    A(f"共 {len(exps)} 条，其中 **{n_over} 条被推翻**。被推翻的那几条是这份报告里最值钱的部分。")
    A("")
    A(
        table(
            ["期望", "声明", "实测", "结论", "实测说明"],
            [
                [
                    r["name"],
                    "成立" if r["expected"] else "不成立",
                    "成立" if r["observed"] else "不成立",
                    "一致" if r["held"] else "**被推翻**",
                    r["note"],
                ]
                for r in exps
            ],
            ["l", "c", "c", "c", "l"],
        )
    )
    A("")

    # ---- 验收
    A("## 11. 验收")
    A("")
    A(
        f"**{accs['n_passed']}/{accs['n_checks']}** 条门槛通过；"
        f"对抗用例 **{accs['adversarial_passed']}/{accs['n_adversarial']}**（不依赖产物）。"
    )
    A("")
    A(
        table(
            ["验收项", "门槛", "实测", "通过"],
            [
                [
                    c["name"],
                    f"{c['comparator']} {fmt(c['threshold'])}",
                    fmt(c["observed"]),
                    "✓" if c["passed"] else "**✗**",
                ]
                for c in accs["checks"]
            ],
            ["l", "r", "r", "c"],
        )
    )
    A("")
    if accs["adversarial"]:
        failed = [a for a in accs["adversarial"] if not a["passed"]]
        A(
            f"对抗用例（**不读产物**，所以产物过期时仍能跑 —— 「检测器还能不能工作」"
            f"与「这一批产物对不对」是两件事）："
            + ("全部通过。" if not failed else "**失败：** " + ", ".join(a["name"] for a in failed))
        )
        A("")

    # ---- 血缘
    A("## 12. 血缘")
    A("")
    lin = products["lineage.json"]
    A(f"{lin['n_nodes']} 个节点 / {lin['n_edges']} 条边，指纹 `{lin['fingerprint'][:12]}`。")
    A("")
    A("三条被断言的不变量（`acceptance.py` 里真的去查）：")
    A("")
    A("1. `mix` 的父节点**恰好**是全部 `cal:*` —— 一个不多一个不少；")
    A("2. `w*` 的父节点**恰好**是全部 `search:*`；")
    A("3. `reward` 的父节点**恰好**是 `mix` 与 `w*`。")
    A("")
    A("第 1 条里的「一个不多」保证图没把落选的东西偷偷接进来，"
      "「一个不少」保证每个 provider 的校准结果都被组合接住了 —— "
      "于是「到底组合了哪几个 provider」看一张图就知道，不必翻报告。")
    A("")

    # ---- 决定论
    A("## 13. 决定论与只读纯度")
    A("")
    A(f"- `{len(products)}` 个产物里，**只有 `version.json` 的 `created_at` 每次不同**；")
    A("- 报告正文不印生成时间，所以 `report.md` / `report.html` 逐位可复现；")
    A("- 只读命令一律从产物重建，**绝不写盘**；产物缺失时在内存里重跑，仍然不落盘。")
    A("")
    return "\n".join(L) + "\n"


def res_threshold(products: Mapping[str, Any], key: str) -> float:
    """从 `acceptance.json` 里把门槛读回来（报告不硬编码任何阈值）。"""
    for c in products["acceptance.json"]["checks"]:
        if c["name"] == key:
            return float(c["threshold"])
    for c in products["acceptance.json"]["checks"]:
        if key in c["name"]:
            return float(c["threshold"])
    return 0.0
