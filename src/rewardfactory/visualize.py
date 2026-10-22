"""自包含 HTML 报告。**零外部资源**：没有 `<link>`、没有 `<script>`、没有 `<img>`、
没有 `@import`、没有 `fetch`。所有图形都是内联 SVG，所有样式都在 `<style>` 里。

这条约束不是洁癖：报告要能拷到任何地方打开。一个引用了 CDN 的图，
在离线的评审机上就是一块空白 —— 而评审人不会告诉你「你的图没显示」，
他只会说「这个结论我看不出来」。

**浅色主题**：产物要打印、要贴进文档、要在亮色环境里看。深色图表贴进白底文档
会变成一块黑洞。

**所有 SVG 都是合法 XML**：属性值必须加引号（`stroke-dasharray="4,3"` 而不是
`stroke-dasharray=4,3`）。HTML 解析器容忍不合法 XML，但把它当图片嵌进别的地方
（或喂给 XML 解析器）就会炸 —— 这类错误肉眼看不出来，因为浏览器会正常显示。

`created_at` 不出现在 HTML 里，理由与 `report.md` 相同。
"""

from __future__ import annotations

import html
import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import BASELINE_LABELS, COMPONENTS

__all__ = ["render_html"]

_C = {
    "bg": "#ffffff",
    "panel": "#f7f8fa",
    "line": "#e3e6ea",
    "text": "#1f2328",
    "muted": "#6b7280",
    "mix": "#2563eb",
    "best": "#16a34a",
    "naive": "#dc2626",
    "oracle": "#9333ea",
    "chance": "#9ca3af",
    "w": "#2563eb",
    "truth": "#f59e0b",
    "grid": "#e5e7eb",
}

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 28px 64px; background: #ffffff; color: #1f2328;
  font: 14px/1.65 -apple-system, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -.01em; }
h2 { font-size: 17px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e3e6ea; }
h3 { font-size: 14px; margin: 20px 0 8px; color: #374151; }
p, li { margin: 6px 0; }
.sub { color: #6b7280; font-size: 13px; }
.wrap { max-width: 1180px; margin: 0 auto; }
.panel { background: #f7f8fa; border: 1px solid #e3e6ea; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }
.kv { display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 13px; }
.kv b { color: #111827; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border-bottom: 1px solid #eceef1; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; color: #374151; white-space: nowrap; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.hi td { background: #eff6ff; }
tr.bad td { background: #fef2f2; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 12.5px;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
.bar { height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.ok { background: #dcfce7; color: #166534; }
.no { background: #fee2e2; color: #991b1b; }
.warn { background: #fef3c7; color: #92400e; }
.note { color: #6b7280; font-size: 12.5px; margin-top: 4px; }
ul { padding-left: 20px; }
.flex { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.flex > * { flex: 1 1 320px; }
"""


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _f(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _e(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:  # nan：无定义，不是 0%
        return "—"
    return f"{f * 100:.{nd}f}%"


def _tbl(headers: Sequence[str], rows: Iterable[Sequence[Any]], numeric: Sequence[int] = (), hi: Sequence[int] = ()) -> str:
    th = "".join(
        f'<th class="{"n" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = ' class="n"' if i in numeric else ""
            tds.append(f"<td{cls}>{c}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _hbar(value: float, vmax: float, color: str, width: int = 150) -> str:
    w = max(0, min(width, int(round(width * (value / vmax))))) if vmax > 0 else 0
    return (
        f'<span class="bar" style="width:{w}px;background:{color}"></span> '
        f'<span style="font-variant-numeric:tabular-nums">{value:.4f}</span>'
    )


def _method_chart(products: Mapping[str, Any]) -> str:
    """八个方法的一致性条形图。**每个方法一个颜色**，一眼看出组合在哪一档。"""
    methods = products["benchmark.json"]["methods"]
    if not methods:
        return ""
    vmax = max(0.5, max(float(m["agreement"]) for m in methods))
    colors = {
        "chance": _C["chance"],
        "naive_raw_mean": _C["naive"],
        "best_single_raw": "#f97316",
        "calibrated_scalar_mean": "#0ea5e9",
        "best_single_calibrated": _C["best"],
        "component_equal_w": "#14b8a6",
        "component_searched_w": _C["mix"],
        "oracle_truth_w": _C["oracle"],
    }
    w, rowh, pad_l, pad_t = 860, 30, 250, 16
    h = pad_t + rowh * len(methods) + 10
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_C["bg"]}"/>')
    for i, m in enumerate(methods):
        y = pad_t + i * rowh
        name = str(m["method"])
        v = float(m["agreement"])
        barw = int(round((w - pad_l - 90) * (v / vmax)))
        col = colors.get(name, "#64748b")
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + 15}" text-anchor="end" font-size="12.5" '
            f'fill="{_C["text"]}">{_e(name)}</text>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 4}" width="{barw}" height="16" rx="3" fill="{col}"/>'
        )
        parts.append(
            f'<text x="{pad_l + barw + 8}" y="{y + 17}" font-size="12.5" '
            f'fill="{_C["muted"]}">{v * 100:.2f}%</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t - 4}" x2="{pad_l}" y2="{h - 10}" '
        f'stroke="{_C["line"]}" stroke-width="1"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _weights_chart(products: Mapping[str, Any]) -> str:
    """`w*` vs 注入真值 —— 并排条形。**看的是「CI 有多宽」而不是「点估计对不对」。**"""
    wts = products["weights.json"]
    boot = {e["component"]: e for e in wts["bootstrap"]["estimates"]}
    n = len(COMPONENTS)
    rowh, pad_l, pad_t = 44, 130, 20
    w = 820
    h = pad_t + rowh * n + 26
    span = w - pad_l - 130
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_C["bg"]}"/>')
    for i, k in enumerate(COMPONENTS):
        y = pad_t + i * rowh
        est = boot.get(k, {})
        lo, hi = float(est.get("lo", 0.0)), float(est.get("hi", 0.0))
        wv, tv = float(wts["w"][k]), float(wts["w_truth"][k])
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + 16}" text-anchor="end" font-size="12.5" '
            f'fill="{_C["text"]}">{_e(k)}</text>'
        )
        parts.append(
            f'<rect x="{pad_l + lo * span}" y="{y + 2}" width="{max(1, (hi - lo) * span)}" '
            f'height="20" rx="3" fill="#dbeafe" stroke="#bfdbfe"/>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 4}" width="{max(0, wv * span)}" height="7" rx="2" fill="{_C["w"]}"/>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 13}" width="{max(0, tv * span)}" height="7" rx="2" fill="{_C["truth"]}"/>'
        )
        parts.append(
            f'<text x="{pad_l + span + 8}" y="{y + 17}" font-size="12" fill="{_C["muted"]}">'
            f'w*={wv:.3f} / 真值={tv:.3f}</text>'
        )
    parts.append(
        f'<text x="{pad_l}" y="{h - 6}" font-size="12" fill="{_C["muted"]}">'
        f'浅蓝条 = {int(float(wts["bootstrap"]["ci_level"]) * 100)}% bootstrap CI；'
        f'蓝条 = w*；橙条 = 注入真值</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _forward_chart(products: Mapping[str, Any]) -> str:
    """`forward` 的路径：**单源 → 组合**。纵轴是训练集一致性。"""
    path = products["search.json"].get("forward_path") or []
    if len(path) < 2:
        return '<p class="note">（forward 路径不足两步，无可画）</p>'
    vals = [float(r["agreement"]) for r in path]
    lo, hi = min(vals), max(vals)
    span = max(1e-6, hi - lo)
    w, h, pad_l, pad_t, pad_b = 820, 210, 70, 20, 46
    n = len(vals)
    step = (w - pad_l - 30) / max(1, n - 1)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_C["bg"]}"/>')
    for g in range(5):
        gy = pad_t + (h - pad_t - pad_b) * g / 4
        gv = hi - span * g / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - 30}" y2="{gy:.1f}" stroke="{_C["grid"]}"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11.5" '
            f'fill="{_C["muted"]}">{gv * 100:.1f}%</text>'
        )
    pts = []
    for i, r in enumerate(path):
        x = pad_l + i * step
        y = pad_t + (h - pad_t - pad_b) * (1.0 - (float(r["agreement"]) - lo) / span)
        pts.append((x, y, r))
    parts.append(
        '<polyline fill="none" stroke="'
        + _C["mix"]
        + '" stroke-width="2" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        + '"/>'
    )
    for i, (x, y, r) in enumerate(pts):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{_C["mix"]}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="11.5" '
            f'fill="{_C["text"]}">{float(r["agreement"]) * 100:.2f}%</text>'
        )
        sup = "+".join(r["support"])
        label = sup if len(sup) <= 22 else sup[:20] + "…"
        parts.append(
            f'<text x="{x:.1f}" y="{h - pad_b + 18}" text-anchor="middle" font-size="11" '
            f'fill="{_C["muted"]}">{_e(label)}</text>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{h - pad_b + 33}" text-anchor="middle" font-size="11" '
            f'fill="{_C["muted"]}">第 {r["step"]} 步</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _decomp_chart(products: Mapping[str, Any]) -> str:
    """误差分解：每个 provider 的**特有误差**条。混合器那一行是结论。"""
    dec = products["benchmark.json"]["decomposition"]
    n = dec["n_pairs"] or 1
    rows = [(pid, dec["specific"][pid]) for pid in sorted(dec["specific"])]
    rows.append(("混合器（w*）", dec["mixture_specific"]))
    vmax = max([v for _k, v in rows] + [1])
    w, rowh, pad_l, pad_t = 820, 28, 170, 16
    h = pad_t + rowh * len(rows) + 34
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_C["bg"]}"/>')
    for i, (name, v) in enumerate(rows):
        y = pad_t + i * rowh
        bw = int(round((w - pad_l - 120) * (v / vmax)))
        col = _C["mix"] if name.startswith("混合器") else "#94a3b8"
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + 15}" text-anchor="end" font-size="12.5" '
            f'fill="{_C["text"]}">{_e(name)}</text>'
        )
        parts.append(f'<rect x="{pad_l}" y="{y + 4}" width="{bw}" height="16" rx="3" fill="{col}"/>')
        parts.append(
            f'<text x="{pad_l + bw + 8}" y="{y + 17}" font-size="12" fill="{_C["muted"]}">'
            f'{v} / {n}（{v / n * 100:.1f}%）</text>'
        )
    parts.append(
        f'<text x="{pad_l}" y="{h - 10}" font-size="12" fill="{_C["muted"]}">'
        f'共有误差（所有 provider 都错，**不可约**）= {dec["shared"]} / {n}'
        f'（{dec["shared"] / n * 100:.1f}%）</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_html(products: Mapping[str, Any]) -> str:
    """主入口。**只吃产物字典**，与 `render_report` 同一批输入。"""
    world = products["world.json"]
    prov = products["providers.json"]
    cal = products["calibration.json"]
    wts = products["weights.json"]
    srch = products["search.json"]
    bench = products["benchmark.json"]
    ver = products["version.json"]
    exps = products["expectations.json"]["rows"]
    accs = products["acceptance.json"]

    mix = next((m for m in bench["methods"] if m["method"] == "component_searched_w"), {})
    bs = next((m for m in bench["methods"] if m["method"] == "best_single_calibrated"), {})
    naive = next((m for m in bench["methods"] if m["method"] == "naive_raw_mean"), {})
    oracle = next((m for m in bench["methods"] if m["method"] == "oracle_truth_w"), {})
    gain = float(mix.get("agreement", 0)) - float(bs.get("agreement", 0))
    n_over = sum(1 for r in exps if r["overturned"])

    P: list[str] = []
    A = P.append
    A('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width,initial-scale=1">')
    A("<title>RewardFactory 运行报告</title>")
    A(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    A("<h1>RewardFactory 运行报告</h1>")
    A(
        f"<p class='sub'>版本 <code>{_e(ver['id'])}</code> ｜ 协议摘要 "
        f"<code>{_e(ver['protocol_digest'][:12])}</code> ｜ 生成时间见 "
        f"<code>version.json</code>（报告本身不印它，否则两份相同输入就不逐位相同）</p>"
    )
    A(
        "<div class='panel'><div class='kv'>"
        f"<span>题面 <b>{world['n_prompts']}</b></span>"
        f"<span>候选 <b>{world['n_candidates']}</b></span>"
        f"<span>成对标注 <b>{world['n_pairs']}</b></span>"
        f"<span>anchor 候选 <b>{world['splits']['anchor']['n_candidates']}</b></span>"
        f"<span>train 对 <b>{world['splits']['train']['n_pairs']}</b></span>"
        f"<span>holdout 对 <b>{world['splits']['holdout']['n_pairs']}</b></span>"
        f"<span>provider <b>{ver['n_providers']}</b></span>"
        f"<span>分量 <b>{ver['n_components']}</b></span>"
        f"<span>选中的搜索器 <code>{_e(ver['searched'])}</code></span>"
        "</div></div>"
    )

    # ---- 结论
    A("<h2>0. 一句话结论</h2>")
    A(
        f"<p>五个<b>尺度互不可比</b>的打分器，各自都有一种别处没有的缺陷；把它们"
        f"<b>先校准再组合</b>，holdout 成对一致性 <b>{_pct(mix.get('agreement'))}</b>，"
        f"比最好的单个 provider（{_pct(bs.get('agreement'))}）高 "
        f"<b>{_pct(gain)}</b> 个百分点，距注入真值权重的上界"
        f"（{_pct(oracle.get('agreement'))}）还差 "
        f"{_pct(float(oracle.get('agreement', 0)) - float(mix.get('agreement', 0)))}。</p>"
    )
    A(
        f"<p>而<b>不校准</b>地等权平均同一个池子，只有 <b>{_pct(naive.get('agreement'))}</b> —— "
        f"比最好的单源还差 {_pct(float(bs.get('agreement', 0)) - float(naive.get('agreement', 0)))}。"
        f"「Reward 不应该只有一个模型」是对的，但<b>也别随便把它们加起来</b>。</p>"
    )

    # ---- 方法对照
    A("<h2>1. holdout 评测：八个方法同台</h2>")
    A(
        f"<p class='sub'>holdout {bench['n_pairs']} 对；实测标签噪声率 "
        f"{_pct(bench['label_noise_measured'])}（注入 12%）；"
        f"过拟合差额（训练集 − holdout）= {_pct(bench['overfit_gap'])}</p>"
    )
    A(_method_chart(products))
    A(
        _tbl(
            ["方法", "说明", "一致性", "平局率", "果断率", "果断时的一致性"],
            [
                [
                    f"<code>{_e(m['method'])}</code>",
                    _e(BASELINE_LABELS.get(m["method"], "")),
                    _pct(m["agreement"]),
                    _pct(m["tie_rate"]),
                    _pct(m["decisive_rate"]),
                    _pct(m["agreement_decisive"]),
                ]
                for m in bench["methods"]
            ],
            numeric=(2, 3, 4, 5),
        )
    )
    A(
        "<p class='note'>三列必须一起看：<b>一致性</b>可以被「全判平局」刷成 0，"
        "<b>果断时的一致性</b>可以被「只在有把握时下结论」刷成 1，<b>平局率</b>就是那个代价。</p>"
    )

    # ---- 权重
    A("<h2>2. 权重搜索：连「什么才算好」也是搜出来的</h2>")
    A(f"<p><code>R = Σ_k w[k]·C[k]</code>，<code>w</code> 从人类偏好标注里搜出来。</p>")
    A(_weights_chart(products))
    boot = bench["bootstrap"]
    A(
        f"<p class='sub'>bootstrap 重采样 {boot['n_resamples']} 次（每次重跑 "
        f"<code>{_e(boot['searcher'])}</code>，预算 {srch['budget']}）；"
        f"CI 覆盖了 <b>{boot['n_covered']}/{boot['n_components']}</b> 个分量的注入真值。</p>"
    )
    A(
        _tbl(
            ["分量", "w*", "真值", "点误差", "CI 均值", "CI 下界", "CI 上界", "CI 宽度", "覆盖真值"],
            [
                [
                    _e(k),
                    _f(wts["w"][k], 4),
                    _f(wts["w_truth"][k], 4),
                    _f(wts["point_error"][k], 4),
                    _f(e["mean"], 4),
                    _f(e["lo"], 4),
                    _f(e["hi"], 4),
                    _f(e["width"], 4),
                    "<span class='badge ok'>是</span>" if e["covers_truth"] else "<span class='badge no'>否</span>",
                ]
                for k, e in ((k, next(x for x in boot["estimates"] if x["component"] == k)) for k in COMPONENTS)
            ],
            numeric=(1, 2, 3, 4, 5, 6, 7),
        )
    )

    # ---- 搜索器
    A("<h2>3. 四个搜索器的对照</h2>")
    A(
        f"<p class='sub'>分母 <code>1/{srch['den']}</code> ⇒ 单纯形上 "
        f"<b>{srch['n_lattice']}</b> 个格点，所以 <code>grid</code> 是<b>真正的全局最优</b>。</p>"
    )
    A(
        _tbl(
            ["搜索器", "新增评估", "缓存命中", "最优一致性", "支持集"],
            [
                [f"<code>{_e(n)}</code>", s["n_evals_new"], s["n_evals_cached"], _pct(s["best_agreement"], 3), _e(", ".join(s["support"]) or "—")]
                for n, s in sorted(srch["searchers"].items())
            ],
            numeric=(1, 2, 3),
        )
    )
    A(f"<h3>3.1 <code>forward</code> 的路径：单源 → 组合</h3>")
    A(_forward_chart(products))
    A(
        "<p class='note'>每一步不一定都比上一步好 —— <code>forward</code> 一旦加不动就停。"
        "「加分量总是更好」是个<b>会被推翻</b>的期望，代码不许假设它成立。</p>"
    )

    # ---- provider
    A("<h2>4. provider 档案：声明 vs 实测</h2>")
    A(
        _tbl(
            ["provider", "种类", "尺度族", "声明方向", "实测方向", "盲区", "覆盖", "MAE 前", "MAE 后(CV)", "ρ", "单源(原始)", "单源(校准)", "特有误差"],
            [
                [
                    f"<code>{_e(p['provider_id'])}</code>",
                    _e(p["kind"]),
                    f"<code>{_e(p['scale'])}</code>",
                    f"{p['declared_direction']:+d}",
                    f"{p['detected_direction']:+d}" + ("" if p["direction_correct"] else " <span class='badge no'>反</span>"),
                    _e(", ".join(p["blind_components"]) or "—"),
                    _pct(p["coverage"]),
                    _f(p["mae_before"], 3),
                    _f(p["mae_after"], 3),
                    _f(p["reliability"], 3),
                    _pct(p["solo_agreement"]),
                    _pct(p["solo_agreement_cal"]),
                    f"{p['specific_error']} / {p['n_pairs']}",
                ]
                for p in prov["reports"]
            ],
            numeric=(3, 4, 6, 7, 8, 9, 10, 11, 12),
        )
    )
    A(
        "<p class='note'>「单源」两列与主方法<b>同口径</b>：对<b>干净标签</b> "
        "<code>sign(Δq)</code> 打分、平局算<b>半分</b>。不用「与人类标签逐对相等」"
        "—— 那个口径被平局率主导：一个三档打分的 provider 有 97% 的候选输出同一个值，"
        "人类标签口径会判它 13%，而那 13% 说的是「它不表态」，不是「它错了 87%」。</p>"
    )

    # ---- 校准
    A("<h2>5. 校准</h2>")
    A(
        f"<p>默认方法 <code>{_e(cal['method'])}</code> —— {_e(cal['note'])}。"
        f"全分量平均 MAE <b>{_f(cal['mean_mae_before'], 4)} → {_f(cal['mean_mae_after'], 4)}</b>"
        f"（4 折<b>按 x 分层</b>交叉验证口径）。</p>"
    )
    rows = []
    for m in cal["maps"]:
        rows.append(
            [
                f"<code>{_e(m['provider_id'])}</code>",
                _e(m["component"]),
                _f(m["mae_before"], 3),
                _f(m["mae_after"], 3),
                _f(m["mae_fit"], 3),
                _f(m["reliability"], 3),
                m["n_points"],
                "<span class='badge warn'>盲</span>" if m["blind"] else "",
            ]
        )
    A(_tbl(["provider", "分量", "MAE 前", "MAE 后(CV)", "MAE 拟合", "ρ", "断点数", ""], rows, numeric=(2, 3, 4, 5, 6)))
    A(
        "<p class='note'><b>MAE 拟合</b>与 <b>MAE 后(CV)</b> 的差就是「这个方法有多容易过拟合 anchor」。"
        "保序回归的拟合误差必然最小（自由度 = anchor 个数），所以<b>只用拟合误差选校准法，一定会选中它</b>。</p>"
    )

    # ---- 误差分解
    A("<h2>6. 误差分解：组合<b>为什么</b>有用</h2>")
    dec = bench["decomposition"]
    A(
        f"<p class='sub'>用干净标签（<code>|Δq| &gt; 0.02</code> 的 {dec['n_pairs']} 对，标签取 "
        f"<code>sign(Δq)</code>）算，于是「不可约误差」才有意义。</p>"
    )
    A(_decomp_chart(products))
    A(
        "<p class='note'>每个 provider 的特有误差来自它独有的那一种偏差"
        "（<code>rm_large</code> 的长度偏差、<code>llm_judge</code> 的家族偏好、"
        "<code>rule</code> 的判据外全盲），而这些偏差彼此<b>正交</b> —— "
        "所以组合能把它们平均掉。混合器那一行显著低于任何单个 provider，就是这件事真的发生了。</p>"
    )

    # ---- 不确定度
    A("<h2>7. 不确定度：它是<b>参与判定</b>的</h2>")
    A(
        f"<p>判据：<code>|ΔR| ≤ κ·(u_a + u_b)</code> ⇒ 判平局。估小了会强迫在含糊的对上硬选，"
        f"估大了会把所有对都判平 —— 两个方向都掉一致性。</p>"
    )
    A(
        _tbl(
            ["指标", "实测", "含义"],
            [
                ["ECE（期望校准误差）", _f(bench["ece"], 4), "报的不确定度与真实误差差多少（越小越好）"],
                ["u 的均值", _f(bench["uncertainty_mean"], 4), "报出去的平均不确定度"],
                ["u 与 |R−真值| 的 Spearman", _f(bench["uncertainty_mae_correlation"], 4), "不确定度有没有指向「我不确定的地方」"],
            ],
            numeric=(1,),
        )
    )
    A(
        "<p class='note'>一个必须说清的<b>偏保守</b>来源：<code>u</code> 的输入是「校准后分量 vs "
        "<b>人类分量评分</b>」的残差，而人类评分自己有噪声（σ=0.08），这块噪声必然进到 "
        "<code>u</code> 里。你手上只有带噪声的标注，误差估计里就必然含标注噪声 —— "
        "这不是 bug，是信息论上的必然。</p>"
    )

    # ---- 期望
    A(f"<h2>8. 设计期望 vs 实测（<b>{n_over} 条被推翻</b>）</h2>")
    A(
        _tbl(
            ["期望", "声明", "实测", "结论", "实测说明"],
            [
                [
                    f"<code>{_e(r['name'])}</code>",
                    "成立" if r["expected"] else "不成立",
                    "成立" if r["observed"] else "不成立",
                    "<span class='badge ok'>一致</span>" if r["held"] else "<span class='badge no'>被推翻</span>",
                    _e(r["note"]),
                ]
                for r in exps
            ],
        )
    )

    # ---- 验收
    A("<h2>9. 验收</h2>")
    A(
        f"<p>门槛 <b>{accs['n_passed']}/{accs['n_checks']}</b> 通过；"
        f"对抗用例 <b>{accs['adversarial_passed']}/{accs['n_adversarial']}</b>"
        f"（<b>不依赖产物</b> —— 「检测器还能不能工作」与「这一批产物对不对」是两件事）。</p>"
    )
    A(
        _tbl(
            ["验收项", "门槛", "实测", "通过"],
            [
                [
                    f"<code>{_e(c['name'])}</code>",
                    f"{_e(c['comparator'])} {_f(c['threshold'])}",
                    _f(c["observed"]),
                    "<span class='badge ok'>✓</span>" if c["passed"] else "<span class='badge no'>✗</span>",
                ]
                for c in accs["checks"]
            ],
            numeric=(1, 2),
        )
    )

    # ---- 血缘
    lin = products["lineage.json"]
    A("<h2>10. 血缘</h2>")
    A(
        f"<p>{lin['n_nodes']} 个节点 / {lin['n_edges']} 条边，指纹 "
        f"<code>{_e(lin['fingerprint'][:12])}</code>。"
        f"三条不变量在 <code>acceptance.py</code> 里真的被查："
        f"<code>mix</code> 的父节点恰好是全部 <code>cal:*</code>；"
        f"<code>w*</code> 的父节点恰好是全部 <code>search:*</code>；"
        f"<code>reward</code> 的父节点恰好是 <code>mix</code> 与 <code>w*</code>。</p>"
    )

    A(
        "<h2>11. 决定论与只读纯度</h2><ul>"
        "<li>只有 <code>version.json</code> 的 <code>created_at</code> 每次不同，其余产物逐位可复现；</li>"
        "<li>报告正文不印生成时间，所以 <code>report.md</code> / <code>report.html</code> 也逐位可复现；</li>"
        "<li>只读命令一律从产物重建，<b>绝不写盘</b>；产物缺失时在内存里重跑，仍然不落盘。</li>"
        "</ul>"
    )

    A("</div></body></html>")
    return "".join(P)
