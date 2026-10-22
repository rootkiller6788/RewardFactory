"""命令行。**读命令不写盘**，写命令只有 `run` / `visualize`。

这不是洁癖：`out/` 里那份结果可能对应另一份配置，用户看一眼报告就会拿它下结论。
如果「看一眼」会顺手改写产物，结论与配置的对应关系就断了。所以：

- 读命令**从产物重建**（`pipeline.read_products`），绝不重跑流水线；
- 产物缺失时，读命令在内存里跑一遍 —— 仍然**不落盘**，并明确告诉你「本次是内存重跑」；
- `acceptance` 的**对抗用例根本不读产物**（`acceptance.py` 只吃配置），
  于是「代码在遇到坏输入时会怎样」不会因为产物过期而失真。

`score` 是这个项目独有的命令：它把 `out/` 里的**校准映射与权重**取出来，
对一个新的输入算一遍 `RewardEvidence`。这是「产物是一个**函数**，不是一个数」
这句话的可执行版本 —— README 第四节那个统一接口，在这里真的被调用了一次。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate import normalize_w
from .config import load_config
from .pipeline import (
    EXTRA_FILES,
    OUT_FILES,
    build_all_products,
    read_products,
    run_pipeline,
    write_outputs,
)
from .schema import COMPONENTS, CalibrationMap, RewardEvidence, label_name
from .visualize import render_html

__all__ = ["main", "build_parser", "READ_ONLY", "WRITE"]

#: 项目根目录（本文件在 `src/rewardfactory/cli.py`）。
ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_CONFIG = os.path.join("configs", "default.json")

READ_ONLY = (
    "summary",
    "providers",
    "calibration",
    "weights",
    "search",
    "benchmark",
    "pairs",
    "scores",
    "evidence",
    "score",
    "expectations",
    "acceptance",
    "lineage",
)

WRITE = ("run", "visualize")


def _out_dir(args: argparse.Namespace) -> Path:
    p = Path(args.out)
    return p if p.is_absolute() else ROOT / p


# ------------------------------------------------------------ --json 的出口

#: 真正的 stdout。`--json` 时命令里的 `print` 会被 `_quiet_prose` 吞掉。
_REAL_STDOUT = sys.stdout


def _emit_json(payload: Any) -> None:
    """`--json` 的唯一出口：**只**打印一段 JSON，不夹任何说明文字。

    `summary --json | jq .` 必须能工作 —— 这要求 stdout 上**恰好**是一段 JSON。
    如果命令里还有 `print("正在跑…")`，管道那一头会拿到一段解析不了的文本。
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    buf = getattr(_REAL_STDOUT, "buffer", None)
    if buf is not None:
        # 显式写 UTF-8：管道那一头的 jq / json.loads 不该受本机代码页影响。
        buf.write(text.encode("utf-8"))
        buf.flush()
    else:
        _REAL_STDOUT.write(text)
        _REAL_STDOUT.flush()


def _configure_console() -> None:
    """让 Windows 默认的 GBK 控制台不会因为排版符号直接崩掉。

    不强制改成 UTF-8：老式 cmd 的代码页不会跟着变，那样中文反而会乱码。
    保留当前编码、只把无法编码的字符替换掉，是这里最稳的做法。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


class _quiet_prose:
    """把命令里的 `print` 全部吞掉，只在 `--json` 时使用。"""

    def __enter__(self) -> "_quiet_prose":
        import io

        self._saved = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *exc: Any) -> bool:
        sys.stdout = self._saved
        return False


def _w(s: Any) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s: Any, width: int, *, right: bool = False) -> str:
    gap = " " * max(0, width - _w(s))
    return (gap + str(s)) if right else (str(s) + gap)


def _row(cells: Sequence[tuple[str, int]], *, right: Sequence[int] = ()) -> str:
    parts = [_pad(v, w, right=i in right) for i, (v, w) in enumerate(cells)]
    return "  " + " ".join(parts).rstrip()


def _header(cells: Sequence[tuple[str, int]], *, right: Sequence[int] = ()) -> str:
    line = _row(cells, right=right)
    return line + "\n  " + "-" * (len(line) - 2)


# ---------------------------------------------------------------------- 取数


def _products(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    """返回 `(产物字典, 是否来自磁盘)`。**永不落盘。**"""
    loaded = read_products(_out_dir(args))
    if loaded is not None:
        return dict(loaded.products), True
    cfg = load_config(args.config, root=ROOT)
    return build_all_products(run_pipeline(cfg)), False


def _note(from_products: bool) -> str:
    return "" if from_products else "  （产物不存在：本次在内存中重跑，未写盘）"


def _f(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:  # nan：无定义，不是 0%
        return "—"
    return f"{f * 100:.{nd}f}%"


def _method(p: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for m in p["benchmark.json"]["methods"]:
        if m["method"] == name:
            return m
    return {}


# ---------------------------------------------------------------------- 命令


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=ROOT)
    res = run_pipeline(cfg)
    paths = write_outputs(res, _out_dir(args))
    bench = res.benchmark
    mix = bench.method("component_searched_w")
    bs = bench.method("best_single_calibrated")
    naive = bench.method("naive_raw_mean")
    oracle = bench.method("oracle_truth_w")
    payload = {
        "version": res.version.to_dict(),
        "n_candidates": len(res.world.candidates),
        "n_pairs": len(res.world.pairs),
        "n_providers": len(res.pool.specs),
        "searched": res.searched,
        "weights": {k: round(v, 6) for k, v in zip(COMPONENTS, res.weights)},
        "agreement": {
            "mixture": mix.agreement,
            "best_single_calibrated": bs.agreement,
            "naive_raw_mean": naive.agreement,
            "oracle_truth_w": oracle.agreement,
        },
        "weight_ci_coverage": bench.bootstrap.coverage,
        "ece": bench.ece,
        "written": [str(p) for p in paths],
    }
    if args.json:
        _emit_json(payload)
        return 0
    print(f"版本 {res.version.id}（协议 {res.version.protocol_digest[:12]}）")
    print(
        f"  题面 {len(res.world.prompts)} / 候选 {len(res.world.candidates)} / "
        f"成对标注 {len(res.world.pairs)}（anchor {len(res.world.pairs_of('anchor'))} / "
        f"train {len(res.world.pairs_of('train'))} / holdout {len(res.world.pairs_of('holdout'))}）"
    )
    print(f"  provider {len(res.pool.specs)} 个 / 分量 {len(COMPONENTS)} 个")
    print(f"  方向检测：", end="")
    print(
        "  ".join(
            f"{d.provider_id}={'+' if d.detected_direction > 0 else '-'}"
            for d in res.normalize.diagnostics
        )
    )
    print(
        f"  校准 {res.calibrate.method}：平均 MAE {_f(res.calibrate.mean_mae_before, 4)} → "
        f"{_f(res.calibrate.mean_mae_after, 4)}"
    )
    print(f"  权重 w* = " + " ".join(f"{k}={_f(v, 3)}" for k, v in zip(COMPONENTS, res.weights)))
    print(f"  搜索器：" + "  ".join(f"{n}({s.n_evals_new})" for n, s in sorted(res.search.items())))
    print(
        f"  holdout 一致性：组合 {_pct(mix.agreement)} / 最好单源 {_pct(bs.agreement)} / "
        f"不校准等权 {_pct(naive.agreement)} / 真值权重上界 {_pct(oracle.agreement)}"
    )
    print(
        f"  权重 CI 覆盖 {bench.bootstrap.n_covered}/{bench.bootstrap.n_components}；"
        f"ECE {_f(bench.ece, 4)}；过拟合差额 {_pct(bench.overfit_gap)}"
    )
    verify = res.ablations.get("_verify")
    print(f"  写出 {len(paths)} 个产物 → {_out_dir(args)}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    out = _out_dir(args)
    products, from_products = _products(args)
    html = render_html(products)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "report.html"
    p.write_text(html, encoding="utf-8", newline="\n")
    if args.json:
        _emit_json({"written": [str(p)], "bytes": len(html.encode("utf-8")), "from_products": from_products})
        return 0
    print(f"写出 {p}（{len(html.encode('utf-8'))} 字节）{_note(from_products)}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    world = products["world.json"]
    ver = products["version.json"]
    bench = products["benchmark.json"]
    wts = products["weights.json"]
    mix = _method(products, "component_searched_w")
    bs = _method(products, "best_single_calibrated")
    naive = _method(products, "naive_raw_mean")
    oracle = _method(products, "oracle_truth_w")
    chance = _method(products, "chance")
    if args.json:
        _emit_json(
            {
                "version": ver,
                "world": world,
                "weights": wts,
                "methods": bench["methods"],
                "providers": bench["providers"],
                "decomposition": bench["decomposition"],
                "bootstrap": bench["bootstrap"],
                "ece": bench["ece"],
                "ablations": bench.get("ablations", {}),
            }
        )
        return 0
    print(f"RewardFactory 版本 {ver['id']}（协议 {ver['protocol_digest'][:12]}）{_note(fp)}")
    print()
    print(
        f"  题面 {world['n_prompts']} / 候选 {world['n_candidates']} / 成对标注 {world['n_pairs']}"
        f"（anchor {world['splits']['anchor']['n_candidates']} 候选带人工评分 / "
        f"train {world['splits']['train']['n_pairs']} 对 / holdout {world['splits']['holdout']['n_pairs']} 对）"
    )
    print(f"  provider {ver['n_providers']} 个 / 分量 {ver['n_components']} 个 / 选中搜索器 {ver['searched']}")
    print()
    print("  注入真值权重：" + "  ".join(f"{k}={_f(v, 3)}" for k, v in wts["w_truth"].items()))
    print("  搜出的 w*    ：" + "  ".join(f"{k}={_f(v, 3)}" for k, v in wts["w"].items()))
    print(
        f"  bootstrap {_pct(bench['bootstrap']['ci_level'], 0)} CI 覆盖 "
        f"{bench['bootstrap']['n_covered']}/{bench['bootstrap']['n_components']} 个分量的真值；"
        f"支持集 {', '.join(wts['support']) or '—'}"
    )
    print()
    print("  holdout 一致性（9 个方法）：")
    print(f"    {'方法':<24} {'一致性':>8}  {'平局':>6}  {'果断时':>8}  {'对干净标签':>10}")
    for m in bench["methods"]:
        mark = " ←" if m["method"] in ("component_searched_w", "best_single_calibrated", "naive_raw_mean") else ""
        print(
            f"    {m['method']:<24} {_pct(m['agreement'], 2):>8}  "
            f"平局 {_pct(m['tie_rate'], 1):>6}  果断时 {_pct(m['agreement_decisive'], 2):>8}  "
            f"{_pct(m['agreement_truth'], 2):>10}{mark}"
        )
    print()
    print("  主结论看「对干净标签」一列：对 `sign(Δq)` 算，平局算半分 —— 没有人类噪声，也不罚谨慎。")
    print(
        f"  提升：组合 - 最好单源 = {_pct(float(mix['agreement_truth']) - float(bs['agreement_truth']))}"
        f"（同一口径）；不校准等权平均 {_pct(naive['agreement_truth'])}"
    )
    print(f"  距注入真值权重的上界还差 {_pct(float(oracle['agreement_truth']) - float(mix['agreement_truth']))}")
    print()
    print(f"  误差分解：共有（不可约）{bench['decomposition']['shared']} 对；"
          f"混合器特有 {bench['decomposition']['mixture_specific']} 对")
    print(f"  不确定度：ECE {_f(bench['ece'], 4)}；u 均值 {_f(bench['uncertainty_mean'], 4)}")
    print(f"  验收 {products['acceptance.json']['n_passed']}/{products['acceptance.json']['n_checks']}；"
          f"对抗 {products['acceptance.json']['adversarial_passed']}/{products['acceptance.json']['n_adversarial']}")
    over = [r["name"] for r in products["expectations.json"]["rows"] if r["overturned"]]
    print(f"  期望被推翻 {len(over)} 条：" + (", ".join(over) if over else "无"))
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["providers.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(f"provider 池（{len(doc['providers'])} 个）{_note(fp)}")
    print()
    for p in doc["providers"]:
        print(f"  [{p['id']}] {p['kind']} · 尺度 {p['scale']} · 成本 {_f(p['cost_ms'], 1)} ms")
        print(f"      {p['label']}")
        print(
            "      视力 "
            + "  ".join(f"{k}={_f(v, 2)}" for k, v in p["vision"].items())
            + (f"   全盲：{', '.join(p['blind_declared'])}" if p["blind_declared"] else "")
        )
        print(
            f"      原始标量 均值 {_f(p['raw_mean'], 4)} 方差 {_f(p['raw_variance'], 4)} "
            f"范围 [{_f(p['raw_range'][0], 3)}, {_f(p['raw_range'][1], 3)}]"
            + (f"   长度偏差 {_f(p['length_bias'], 2)}" if p["length_bias"] else "")
            + (f"   家族偏差 {p['family_bias']}" if p["family_bias"] else "")
        )
        if p["note"]:
            print(f"      注：{p['note']}")
    print()
    print("  声明 vs 实测：")
    print(_header([("provider", 12), ("声明", 6), ("实测", 6), ("盲区", 30), ("覆盖", 8), ("ρ", 8), ("MAE前", 8), ("MAE后", 8), ("单源·原", 10), ("单源·校", 10)], right=(1, 2, 4, 5, 6, 7, 8, 9)))
    for p in doc["reports"]:
        print(
            _row(
                [
                    (p["provider_id"], 12),
                    (f"{p['declared_direction']:+d}", 6),
                    (f"{p['detected_direction']:+d}", 6),
                    (", ".join(p["blind_components"]) or "—", 30),
                    (_pct(p["coverage"], 1), 8),
                    (_f(p["reliability"], 3), 8),
                    (_f(p["mae_before"], 3), 8),
                    (_f(p["mae_after"], 3), 8),
                    (_pct(p["solo_agreement"], 1), 10),
                    (_pct(p["solo_agreement_cal"], 1), 10),
                ],
                right=(1, 2, 4, 5, 6, 7, 8, 9),
            )
        )
    print()
    print(f"  {doc['self_uncertainty_note']}")
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["calibration.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(f"校准 · 方法 {doc['method']} —— {doc['note']}{_note(fp)}")
    print(
        f"  全分量平均 MAE {_f(doc['mean_mae_before'], 4)} → {_f(doc['mean_mae_after'], 4)}"
        f"（4 折按 x 分层交叉验证）"
    )
    print()
    print(_header([("provider", 12), ("分量", 14), ("MAE前", 8), ("MAE后CV", 9), ("MAE拟合", 9), ("ρ", 8), ("断点", 6), ("盲", 4)], right=(2, 3, 4, 5, 6)))
    for m in doc["maps"]:
        print(
            _row(
                [
                    (m["provider_id"], 12),
                    (m["component"], 14),
                    (_f(m["mae_before"], 3), 8),
                    (_f(m["mae_after"], 3), 9),
                    (_f(m["mae_fit"], 3), 9),
                    (_f(m["reliability"], 3), 8),
                    (m["n_points"], 6),
                    ("盲" if m["blind"] else "", 4),
                ],
                right=(2, 3, 4, 5, 6),
            )
        )
    print()
    print("  MAE拟合 与 MAE后CV 的差 = 该方法有多容易过拟合 anchor（保序回归必然最极端）。")
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["weights.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(f"权重搜索 · 搜索器 {doc['searched']}{_note(fp)}")
    print(f"  搜索器 argmax 的支持集：{', '.join(doc['support']) or '—'}")
    print(
        "  交付的 w* 是 bootstrap 均值（bagged），不是 argmax —— 下面这张表的 w* 列"
        "可以横跨多个分量，而 argmax 通常落在单纯形的一个顶点上。"
    )
    print()
    boot = {e["component"]: e for e in doc["bootstrap"]["estimates"]}
    print(_header([("分量", 14), ("w*", 8), ("真值", 8), ("点误差", 8), ("CI均值", 8), ("CI下界", 8), ("CI上界", 8), ("CI宽", 8), ("覆盖", 6)], right=(1, 2, 3, 4, 5, 6, 7, 8)))
    for k in COMPONENTS:
        e = boot[k]
        print(
            _row(
                [
                    (k, 14),
                    (_f(doc["w"][k], 4), 8),
                    (_f(doc["w_truth"][k], 4), 8),
                    (_f(doc["point_error"][k], 4), 8),
                    (_f(e["mean"], 4), 8),
                    (_f(e["lo"], 4), 8),
                    (_f(e["hi"], 4), 8),
                    (_f(e["width"], 4), 8),
                    ("是" if e["covers_truth"] else "否", 6),
                ],
                right=(1, 2, 3, 4, 5, 6, 7, 8),
            )
        )
    b = doc["bootstrap"]
    print()
    print(
        f"  bootstrap {b['n_resamples']} 次（重跑 {b['searcher']}）；"
        f"CI 覆盖 {b['n_covered']}/{b['n_components']}（{_pct(b['coverage'], 1)}）"
    )
    print(f"  一致性本身的抽样误差：{_pct(b['agreement_lo'], 2)} ~ {_pct(b['agreement_hi'], 2)}"
          f"（点估计 {_pct(b['point_agreement'], 2)}，std {_f(b['agreement_std'], 4)}）")
    print()
    print(f"  {doc['note']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["search.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(
        f"四个搜索器 · 预算 {doc['budget']} · 分母 1/{doc['den']} => "
        f"单纯形上 {doc['n_lattice']} 个格点{_note(fp)}"
    )
    print()
    print(_header([("搜索器", 12), ("新增评估", 10), ("缓存命中", 10), ("最优一致性", 12), ("支持集", 40)], right=(1, 2, 3)))
    grid = next((s for s in doc["searchers"].values() if s["searcher"] == "grid"), None)
    for name, s in sorted(doc["searchers"].items()):
        print(
            _row(
                [
                    (name, 12),
                    (s["n_evals_new"], 10),
                    (s["n_evals_cached"], 10),
                    (_pct(s["best_agreement"], 3), 12),
                    (", ".join(s["support"]) or "—", 40),
                ],
                right=(1, 2, 3),
            )
        )
    print()
    print(f"  {doc['shared_cache_note']}")
    path = doc.get("forward_path") or []
    if path:
        print()
        print("  forward 的路径（单源 → 组合）：")
        print(_header([("步", 4), ("支持集", 46), ("权重", 40), ("训练集一致性", 12)], right=(0, 3)))
        for r in path:
            wstr = " ".join(f"{k}={_f(v, 3)}" for k, v in r["w"].items() if float(v) > 1e-9)
            print(
                _row(
                    [
                        (r["step"], 4),
                        (", ".join(r["support"]), 46),
                        (wstr, 40),
                        (_pct(r["agreement"], 3), 12),
                    ],
                    right=(0, 3),
                )
            )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["benchmark.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(
        f"holdout 评测 · {doc['n_pairs']} 对（A {doc['label_counts']['a']} / B "
        f"{doc['label_counts']['b']} / 平局 {doc['label_counts']['tie']}）{_note(fp)}"
    )
    print(
        f"  标签噪声率实测 {_pct(doc['label_noise_measured'], 2)}；"
        f"过拟合差额 {_pct(doc['overfit_gap'], 2)}（训练 {_pct(doc['train_agreement'], 2)}）"
    )
    print()
    print("  主结论看最后一列「对干净标签」：对 `sign(Δq)` 算、平局算半分 —— 没有人类噪声，也不罚谨慎。")
    print(_header([("方法", 26), ("一致性", 10), ("平局率", 9), ("果断率", 9), ("果断时", 10), ("对干净标签", 12)], right=(1, 2, 3, 4, 5)))
    for m in doc["methods"]:
        print(
            _row(
                [
                    (m["method"], 26),
                    (_pct(m["agreement"], 2), 10),
                    (_pct(m["tie_rate"], 1), 9),
                    (_pct(m["decisive_rate"], 1), 9),
                    (_pct(m["agreement_decisive"], 2), 10),
                    (_pct(m["agreement_truth"], 2), 12),
                ],
                right=(1, 2, 3, 4, 5),
            )
        )
    dec = doc["decomposition"]
    print()
    print(f"  误差分解（干净标签 {dec['n_pairs']} 对，共有误差 {dec['shared']} 对）：")
    print(_header([("对象", 26), ("特有误差", 10), ("特有误差率", 12)], right=(1, 2)))
    for pid in sorted(dec["specific"]):
        v = dec["specific"][pid]
        print(_row([(pid, 26), (v, 10), (_pct(v / dec["n_pairs"] if dec["n_pairs"] else 0, 2), 12)], right=(1, 2)))
    print(
        _row(
            [
                ("混合器（w*）", 26),
                (dec["mixture_specific"], 10),
                (_pct(dec["mixture_specific"] / dec["n_pairs"] if dec["n_pairs"] else 0, 2), 12),
            ],
            right=(1, 2),
        )
    )
    print()
    print(f"  不确定度：ECE {_f(doc['ece'], 4)}（误差已去偏）；u 均值 {_f(doc['uncertainty_mean'], 4)}")
    print(f"           u 与 |R-真值| 的 Spearman {_f(doc['uncertainty_mae_correlation'], 4)}；"
          f"四分位提升 {_f(doc['uncertainty_lift'], 3)}（最高四分位 u 的 mean|误差| ÷ 最低四分位的）")
    print(f"  尺度错位：mean(R - quality) = {_f(doc['reward_bias'], 4)}"
          f"（reward 只有序意义，常数错位要靠校准修，不归不确定度管）")
    if doc.get("per_family"):
        print()
        print("  分家族：")
        print(_header([("家族", 10), ("对数", 6), ("组合", 10), ("最好单源", 10), ("不校准等权", 12)], right=(1, 2, 3, 4)))
        for fam, row in sorted(doc["per_family"].items()):
            print(
                _row(
                    [
                        (fam, 10),
                        (int(row["n_pairs"]), 6),
                        (_pct(row["component_searched_w"], 2), 10),
                        (_pct(row["best_single_calibrated"], 2), 10),
                        (_pct(row["naive_raw_mean"], 2), 12),
                    ],
                    right=(1, 2, 3, 4),
                )
            )
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    rows = _read_csv(products["pairs.csv"])
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if args.wrong:
        rows = [r for r in rows if r["correct"] == "0"]
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        _emit_json({"n": len(rows), "rows": rows})
        return 0
    print(f"holdout 成对判定（{len(rows)} 行）{_note(fp)}")
    print()
    print(
        _header(
            [
                ("pair_id", 20),
                ("家族", 9),
                ("Δq", 8),
                ("ΔR", 8),
                ("u_pair", 8),
                ("标签", 5),
                ("判定", 5),
                ("对", 3),
                ("噪声", 5),
            ],
            right=(2, 3, 4),
        )
    )
    for r in rows:
        print(
            _row(
                [
                    (r["pair_id"], 20),
                    (r["family"], 9),
                    (_f(r["delta_quality"], 3), 8),
                    (_f(r["delta_r"], 3), 8),
                    (_f(r["u_pair"], 3), 8),
                    (r["label_name"], 5),
                    (label_name(int(r["pred"])), 5),
                    ("√" if r["correct"] == "1" else "×", 3),
                    ("是" if r["noisy"] == "1" else "", 5),
                ],
                right=(2, 3, 4),
            )
        )
    return 0


def cmd_scores(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    rows = _read_csv(products["scores.csv"])
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    rows = sorted(rows, key=lambda r: -float(r["total"] or 0))
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        _emit_json({"n": len(rows), "rows": rows})
        return 0
    print(f"候选分数（{len(rows)} 行，按 total 降序）{_note(fp)}")
    print()
    cols = ["candidate_id", "split", "family", "n_tokens", "quality", "total", "uncertainty"]
    widths = [12, 8, 9, 8, 9, 9, 11]
    print(_header(list(zip(cols, widths)), right=tuple(range(3, 7))))
    for r in rows:
        print(
            _row(
                [
                    (r["candidate_id"], 12),
                    (r["split"], 8),
                    (r["family"], 9),
                    (r["n_tokens"], 8),
                    (_f(r["quality"], 3), 9),
                    (_f(r["total"], 3), 9),
                    (_f(r["uncertainty"], 3), 11),
                ],
                right=(3, 4, 5, 6),
            )
        )
    print()
    print("  quality 是**真值**，只在评测侧可见；total 是组合出来的 reward。")
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["evidence.json"]
    rows = doc["evidences"]
    if args.candidate:
        rows = [e for e in rows if e["candidate_id"] == args.candidate]
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        _emit_json({"interface": doc["interface"], "n": len(rows), "evidences": rows})
        return 0
    print(f"统一接口 `{doc['interface']}` 的返回体（{len(rows)} 条）{_note(fp)}")
    print()
    for e in rows:
        print(f"  {e['candidate_id']}  total={_f(e['total_score'], 4)}  u={_f(e['uncertainty'], 4)}")
        for line in e["evidence"]:
            print(f"      {line}")
        print()
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """**统一接口的可执行版本。**

    两种用法：

    - `score --candidate c0007-2`：把产物里那个候选的 `RewardEvidence` 取出来；
    - `score --text "..."`：把 `out/` 里的**校准映射与权重**取出来，
      对一个全新输入算一遍 —— 这是「产物是一个函数」这句话的证明。

    `--text` 那条路径上，latent 分量是从文本的 `sha256` **确定性派生**的
    （模拟里没有真的打分器），所以返回体里带 `stub: true`。
    不标出来的话，读者会以为这是真的打分结果。
    """
    products, fp = _products(args)
    if args.candidate:
        e = next((x for x in products["evidence.json"]["evidences"] if x["candidate_id"] == args.candidate), None)
        if e is None:
            print(f"没有这个候选：{args.candidate}", file=sys.stderr)
            return 2
        if args.json:
            _emit_json(e)
            return 0
        print(f"{e['candidate_id']}（来自产物）{_note(fp)}")
        print(f"  total_score = {_f(e['total_score'], 6)}")
        print(f"  uncertainty = {_f(e['uncertainty'], 6)}")
        for k, v in e["components"].items():
            print(f"  components[{k}] = {_f(v, 6)}")
        for line in e["evidence"]:
            print(f"  · {line}")
        return 0

    if not args.text:
        print("用法：score --candidate <id>  或  score --text \"<回答>\"", file=sys.stderr)
        return 2
    ev = _stub_score(str(args.text), products)
    if args.json:
        _emit_json(ev.to_dict())
        return 0
    print(f"stub 打分（latent 由 sha256 派生；校准映射与 w* 来自产物）{_note(fp)}")
    print(f"  total_score = {_f(ev.total_score, 6)}")
    print(f"  uncertainty = {_f(ev.uncertainty, 6)}")
    for k, v in ev.components.items():
        print(f"  components[{k}] = {_f(v, 6)}")
    for line in ev.evidence:
        print(f"  · {line}")
    return 0


def _stub_score(text: str, products: Mapping[str, Any]) -> RewardEvidence:
    """对任意文本算一遍 reward。**这是产物可用性的证明。**

    它只吃 `calibration.json` / `aggregate.json` / `weights.json` 三个产物，
    不重跑流水线 —— 也就是说：拿到这三个文件的人，不需要我们的代码就能复现这个分数
    （分段线性表的 `apply` 是 20 行的事）。
    """
    from .rng import Rng, derive_seed

    cfg_providers = {p["id"]: p for p in products["providers.json"]["providers"]}
    diag = {d["provider_id"]: d for d in products["providers.json"]["normalize"]["diagnostics"]}
    maps: dict[tuple[str, str], CalibrationMap] = {}
    for m in products["calibration.json"]["maps"]:
        maps[(m["provider_id"], m["component"])] = CalibrationMap.from_dict(m)
    eff = products["aggregate.json"]["effective_weight"]
    w = products["weights.json"]["w"]

    rng = Rng(derive_seed("stub", text))
    # 从文本长度派生一个「长度 z」—— 与主流水线同口径的做法（用词数近似）。
    n_tokens = float(max(20, min(400, len(text) * 1.6)))
    comps: dict[str, tuple[float | None, ...]] = {}
    for pid, spec in sorted(cfg_providers.items()):
        blind = diag.get(pid, {}).get("blind", {})
        vals: list[float | None] = []
        for k in COMPONENTS:
            vis = float(spec["vision"][k])
            if vis <= 0.0 or blind.get(k):
                vals.append(None)
                continue
            v = 0.5 + vis * (rng.random() - 0.5) * 1.4
            if float(spec.get("length_bias", 0.0)):
                v += float(spec["length_bias"]) * ((n_tokens - 220.0) / 90.0) * 0.15
            vals.append(v)
        comps[pid] = tuple(vals)

    num = [0.0] * len(COMPONENTS)
    den = [0.0] * len(COMPONENTS)
    for pid in sorted(cfg_providers):
        for i, k in enumerate(COMPONENTS):
            e = float(eff.get(pid, {}).get(k, 0.0))
            if e <= 0.0:
                continue
            m = maps.get((pid, k))
            v = comps[pid][i]
            if m is None or v is None:
                continue
            num[i] += e * m.apply(float(v))
            den[i] += e
    Cs = [(num[i] / den[i]) if den[i] > 0 else 0.5 for i in range(len(COMPONENTS))]
    total = math.fsum(float(w[k]) * Cs[i] for i, k in enumerate(COMPONENTS))
    u = math.fsum(float(w[k]) * max(0.02, 0.10) for k in COMPONENTS) / max(1e-9, math.fsum(float(x) for x in w.values()))
    return RewardEvidence(
        candidate_id="<text>",
        prompt_id="<text>",
        total_score=total,
        components={k: Cs[i] for i, k in enumerate(COMPONENTS)},
        uncertainty=u,
        providers={pid: comps[pid][0] or 0.0 for pid in sorted(cfg_providers)},
        evidence=(
            "stub 打分：latent 分量由 sha256(text) 确定性派生（模拟里没有真的 RM）。",
            "校准映射与 w* 来自 out/calibration.json 与 out/weights.json —— 产物在这里被当作一个**函数**调用。",
            f"文本长度 {int(n_tokens)} 词，用于驱动长度偏差项。",
        ),
        stub=True,
    )


def cmd_expectations(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    rows = products["expectations.json"]["rows"]
    if args.json:
        _emit_json({"rows": rows})
        return 0
    n_over = sum(1 for r in rows if r["overturned"])
    print(f"设计期望 vs 实测（{len(rows)} 条，其中 {n_over} 条被推翻）{_note(fp)}")
    print()
    print(_header([("期望", 52), ("声明", 6), ("实测", 6), ("结论", 10)], right=(1, 2)))
    for r in rows:
        print(
            _row(
                [
                    (r["name"], 52),
                    ("成立" if r["expected"] else "不成立", 6),
                    ("成立" if r["observed"] else "不成立", 6),
                    ("一致" if r["held"] else "被推翻", 10),
                ],
                right=(1, 2),
            )
        )
    print()
    for r in rows:
        print(f"  [{r['name']}] {r['claim']}")
        print(f"      实测：{r['note']}")
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["acceptance.json"]
    if args.json:
        _emit_json(doc)
        return 0
    print(f"验收 {doc['n_passed']}/{doc['n_checks']}；对抗 {doc['adversarial_passed']}/{doc['n_adversarial']}{_note(fp)}")
    print()
    print(_header([("验收项", 38), ("比较", 10), ("门槛", 10), ("实测", 12), ("通过", 5)], right=(2, 3)))
    for c in doc["checks"]:
        print(
            _row(
                [
                    (c["name"], 38),
                    (c["comparator"], 10),
                    (_f(c["threshold"], 4), 10),
                    (_f(c["observed"], 4), 12),
                    ("√" if c["passed"] else "×", 5),
                ],
                right=(2, 3),
            )
        )
    print()
    print("  对抗用例（**不读产物**，产物过期时仍能跑）：")
    for a in doc["adversarial"]:
        print(f"    [{'√' if a['passed'] else '×'}] {a['name']}")
        if a["detail"]:
            print(f"        {a['detail']}")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    products, fp = _products(args)
    doc = products["lineage.json"]
    if args.dot:
        text = products["lineage_reward.dot"]
        if args.json:
            _emit_json({"dot": text})
        else:
            print(text, end="")
        return 0
    if args.json:
        _emit_json(doc)
        return 0
    print(f"血缘 {doc['n_nodes']} 节点 / {doc['n_edges']} 边，指纹 {doc['fingerprint'][:12]}{_note(fp)}")
    print()
    parents: dict[str, list[str]] = {}
    for e in doc["edges"]:
        parents.setdefault(e["dst"], []).append(e["src"])
    for n in doc["nodes"]:
        ps = parents.get(n["id"], [])
        print(f"  {n['id']:<20} {n['kind']:<12} ← {' '.join(ps) if ps else '(根)'}")
    print()
    print("  三条不变量：mix 的父节点恰好是全部 cal:*；w* 的恰好是全部 search:*；")
    print("              reward 的恰好是 mix 与 w*。")
    return 0


def _read_csv(text: str) -> list[dict[str, str]]:
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return []
    head = lines[0].split(",")
    out: list[dict[str, str]] = []
    for ln in lines[1:]:
        cells = ln.split(",")
        out.append({h: (cells[i] if i < len(cells) else "") for i, h in enumerate(head)})
    return out


# ---------------------------------------------------------------------- 解析

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rewardfactory",
        description="RewardFactory：多来源 Reward 的校准、组合与权重搜索 —— Objective Function Factory。",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"配置文件（默认 {DEFAULT_CONFIG}）")
    p.add_argument("--out", default="out", help="产物目录（默认 out）")
    p.add_argument("--json", action="store_true", help="只输出 JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 子命令上再挂一份 `--config/--out/--json`，**默认值用 SUPPRESS**。
    # 用普通默认值的话，子解析器会把主解析器已经设好的值覆写掉：
    # `rewardfactory --json summary` 里的 `--json` 会被子命令的 `False` 吃掉，
    # 于是「明明加了 --json 却印了一屏散文」。
    for name in READ_ONLY + WRITE:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        sp.add_argument("--out", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="只输出 JSON")

    sub.choices["pairs"].add_argument("--family", default="", help="只看某个家族")
    sub.choices["pairs"].add_argument("--wrong", action="store_true", help="只看判错的")
    sub.choices["pairs"].add_argument("--limit", type=int, default=0, help="最多几行（0 = 全部）")
    sub.choices["scores"].add_argument("--split", default="", help="只看某个划分")
    sub.choices["scores"].add_argument("--limit", type=int, default=0, help="最多几行（0 = 全部）")
    sub.choices["evidence"].add_argument("--candidate", default="", help="只看某个候选")
    sub.choices["evidence"].add_argument("--limit", type=int, default=0, help="最多几条（0 = 全部）")
    sub.choices["score"].add_argument("--candidate", default="", help="对已有候选打分（从产物取）")
    sub.choices["score"].add_argument("--text", default="", help="对任意文本打分（stub，latent 由 sha256 派生）")
    sub.choices["lineage"].add_argument("--dot", action="store_true", help="输出 DOT")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console()
    args = build_parser().parse_args(argv)
    fn = {
        "run": cmd_run,
        "visualize": cmd_visualize,
        "summary": cmd_summary,
        "providers": cmd_providers,
        "calibration": cmd_calibration,
        "weights": cmd_weights,
        "search": cmd_search,
        "benchmark": cmd_benchmark,
        "pairs": cmd_pairs,
        "scores": cmd_scores,
        "evidence": cmd_evidence,
        "score": cmd_score,
        "expectations": cmd_expectations,
        "acceptance": cmd_acceptance,
        "lineage": cmd_lineage,
    }[args.cmd]
    if args.json:
        with _quiet_prose():
            return fn(args)
    return fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
