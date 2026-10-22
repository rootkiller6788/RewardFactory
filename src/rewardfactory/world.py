"""真值层：把「抽象的『好不好』」变成一个**有已知答案**的世界。

这是整个项目的判分基础。README 的验收标准是「多 Provider 组合相对单 Provider 的
一致性提升」与「权重搜索结果可复现并带不确定度」—— 这两句话里**没有一个可观测的
产物字段**（不像「前沿」「矩阵」那样一眼能数）。所以它们必须靠**注入的已知量**来判分：

- 注入五个 latent 分量与真值权重 `w_true` ⇒ `quality = Σ w_true·comp + 残差`；
- 注入一组**带噪声的人类成对标注** ⇒ 一致性有了「正确答案」；
- 注入一个**不可观测残差** ⇒ 任何 provider 都不可能是神谕，一致性上界严格小于 1。

三条纪律：

1. **`quality` 里必须有一块谁也读不到的残差**（`residual_sigma`）。没有它，
   一个「直接读分量」的 provider 就能拿满分，验收标准退化成同义反复
   —— 这与 `PreferenceFactory` 里「真值必须留一层不可观测」是同一条。
2. **人类标注必须有噪声，且近平局要真的被标成平局**。标签噪声是「w* 点估计必然
   偏离真值」的原因，也是「必须报不确定度」的理由；平局是一等判决，不是兜底。
3. **`safety` 分量的方差是刻意压小的**（违规率 6%）。候选间方差极小 ⇒ `w_safety`
   在数学上不可辨识 ⇒ 点估计毫无意义、只有区间有意义。这不是缺陷，是这个项目
   要演示的核心现象之一。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rng import Rng, derive_seed, digest_of
from .schema import (
    COMPONENTS,
    FAMILIES,
    PREFER_A,
    PREFER_B,
    TIE,
    Candidate,
    Pair,
    Prompt,
)

__all__ = ["World", "build_world", "clip", "latent_weights"]

#: 每个家族一小撮词。**文本只是为了读起来像回事** —— 模拟里没有分词器，
#: 真正被 provider 读到的「长度」是 `n_tokens` 字段。
_WORDS: dict[str, tuple[str, ...]] = {
    "arith": ("先", "把", "未知数", "移到", "一边", "，", "再", "两边", "同除", "系数", "，", "得到", "结果"),
    "logic": ("前提", "只有", "一个", "为真", "，", "所以", "结论", "不能", "直接", "推出", "，", "需要", "补充", "条件"),
    "writing": ("这段", "文字", "的", "节奏", "偏", "紧", "，", "如果", "把", "长句", "拆开", "，", "读起来", "会", "顺", "很多"),
    "code": ("这里", "应该", "先", "判断", "空值", "，", "否则", "会", "在", "第", "三", "行", "抛", "异常"),
    "safety": ("这个", "请求", "涉及", "危险", "操作", "，", "我", "不能", "提供", "具体", "步骤", "，", "但", "可以", "讲讲", "原理"),
}


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def latent_weights(cfg: Mapping[str, Any]) -> tuple[float, ...]:
    lw = cfg["world"]["latent_weights"]
    return tuple(float(lw[k]) for k in COMPONENTS)


@dataclass(frozen=True)
class World:
    """一次模拟的全部真值。**只在内存里活着，落盘的只有抽样后的评测用数字。**"""

    prompts: tuple[Prompt, ...]
    candidates: tuple[Candidate, ...]
    pairs: tuple[Pair, ...]
    #: anchor 候选上的**人工分量评分**（带噪声）。校准的唯一输入。
    anchor_ratings: Mapping[str, tuple[float, ...]]
    latent_weights: tuple[float, ...]

    # ---- 查询

    def candidates_of(self, split: str) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.split == split)

    def pairs_of(self, split: str) -> tuple[Pair, ...]:
        return tuple(p for p in self.pairs if p.split == split)

    def candidate(self, cid: str) -> Candidate:
        for c in self.candidates:
            if c.id == cid:
                return c
        raise KeyError(f"没有这个候选：{cid}")

    def prompt(self, pid: str) -> Prompt:
        for p in self.prompts:
            if p.id == pid:
                return p
        raise KeyError(f"没有这个题面：{pid}")

    def rating(self, cid: str) -> tuple[float, ...] | None:
        return self.anchor_ratings.get(cid)

    @property
    def anchor_candidates(self) -> tuple[Candidate, ...]:
        return self.candidates_of("anchor")

    def pair_delta_quality(self, pair: Pair) -> float:
        return pair.delta_quality

    def fingerprint(self) -> str:
        """真值世界的指纹。**不含 `created_at`**，所以同配置两次运行必然相同。"""
        return digest_of(
            {
                "latent_weights": [round(x, 6) for x in self.latent_weights],
                "n_prompts": len(self.prompts),
                "n_candidates": len(self.candidates),
                "n_pairs": len(self.pairs),
                "quality": [round(c.quality, 6) for c in self.candidates],
            }
        )

    def to_dict(self, *, preview: bool = True) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "latent_weights": {k: round(v, 6) for k, v in zip(COMPONENTS, self.latent_weights)},
            "n_prompts": len(self.prompts),
            "n_candidates": len(self.candidates),
            "n_pairs": len(self.pairs),
            "splits": {
                s: {
                    "n_prompts": sum(1 for p in self.prompts if p.split == s),
                    "n_candidates": sum(1 for c in self.candidates if c.split == s),
                    "n_pairs": sum(1 for p in self.pairs if p.split == s),
                    "n_anchor_rated": sum(1 for c in self.candidates if c.split == s and c.id in self.anchor_ratings),
                }
                for s in ("anchor", "train", "holdout")
            },
            "preview": (
                [
                    {
                        "id": c.id,
                        "prompt_id": c.prompt_id,
                        "family": c.family,
                        "n_tokens": c.n_tokens,
                        "quality": round(c.quality, 6),
                        "text": (c.text[:64] + "…") if len(c.text) > 64 else c.text,
                    }
                    for c in self.candidates[:6]
                ]
                if preview
                else []
            ),
        }


# ------------------------------------------------------------------ 生成

def _render(family: str, n_tokens: int, rng: Rng) -> str:
    """把词池拼到大约 `n_tokens` 个词。确定性、无第三方依赖。"""
    pool = _WORDS.get(family, _WORDS["arith"])
    out: list[str] = []
    i = 0
    while len(out) < n_tokens:
        out.append(pool[(i + int(rng.random() * 3)) % len(pool)])
        i += 1
    return "".join(out)


def build_world(cfg: Mapping[str, Any]) -> World:
    """按配置生成真值世界。**完全确定**：同配置必然逐位相同。"""
    w = cfg["world"]
    seed = cfg["seed"]
    root = Rng(derive_seed(seed, "world"))
    lw = latent_weights(cfg)

    n_prompts = int(w["n_prompts"])
    n_cand = int(w["n_candidates"])
    a_n, t_n, h_n = int(w["anchor_prompts"]), int(w["train_prompts"]), int(w["holdout_prompts"])

    prompts: list[Prompt] = []
    candidates: list[Candidate] = []
    pairs: list[Pair] = []
    ratings: dict[str, tuple[float, ...]] = {}

    a_sigma = float(cfg["anchors"]["component_sigma"])
    tie_band = float(w["tie_band"])
    tie_p = float(w["tie_declare_prob"])
    noise_p = float(w["label_noise"])
    res_sigma = float(w["residual_sigma"])
    viol_p = float(w["safety_violation_prob"])
    viol_drop = float(w["safety_violation_drop"])
    lb, lh, lv, ls = (
        float(w["length_base"]),
        float(w["length_help"]),
        float(w["length_verbosity"]),
        float(w["length_sigma"]),
    )
    tok_lo, tok_hi = int(w["min_tokens"]), int(w["max_tokens"])

    for i in range(n_prompts):
        split = "anchor" if i < a_n else ("train" if i < a_n + t_n else "holdout")
        family = FAMILIES[i % len(FAMILIES)]
        pid = f"p{i:04d}"
        prng = root.fork("prompt", pid)
        difficulty = prng.uniform(0.15, 0.85)
        prompts.append(
            Prompt(
                id=pid,
                family=family,
                difficulty=difficulty,
                split=split,
                text=f"[{family}] 题面 {pid}（难度 {difficulty:.2f}）",
            )
        )

        local: list[Candidate] = []
        for j in range(n_cand):
            cid = f"c{i:04d}-{j}"
            crng = root.fork("cand", cid)
            skill = crng.normal(0.0, 0.42)
            style_skill = crng.normal(0.0, 0.50)
            verbosity = crng.normal(0.0, 0.50)
            violation = crng.bernoulli(viol_p)

            help_v = clip(0.50 + 0.55 * skill + 0.25 * (1.0 - difficulty) + crng.normal(0.0, 0.10))
            corr_v = clip(0.45 + 0.70 * skill - 0.30 * difficulty + crng.normal(0.0, 0.12))
            safe_v = clip(0.96 + crng.normal(0.0, 0.04) - (viol_drop if violation else 0.0))
            style_v = clip(0.50 + 0.45 * style_skill + crng.normal(0.0, 0.13))
            task_v = clip(0.50 + 0.60 * skill + 0.20 * (1.0 - difficulty) + crng.normal(0.0, 0.11))
            comps = (help_v, corr_v, safe_v, style_v, task_v)

            n_tokens = int(
                round(
                    clip(
                        lb + lh * help_v + lv * verbosity + crng.normal(0.0, ls),
                        float(tok_lo),
                        float(tok_hi),
                    )
                )
            )
            text = _render(family, max(6, n_tokens // 6), crng)
            core = math.fsum(wk * ck for wk, ck in zip(lw, comps))
            resid = crng.normal(0.0, res_sigma)
            quality = clip(core + resid)
            cand = Candidate(
                id=cid,
                prompt_id=pid,
                family=family,
                split=split,
                text=text,
                n_tokens=n_tokens,
                components=comps,
                quality=quality,
                unobservable=quality - core,
                verbosity=verbosity,
            )
            local.append(cand)
            candidates.append(cand)

            if split == "anchor":
                # 人工分量评分：带噪声。它是「校准的唯一输入」与「可靠性的唯一来源」，
                # 所以噪声不能为 0 —— 否则校准就是在拟合神谕。
                ratings[cid] = tuple(clip(v + crng.normal(0.0, a_sigma)) for v in comps)

        # ---- 成对标注
        for x in range(n_cand):
            for y in range(x + 1, n_cand):
                a, b = local[x], local[y]
                dq = a.quality - b.quality
                prng2 = root.fork("pair", a.id, b.id)
                noisy = False
                if abs(dq) <= tie_band:
                    # 近平局：人类多半会说「差不多」，但被逼着选也会选一个。
                    if prng2.random() < tie_p:
                        label = TIE
                    else:
                        label = PREFER_A if (dq > 0 or (dq == 0 and prng2.random() < 0.5)) else PREFER_B
                else:
                    label = PREFER_A if dq > 0 else PREFER_B
                    if prng2.random() < noise_p:
                        label = -label
                        noisy = True
                pairs.append(
                    Pair(
                        id=f"{a.id}~{b.id}",
                        prompt_id=pid,
                        family=family,
                        split=split,
                        a=a.id,
                        b=b.id,
                        label=label,
                        delta_quality=dq,
                        noisy=noisy,
                    )
                )

    return World(
        prompts=tuple(prompts),
        candidates=tuple(candidates),
        pairs=tuple(pairs),
        anchor_ratings=ratings,
        latent_weights=lw,
    )


def split_summary(world: World) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for s in ("anchor", "train", "holdout"):
        out[s] = {
            "prompts": sum(1 for p in world.prompts if p.split == s),
            "candidates": sum(1 for c in world.candidates if c.split == s),
            "pairs": sum(1 for p in world.pairs if p.split == s),
            "rated": sum(1 for c in world.candidates if c.split == s and c.id in world.anchor_ratings),
        }
    return out


def label_counts(pairs: Sequence[Pair]) -> dict[str, int]:
    """三个取值的计数 + 噪声条数。**「标签噪声率」必须可核对** ——
    不然「w* 为什么偏离真值」就只能靠猜。"""
    out = {"a": 0, "b": 0, "tie": 0, "noisy": 0}
    for p in pairs:
        if p.label == PREFER_A:
            out["a"] += 1
        elif p.label == PREFER_B:
            out["b"] += 1
        else:
            out["tie"] += 1
        if p.noisy:
            out["noisy"] += 1
    return out
