"""provider 池：把「一组不可比的打分器」造出来。

这是**模拟物理**，不是检测器。所以它允许读 `invert`（`providers.py` 是唯一
允许读它的模块）—— 反转的实现方式就是「输出 1−s」，方向信息只以**数值形式**
泄漏出去，检测器只能从 anchor 上把它测回来。

五个 provider，每个带一种别处没有的缺陷，合起来覆盖四种病：

| provider | 病 |
| --- | --- |
| `rule` | **窄**：判据内精确（σ=0），但只覆盖一小片 —— 在 correctness（权重 0.34）上全盲 |
| `rm_small` | 只看得见 correctness 一侧；**尺度是无界 logit**，方差大一个量级 |
| `rm_large` | 覆盖最全，但把「长」读成「好」（非仿射偏差，**校准修不掉**） |
| `llm_judge` | 噪声最大 + 家族偏好（写作类一律抬高） |
| `rm_legacy` | **接线接反了**，却自称分数越高越好 |

「每个 provider 都缺一块、且缺的不是同一块」是这套模拟的**全部意义**：
只要有一个 provider 在多数分量上都接近完美，它单独就够用了，
「Reward 不应该只有一个模型」这句话在数据上就不成立。第一版把 `rule` 写成
`correctness 0.97 / safety 1.00 / task_reward 0.90`，实测就是那个结果
（最好单源 88% 压过 5 源组合 84%）—— 见 README 附录 B。

三条设计要点：

1. **`vision[k] == 0` 的分量输出「head 的偏置 + 一点噪声」**，不是精确常数。
   这不是偷懒，是一个真实 head 的样子：内部表示里没有该分量的信息，输出就是
   它的偏置项，只随输入抖动。**为什么必须加那点噪声**（`blind_noise`）：若输出
   精确常数，`isotonic` 校准会把这条常数拟合到 anchor 上该分量的平均人类评分，
   造出一条「常数完美预言」，让全盲 provider 在该分量上拿到 ρ = 1.00 冒充神谕
   —— 实测中它把真值权重上界压到了 53%。加了噪声，这条映射退化成
   「低方差但无信息」，盲区才真的是盲区。
2. **偏差只加在非仿射的地方**。常数抬高、线性缩放都属于仿射，校准能修掉，
   写进配置只会污染「偏差」这个词。所以这里只有**长度**与**家族**两种偏差。
3. **`self_uncertainty` 是刻意乐观的**（×0.5）。provider 自报的不确定度不可信
   —— 它不知道自己的视力被削弱了多少、也不知道自己有长度偏差。产物里报的
   不确定度一律是 **anchor 上实测的残差**，不是这个自报值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rng import Rng, derive_seed, digest_of
from .schema import (
    COMPONENTS,
    SCALE_FAMILIES,
    ProviderOutput,
    ProviderSpec,
)
from .world import World

__all__ = [
    "ProviderPool",
    "build_pool",
    "to_scale",
    "flip_total",
    "LENGTH_BIAS_TARGETS",
    "FAMILY_BIAS_TARGETS",
    "SELF_UNCERTAINTY_DISCOUNT",
]

#: 长度偏差加在哪几个分量上。「更长 = 更细致」是**近似成立**的启发式，
#: 所以它污染的是「细致程度」相关的判断，而不是 correctness。
LENGTH_BIAS_TARGETS: tuple[str, ...] = ("helpfulness", "style")

#: 家族偏好加在哪几个分量上。一个偏爱写作类题面的裁判，会同时觉得它更
#: 有文采**和**更有帮助 —— 只污染一个分量的话太容易被别的 provider 投掉。
FAMILY_BIAS_TARGETS: tuple[str, ...] = ("helpfulness", "style")

#: provider 自报不确定度的乐观系数。实测下来的差额就是「自报不可用」的证据。
SELF_UNCERTAINTY_DISCOUNT = 0.5

#: 每个尺度族的原始分范围。用于把「±1 方向」翻译成数值翻转。
_ORDINAL_LO, _ORDINAL_HI = 1.0, 10.0
_SATURATION_GAIN = 3.2
_LOGIT_GAIN = 2.6


def to_scale(scale: str, m: float) -> float:
    """把 [0,1] 的内部判断映射到 provider 自己的尺度。

    **这一步就是「不可比」的来源**：`logit` 的方差比 `grid3` 大两个数量级，
    等权平均时前者单方面决定结果 —— 这不是理论担忧，是本项目实测的主结论之一。
    """
    if scale == "logit":
        return _LOGIT_GAIN * (m - 0.5)
    if scale == "unit":
        return m
    if scale == "saturated":
        return 1.0 / (1.0 + math.exp(-_SATURATION_GAIN * (m - 0.5)))
    if scale == "ordinal10":
        return round(_ORDINAL_LO + (_ORDINAL_HI - _ORDINAL_LO) * m)
    if scale == "grid3":
        return round(2.0 * m) / 2.0
    raise ValueError(f"未声明的尺度族：{scale!r}（可选：{SCALE_FAMILIES}）")


def flip_total(scale: str, x: float) -> float:
    """把标量按尺度族翻转。**每个尺度族的翻转不一样**，用一个通用公式会错：
    `logit` 对称于 0，`ordinal10` 对称于 5.5，`unit`/`saturated` 对称于 0.5。"""
    if scale == "logit":
        return -x
    if scale in ("unit", "saturated"):
        return 1.0 - x
    if scale == "ordinal10":
        return (_ORDINAL_LO + _ORDINAL_HI) - x
    if scale == "grid3":
        return 1.0 - x
    raise ValueError(f"未声明的尺度族：{scale!r}")


@dataclass(frozen=True)
class ProviderPool:
    """五个 provider 在全部候选上的原始输出。

    `outputs[pid][cid]` 是 `ProviderOutput`。**只存原始输出** —— 归一化、校准、
    聚合都在后面的模块里做，不在这里偷偷做掉一半。
    """

    specs: tuple[ProviderSpec, ...]
    outputs: Mapping[str, Mapping[str, ProviderOutput]]
    length_z: Mapping[str, float]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.specs)

    def spec(self, pid: str) -> ProviderSpec:
        for s in self.specs:
            if s.id == pid:
                return s
        raise KeyError(f"没有这个 provider：{pid}")

    def output(self, pid: str, cid: str) -> ProviderOutput:
        return self.outputs[pid][cid]

    def raw_components(self, pid: str) -> tuple[tuple[float | None, ...], ...]:
        """该 provider 在全部候选上的原始分量（顺序按 `sorted(cid)`）。

        顺序必须钉死：Spearman 相关、方差比、方向检测都依赖「同一个位置是同一个
        候选」，靠 dict 迭代顺序会静默错位。
        """
        m = self.outputs[pid]
        return tuple(m[cid].components for cid in sorted(m))

    def cids(self) -> tuple[str, ...]:
        return tuple(sorted(self.outputs[self.specs[0].id]))

    def scale_variance(self, pid: str) -> float:
        """该 provider **原始标量**的方差。尺度异质性的直接度量。"""
        m = self.outputs[pid]
        xs = [m[cid].raw_total for cid in sorted(m)]
        if len(xs) < 2:
            return 0.0
        mu = math.fsum(xs) / len(xs)
        return math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1)

    def fingerprint(self) -> str:
        return digest_of(
            {
                "providers": [s.id for s in self.specs],
                "scales": [s.scale for s in self.specs],
                "raw": {
                    s.id: [round(self.outputs[s.id][c].raw_total, 6) for c in self.cids()]
                    for s in self.specs
                },
            }
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"fingerprint": self.fingerprint(), "providers": []}
        for s in self.specs:
            m = self.outputs[s.id]
            raw = [m[c].raw_total for c in self.cids()]
            mu = math.fsum(raw) / len(raw) if raw else 0.0
            var = self.scale_variance(s.id)
            out["providers"].append(
                {
                    "id": s.id,
                    "kind": s.kind,
                    "label": s.label,
                    "scale": s.scale,
                    "declared_direction": s.declared_direction,
                    "sigma": round(s.sigma, 6),
                    "blind_noise": round(s.blind_noise, 6),
                    "cost_ms": round(s.cost_ms, 4),
                    "vision": {k: round(v, 4) for k, v in zip(COMPONENTS, s.vision)},
                    "blind_declared": [k for k, v in zip(COMPONENTS, s.vision) if v == 0.0],
                    "length_bias": round(s.length_bias, 6),
                    "family_bias": dict(s.family_bias),
                    "raw_mean": round(mu, 6),
                    "raw_variance": round(var, 6),
                    "raw_std": round(math.sqrt(var), 6),
                    "raw_range": [round(min(raw), 6), round(max(raw), 6)] if raw else [0.0, 0.0],
                    "n_outputs": len(raw),
                    "note": s.note,
                }
            )
        return out


# ------------------------------------------------------------------ 生成

def _spec_from(raw: Mapping[str, Any]) -> ProviderSpec:
    vis = raw["vision"]
    return ProviderSpec(
        id=str(raw["id"]),
        kind=str(raw["kind"]),
        label=str(raw.get("label", raw["id"])),
        vision=tuple(float(vis[k]) for k in COMPONENTS),
        sigma=float(raw["sigma"]),
        scale=str(raw["scale"]),
        declared_direction=int(raw["declared_direction"]),
        blind_noise=float(raw.get("blind_noise", 0.0)),
        length_bias=float(raw.get("length_bias", 0.0)),
        family_bias={str(k): float(v) for k, v in dict(raw.get("family_bias", {})).items()},
        cost_ms=float(raw.get("cost_ms", 0.0)),
        note=str(raw.get("note", "")),
    )


def build_pool(world: World, cfg: Mapping[str, Any]) -> ProviderPool:
    """生成 provider 池。

    **噪声挂在 `(provider, 候选)` 上**（`root.fork("prov", pid, cid)`），不挂在
    「这一次打分」上：同一对 (provider, 候选) 反复打分必须得到同一个数，
    否则「打两次不一致」的探针会把噪声误报成不稳定。这与
    `PreferenceFactory` 里「噪声必须挂在对象上」是同一条纪律。
    """
    root = Rng(derive_seed(cfg["seed"], "providers"))
    specs = tuple(_spec_from(p) for p in cfg["providers"])
    invert_of = {str(p["id"]): bool(p.get("invert", False)) for p in cfg["providers"]}

    # 长度标准化：只在 **anchor + train** 上拟合。holdout 的长度统计不许进来看一眼 ——
    # 「没被任何环节看过」这条对 holdout 是硬约束，连一个均值和方差都不许借。
    ref = [c for c in world.candidates if c.split in ("anchor", "train")]
    lens = [float(c.n_tokens) for c in ref]
    mu = math.fsum(lens) / len(lens)
    var = math.fsum((x - mu) ** 2 for x in lens) / (len(lens) - 1)
    sd = math.sqrt(var) if var > 0 else 1.0
    length_z = {c.id: (float(c.n_tokens) - mu) / sd for c in world.candidates}

    outputs: dict[str, dict[str, ProviderOutput]] = {}
    for spec in specs:
        per: dict[str, ProviderOutput] = {}
        inv = invert_of[spec.id]
        for cand in world.candidates:
            rng = root.fork("prov", spec.id, cand.id)
            z_len = length_z[cand.id]
            comps: list[float | None] = []
            # **参与标量合成的分量**与**对外导出的分量**是两份不同的东西。
            # 全盲分量要出现在 `comps` 里（带一点 `blind_noise`），否则
            # 「盲区的标准差是不是 0」这个判据就永远成立、盲区检测退化成一
            # 行 `is None`；但它**不能进标量** —— 把三个 ~N(0.5, 0.1) 的噪声
            # 平均进一个只看得见两三个分量的 provider，等于用噪声把它的
            # 输出拽向 0.5，再被 `grid3` 量化压成**精确常数**（实测踩过：
            # `rule` 的 raw_total 恒为 0.5，`scale_variance_ratio` 变成 max/0）。
            seen: list[float] = []
            for k, vis in zip(COMPONENTS, spec.vision):
                if vis <= 0.0:
                    comps.append(0.5 + rng.normal(0.0, spec.blind_noise))
                    continue
                v = 0.5 + vis * (cand.comp(k) - 0.5) + rng.normal(0.0, spec.sigma)
                if k in LENGTH_BIAS_TARGETS:
                    v += spec.length_bias * z_len
                if k in FAMILY_BIAS_TARGETS:
                    v += spec.family_bias.get(cand.family, 0.0)
                comps.append(v)
                seen.append(v)

            m = math.fsum(seen) / len(seen) if seen else 0.5
            raw_total = to_scale(spec.scale, m)
            if inv:
                # 反转：分量按 [0,1] 对称翻（接口契约里分量就是 [0,1]），
                # 标量按**该尺度族自己的**对称点翻。
                comps = [None if v is None else (1.0 - v) for v in comps]
                raw_total = flip_total(spec.scale, raw_total)

            per[cand.id] = ProviderOutput(
                provider_id=spec.id,
                candidate_id=cand.id,
                components=tuple(comps),
                raw_total=float(raw_total),
                self_uncertainty=max(0.0, spec.sigma * SELF_UNCERTAINTY_DISCOUNT),
            )
        outputs[spec.id] = per

    return ProviderPool(specs=specs, outputs=outputs, length_z=length_z)
