"""聚合：可靠性加权的分量聚合 + 权重混合 + 平局判定。

两层聚合，职责完全不同：

```
第 1 层（**测出来的**，不搜索）：C[k] = Σ_p ρ[p][k]·c[p][k] / Σ_p ρ[p][k]
第 2 层（**搜索出来的**，本项目的主角）：R = Σ_k w[k]·C[k] / Σ_k w[k]
```

第 1 层刻意**不搜索**：provider 的可信度应该被**测量**（在 anchor 上算残差），
而不是被搜出来。把 ρ 也丢进搜索空间，就等于「搜一组能拟合训练标签的可靠性」——
那是过拟合，不是知识。

## 三个必须写下来的性质

1. **盲区是缺席，不是 0。** `c[p][k] is None` 时该 provider 在分量 k 上**不参与
   归一化分母**。把它当 0 会让「看不见 style 的 rule」在 style 上投一张反对票。
2. **ρ 低于 `min_reliability` 的 provider 直接出局**，不参与。这顺带解决了一件事：
   一个**接反了方向**的 provider 会在 anchor 上暴露（校准残差 ≈ 0.5 ⇒ ρ ≈ 0），
   于是自动出局 —— 不需要任何特殊处理。实测这一点让「方向检测」的价值
   从「救命」降级为「把死重变回贡献者」（见附录 B）。
3. **`provider_contrib` 是可加的**：`Σ_p contrib[p] == R`（浮点意义下）。
   这不是装饰 —— 报告里要能回答「这个候选的 0.71 分里，`rule` 出了多少」。
   分解式：`R = Σ_p Σ_k w[k]·(ρ[p][k]/Σ_p'ρ[p'][k])·c[p][k]`。

## 平局判定用的就是不确定度

`|ΔR| ≤ κ·(u_a + u_b)` ⇒ 判平局。这让 `u` 从装饰性字段变成**功能性组件**：
估小了会强迫在含糊的对上硬选，估大了会把所有对都判平，两个方向都掉一致性。
`κ` 写在配置里（`uncertainty.tie_kappa`），**不是**魔法常数。

## u 的传播必须是平方和，不是算术和

`R = Σ_k w_k·C_k` 的误差是 `sqrt(Σ_k (w_k·u_k)²)`。写成 `Σ_k w_k·u_k` 会把 u 放大
最多 √5 ≈ 2.24 倍（5 个分量）—— 平局带随之吞掉大部分对，判定退化成不作为。
实测这一处口径差把「最好单源」的一致性从 6.25% 一路带到 50%+ 的区间，
是继「全盲分量输出精确常数」之后第二个纯量级 bug。**口径统一比公式精细重要。**

## u 必须随候选变化，否则 ECE 无从谈起

`u_k` 有两项，缺一不可：

- **系统项** `Σ_p w_pk²·mae_pk²`：各 provider 在 anchor 上的交叉验证残差。
  它是 **provider 级**的常数 —— 只看它，`u` 对每个候选都一样。
- **分歧项** `Σ_p w_pk²·(c_pk − C_k)² / (1 − Σ_p w_pk²)`：provider 之间对这个
  候选有多不一致。这是 `u` 里**唯一随候选变化**的部分。

只有系统项时，`u` 是一个常数，于是「u 与 |误差| 的相关性」恒为 0、ECE 退化成
`|mean(u) − mean(|err|)|` —— 不确定度变成了一个装饰性字段。除 `1 − Σ w_p²`
这一步不能省（那是加权样本方差的无偏修正）：不除会把 u 放大 √N 倍。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .calibrate import SCALAR_COMPONENT, CalibrateResult
from .normalize import NormalizeResult
from .schema import COMPONENTS, PREFER_A, PREFER_B, TIE, MixedScore

__all__ = [
    "AggregateResult",
    "aggregate",
    "component_table",
    "mix",
    "normalize_w",
    "decide",
    "uniform_w",
    "vertex_w",
    "SIMPLEX_TOL",
]

#: 单纯形判定的容差。权重是浮点数，`sum(w) == 1.0` 在浮点下必然偶发为假。
SIMPLEX_TOL = 1e-9


def normalize_w(w: Sequence[float]) -> tuple[float, ...]:
    """把权重归一到单纯形。**负权重直接报错** —— 「某个分量越多越差」这种需求
    应该用「把该分量的校准映射翻过来」实现，而不是给权重一个负号：
    负权重会让「R 高 = 好」这个全局约定失效，而所有下游都依赖它。"""
    vals = [float(x) for x in w]
    for x in vals:
        if x < -SIMPLEX_TOL:
            raise ValueError(f"权重不许为负：{w}（要反向请改校准映射，不要改符号）")
    vals = [max(0.0, x) for x in vals]
    s = math.fsum(vals)
    if s <= SIMPLEX_TOL:
        raise ValueError(f"权重之和必须为正：{w}（全 0 的权重等于没有目标函数）")
    return tuple(x / s for x in vals)


def uniform_w(n: int) -> tuple[float, ...]:
    return tuple(1.0 / n for _ in range(n))


def vertex_w(n: int, k: int) -> tuple[float, ...]:
    """单纯形的顶点：只押一个分量。**「单源」在权重空间里的对应物。**"""
    if not 0 <= k < n:
        raise IndexError(f"顶点下标越界：{k}（共 {n} 个分量）")
    return tuple(1.0 if i == k else 0.0 for i in range(n))


def mix(components: Sequence[float], w: Sequence[float]) -> float:
    """`R = Σ w·C / Σ w`。权重先归一，所以 `mix` 对 w 的整体缩放不变。"""
    wn = normalize_w(w)
    return math.fsum(a * b for a, b in zip(wn, components))


def decide(delta_r: float, u_a: float, u_b: float, kappa: float) -> int:
    """按不确定度判平局。**这是 `u` 唯一被真正用到的地方**（其余地方只是报告）。

    判据是**对称**的：`|ΔR| ≤ κ(u_a + u_b)`。两个候选各自的不确定度相加，
    因为「差值的误差」是两个独立误差的和 —— 用 `max` 会低估。
    """
    if abs(delta_r) <= kappa * (u_a + u_b):
        return TIE
    return PREFER_A if delta_r > 0 else PREFER_B


@dataclass(frozen=True)
class AggregateResult:
    """分量表 + 每分量不确定度 + 每 (provider, 分量) 的有效权重。"""

    #: `C[k]` per candidate
    components: Mapping[str, tuple[float, ...]]
    #: `u_k` per candidate（只含分量项，不含权重项）
    u_components: Mapping[str, tuple[float, ...]]
    #: `ρ[p][k]` 与归一化后的有效权重 `ρ[p][k] / Σ_p' ρ[p'][k]`
    reliability: Mapping[str, tuple[float, ...]]
    effective: Mapping[str, tuple[float, ...]]
    #: `c[p][k]` per candidate（校准后），供混合器与基线共用
    calibrated: Mapping[str, Mapping[str, tuple[float | None, ...]]]
    #: 每个分量的「参与 provider 数」
    n_active: tuple[int, ...]

    def cids(self) -> tuple[str, ...]:
        return tuple(sorted(self.components))

    def mix_all(
        self,
        w: Sequence[float],
        kappa: float,
        *,
        u_w: Mapping[str, float] | None = None,
        prompt_of: Mapping[str, str] | None = None,
    ) -> dict[str, MixedScore]:
        wn = normalize_w(w)
        out: dict[str, MixedScore] = {}
        for cid in sorted(self.components):
            comps = self.components[cid]
            total = math.fsum(a * b for a, b in zip(wn, comps))
            # **不确定度按平方和传播，不按算术和。** `R = Σ w_k·C_k` 的误差是
            # `sqrt(Σ (w_k·u_k)²)`。用 `Σ w_k·u_k` 会把 u 放大最多 √5 ≈ 2.24 倍
            # （5 个分量），于是平局带吞掉大部分对，判定退化成不作为 ——
            # 这是实测抓到的第二个量级口径 bug（见 README 附录 C）。
            u_comp = math.sqrt(math.fsum((a * b) ** 2 for a, b in zip(wn, self.u_components[cid])))
            # 判定口径的不确定度**只含分量项**：权重项折进去会成环
            # （算 u_w 要先有 w，求 w 的搜索又要用 u 判平局）。
            contrib = {}
            for pid, eff in self.effective.items():
                vals = self.calibrated[pid][cid]
                s = 0.0
                for k in range(len(COMPONENTS)):
                    v = vals[k]
                    if v is None or eff[k] <= 0.0:
                        continue
                    s += wn[k] * eff[k] * v
                contrib[pid] = s
            out[cid] = MixedScore(
                candidate_id=cid,
                prompt_id=(prompt_of or {}).get(cid, ""),
                total=total,
                components=comps,
                uncertainty=u_comp,
                u_components=self.u_components[cid],
                provider_contrib=contrib,
                u_weight=float((u_w or {}).get(cid, 0.0)),
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": len(self.components),
            "n_active_per_component": {k: n for k, n in zip(COMPONENTS, self.n_active)},
            "reliability": {
                pid: {k: round(v, 4) for k, v in zip(COMPONENTS, rel)}
                for pid, rel in sorted(self.reliability.items())
            },
            "effective_weight": {
                pid: {k: round(v, 4) for k, v in zip(COMPONENTS, eff)}
                for pid, eff in sorted(self.effective.items())
            },
        }


def component_table(
    cal: CalibrateResult,
    nz: NormalizeResult,
    cfg: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, dict[str, tuple[float | None, ...]]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    tuple[int, ...],
]:
    """算出 `C[k]`、校准后分量表、有效权重、逐分量不确定度、活跃 provider 数。

    抽成独立函数是为了让「消融」能复用：消融要换的是**校准方法**或**搜索器**，
    不是这一段聚合逻辑。让它只依赖 `cal` / `nz`，消融就只需要换 `cal`。
    """
    ag = cfg["aggregate"]
    max_sigma = float(ag["max_provider_sigma"])
    floor = float(cfg["uncertainty"]["floor"])

    pids = sorted(cal.scores)
    # ---- 第 1 层的权重：**逆方差**。
    #
    # `w_pk ∝ 1/σ_pk²`，σ = 该 (provider, 分量) 在 anchor 上的交叉验证残差。
    # 这是「对同一个量的多个独立估计」的标准最优组合（逆方差加权 = 最小方差无偏）。
    #
    # 曾经用 `ρ^power`，其中 `ρ = clamp(1 − slope·mae)`。那是错的，而且错得很隐蔽：
    # `slope` 只是一个**单调变换**，但变换会改变权重 —— 线性映射把
    # [0.066, 0.215] 的 mae 压成 [0.87, 0.57] 的 ρ，于是一个 mae 小 2 倍的
    # provider 只拿到 1.5 倍的权重，而不是 4 倍。实测的后果是：`rule` 在
    # correctness 上 mae=0.066（其余 0.083~0.134），却只占 22% 的份额，
    # 被稀释到「组合输给最好单源」。**「谁更可信」不该靠一个拍脑袋的斜率去调。**
    #
    # 门槛：σ > `max_provider_sigma` 的 provider 在该分量上**直接出局**
    # —— 它比「不测」还差。
    raw_eff: dict[str, tuple[float, ...]] = {}
    sigma_of: dict[str, tuple[float, ...]] = {}
    for pid in pids:
        sig: list[float] = []
        eff: list[float] = []
        for k in range(len(COMPONENTS)):
            m = cal.map_of(pid, COMPONENTS[k])
            s = float(m.mae_after)
            sig.append(s)
            eff.append(0.0 if (m.blind or s <= 1e-9 or s > max_sigma) else 1.0 / (s * s))
        sigma_of[pid] = tuple(sig)
        raw_eff[pid] = tuple(eff)

    active = [0] * len(COMPONENTS)
    denom = [0.0] * len(COMPONENTS)
    for pid in pids:
        for k in range(len(COMPONENTS)):
            if raw_eff[pid][k] > 0.0:
                denom[k] += raw_eff[pid][k]
                active[k] += 1
    # **归一化后再存**：`eff[pid][k] = (1/σ²) / Σ_p' (1/σ'²)`。
    # 这一步不能省也不能推迟 —— `mix_all` 的 `provider_contrib` 要能直接
    # `Σ_p contrib[p] == R`（有验收项断言），而那条等式只在权重已归一时成立。
    eff: dict[str, tuple[float, ...]] = {
        pid: tuple(
            (raw_eff[pid][k] / denom[k]) if denom[k] > 0.0 else 0.0
            for k in range(len(COMPONENTS))
        )
        for pid in pids
    }

    calibrated: dict[str, dict[str, tuple[float | None, ...]]] = {
        pid: {cid: cal.scores[pid][cid].components for cid in sorted(cal.scores[pid])}
        for pid in pids
    }
    cids = sorted(calibrated[pids[0]]) if pids else []

    components: dict[str, tuple[float, ...]] = {}
    u_components: dict[str, tuple[float, ...]] = {}
    for cid in cids:
        cvals: list[float] = []
        uvals: list[float] = []
        for k in range(len(COMPONENTS)):
            if denom[k] <= 0.0:
                # 该分量**没有任何可用 provider**：给 0.5（中性）与一个很大的不确定度。
                # 给 0 会让「没数据」变成「很差」，给 0.5 且不确定度大，
                # 混合器会把它当作「不知道」，而平局判定会据此避免下结论。
                cvals.append(0.5)
                uvals.append(0.5)
                continue
            num = 0.0
            sys_var = 0.0
            share_sq = 0.0
            for pid in pids:
                wp = eff[pid][k]
                if wp <= 0.0:
                    continue
                v = calibrated[pid][cid][k]
                if v is None:
                    continue
                num += wp * v
                # 系统项：逐 provider 的残差尺度（CV MAE）当作它的标准差。
                # 逆方差权重下 `Σ_p w_p²·σ_p² == 1 / Σ_p (1/σ_p²)` —— 就是教科书里
                # 「多个独立估计的加权均值」的后验方差，不用另立公式。
                sys_var += (wp * sigma_of[pid][k]) ** 2
                share_sq += wp * wp
            # `eff` 已归一（Σ_p eff[p][k] == 1），所以 `num` 就是 `C_k`。
            ck = num

            # 候选级分歧项：**provider 之间对这个候选有多不一致。**
            # `Σ_p w_p²(v_p − C_k)² / (1 − Σ_p w_p²)` 是「加权均值的方差」的
            # 无偏估计（标准加权样本方差公式）。它是 u 里**唯一随候选变化**的部分
            # —— 没有它，u 就是一个常数，ECE 与「u 和误差的相关性」都无从谈起。
            # 除 `1 − Σ w_p²` 这一步不能省：不除会把 u 放大 √N 倍（实测 5 个
            # provider 时是 2.24 倍），平局带随之吞掉大半的对。
            disp = 0.0
            for pid in pids:
                wp = eff[pid][k]
                if wp <= 0.0:
                    continue
                v = calibrated[pid][cid][k]
                if v is None:
                    continue
                disp += wp * wp * (v - ck) ** 2
            rest = 1.0 - share_sq
            # 只有一个 provider 可用时 `rest == 0`：分歧无从测量 ⇒ 只留系统项。
            disp_var = (disp / rest) if rest > 1e-6 else 0.0

            cvals.append(ck)
            uvals.append(max(floor, math.sqrt(sys_var + disp_var)))
        components[cid] = tuple(cvals)
        u_components[cid] = tuple(uvals)

    return components, calibrated, eff, u_components, tuple(active)


def aggregate(cal: CalibrateResult, nz: NormalizeResult, cfg: Mapping[str, Any]) -> AggregateResult:
    components, calibrated, eff, u_components, active = component_table(cal, nz, cfg)
    return AggregateResult(
        components=components,
        u_components=u_components,
        reliability={pid: cal.reliability[pid] for pid in cal.reliability},
        effective=eff,
        calibrated=calibrated,
        n_active=active,
    )
