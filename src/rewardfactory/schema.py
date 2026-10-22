"""对象链与常量表。**这个文件是「谁在哪一层」的唯一裁决处。**

对象链（从真值到产物，单向）：

```
Prompt         一个题面（家族 + 难度）
  └ Candidate  一个候选回答（5 个 latent 分量 + 长度 + 不可观测残差 → quality）
      └ ProviderOutput  某个 provider 的**原始**输出（自带尺度、自带方向、可能带偏差）
          └ ProviderScore  归一化后（方向已对齐、盲区已标出、尺度族已记录）
              └ CalibratedScore  校准到「与真值同尺度」的分量分 + 可靠性 ρ
                  └ MixedScore  可靠性加权 → 分量 C[k] → 权重混合 → R + u
                      └ RewardEvidence  对外统一接口的返回体（分量 + 不确定度 + 证据）
Pair           同一题面下两个候选的成对标注（人类偏好，带噪声）
```

三条命名纪律，写在这里免得每个模块各自发明一遍：

1. **`raw` / `norm` / `cal` 是三级前缀**，永远指上面那三层。`raw_total` 是 provider
   自己尺度上的标量，`cal_total` 是校准到 [0,1] 的标量。**两者不同量纲，不可相减。**
2. **`blind` 是状态，不是错误**：provider 对某分量没有视力时该分量是 `None`，
   不是 0。0 是「我认为它很差」，`None` 是「我看不见」—— 混起来会让盲区变成打分。
3. **`direction` 用 ±1 而不是字符串**：`-1` 表示「原始分越高越差」（历史遗留的反向
   RM 真的存在）。方向是**检测出来的**，不是配置里声明的 —— 见 `normalize.py`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "COMPONENTS",
    "N_COMPONENTS",
    "PROVIDER_KINDS",
    "SCALE_FAMILIES",
    "CALIBRATION_METHODS",
    "CALIBRATION_NOTES",
    "SEARCHERS",
    "SPLITS",
    "FAMILIES",
    "BASELINES",
    "BASELINE_LABELS",
    "TIE",
    "PREFER_A",
    "PREFER_B",
    "label_name",
    "Prompt",
    "Candidate",
    "ProviderSpec",
    "ProviderOutput",
    "ProviderScore",
    "CalibratedScore",
    "CalibrationMap",
    "MixedScore",
    "RewardEvidence",
    "Pair",
    "PairVerdict",
    "SearchTrial",
    "SearchResult",
    "WeightEstimate",
    "BootstrapResult",
    "ProviderReport",
    "MethodScore",
    "ErrorDecomposition",
    "BenchmarkResult",
    "ExpectationRow",
    "AcceptanceCheck",
    "AdversarialCase",
    "VerifyResult",
    "to_dicts",
]

# --------------------------------------------------------------------- 常量表

#: 五个分量。**顺序即约定**：所有权重向量、分量矩阵都按这个顺序排列，
#: 换顺序等于换项目（指纹会变，但不会报错 —— 所以只在配置里出现一次）。
COMPONENTS: tuple[str, ...] = ("helpfulness", "correctness", "safety", "style", "task_reward")
N_COMPONENTS = len(COMPONENTS)

#: provider 的种类。README 的架构图是「RM Model / LLM Judge / Rule」三类。
PROVIDER_KINDS: tuple[str, ...] = ("rm_model", "llm_judge", "rule")

#: provider 原始输出的**尺度族**。这是「不校准就平均」会失败的根本原因：
#: `logit` 的方差比 `grid3` 大两个数量级，等权平均时前者单方面决定结果。
SCALE_FAMILIES: tuple[str, ...] = ("logit", "unit", "saturated", "ordinal10", "grid3")

#: 五种校准法。`identity` 是**对照**（不校准），不是「另一种选择」。
#:
#: `quantile` 是**经验分位数映射**（直方图匹配）：`g(x) = Q_y(F_x(x))`。
#: 它刻意不叫 `beta` —— 分位数映射不需要拟合任何参数分布，写成
#: 「拟合 Beta 再求逆」只会平白引入一个不完整 Beta 函数的数值实现，
#: 而那个实现本身才是误差来源。
CALIBRATION_METHODS: tuple[str, ...] = ("identity", "minmax", "zscore", "isotonic", "quantile")

#: 五种方法各自「信的是什么」。报告里要写明，否则读者会以为它们在比同一件事。
CALIBRATION_NOTES: dict[str, str] = {
    "identity": "不校准（对照）：只做 clip，什么也不修",
    "minmax": "线性拉伸到 [0,1]：只修仿射，不修形状",
    "zscore": "正态 CDF：假定 provider 的误差是正态的",
    "isotonic": "保序回归（PAVA）：只假定单调，不假定形状",
    "quantile": "经验分位数映射：匹配**分布**而不是匹配条件均值",
}

#: 四个搜索器。`grid` 是穷举（上界参考），`random` 是对照组，`forward` 产出
#: 「单源 → 组合」的路径。
SEARCHERS: tuple[str, ...] = ("grid", "random", "coordinate", "forward")

#: 三个数据划分。**anchor 与 holdout 必须互斥** —— anchor 是被标注过的样本，
#: holdout 是「从没被任何环节看过」的样本。混在一起，验收就是自评。
SPLITS: tuple[str, ...] = ("anchor", "train", "holdout")

#: 题面家族。家族偏差（某个 provider 偏爱某一类）要靠它做分层诊断。
FAMILIES: tuple[str, ...] = ("arith", "logic", "writing", "code", "safety")

#: 全部参与对比的「方法」。顺序即报告里的行序。
BASELINES: tuple[str, ...] = (
    "chance",
    "naive_raw_mean",
    "best_single_raw",
    "calibrated_scalar_mean",
    "best_single_calibrated",
    "component_equal_w",
    "component_argmax_w",
    "component_searched_w",
    "oracle_truth_w",
)

BASELINE_LABELS: dict[str, str] = {
    "chance": "多数类基线（永远猜同一侧）",
    "naive_raw_mean": "不校准、等权平均各 provider 的原始标量",
    "best_single_raw": "最好的单个 provider（原始标量）",
    "calibrated_scalar_mean": "先校准标量、再等权平均",
    "best_single_calibrated": "最好的单个 provider（校准后）",
    "component_equal_w": "分量分解 + 校准，权重等分",
    "component_argmax_w": "分量分解 + 校准 + 搜索权重（**搜索器的原始答案 argmax**）",
    "component_searched_w": "分量分解 + 校准 + 搜索权重（**bootstrap 均值收缩，本项目交付**）",
    "oracle_truth_w": "注入的真值权重（上界参考，现实中拿不到）",
}

#: 成对标注的三个取值。**`TIE` 是一等判决，不是兜底** —— 强行二选一会把
#: 人类在「两个都差不多」时的犹豫固化成一个假的胜负。
PREFER_A = 1
PREFER_B = -1
TIE = 0


def label_name(label: int) -> str:
    return {PREFER_A: "A", PREFER_B: "B", TIE: "tie"}[int(label)]


# ------------------------------------------------------------------ 对象链

@dataclass(frozen=True)
class Prompt:
    """题面。`difficulty` 越高，候选越容易一起变差（成对排序更难）。"""

    id: str
    family: str
    difficulty: float
    split: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "difficulty": round(self.difficulty, 6),
            "split": self.split,
            "text": self.text,
        }


@dataclass(frozen=True)
class Candidate:
    """候选回答。

    `components` 是**真值分量**（latent），`quality` 是 `Σ w_true·comp + 不可观测残差`。
    `unobservable` 单独存着 —— 它是「人类偏好里谁也读不到的那一部分」，
    没有它，一个完美 provider 就能拿满分，整个验收会退化成同义反复。
    """

    id: str
    prompt_id: str
    family: str
    split: str
    text: str
    n_tokens: int
    components: tuple[float, ...]
    quality: float
    unobservable: float
    verbosity: float

    def comp(self, name: str) -> float:
        return self.components[COMPONENTS.index(name)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "family": self.family,
            "split": self.split,
            "n_tokens": self.n_tokens,
            "components": {k: round(v, 6) for k, v in zip(COMPONENTS, self.components)},
            "quality": round(self.quality, 6),
            "unobservable": round(self.unobservable, 6),
        }


@dataclass(frozen=True)
class ProviderSpec:
    """provider 的**声明**（配置里的那一份）。运行时实测的属性在 `ProviderReport` 里。

    `vision[k]` 是「这个 provider 对分量 k 的视力」∈ [0,1]：`vision=0` 就是全盲
    （输出只剩 head 的偏置 + 一点噪声），`vision=1` 表示它看得见真值分量
    （仍可能叠加噪声）。
    `bias` 是**非仿射**的偏差项 —— 仿射项（常数抬高、线性缩放）校准能修，
    所以只有非仿射的才值得写进配置：长度偏差、家族偏好。

    `blind_noise` 是**全盲分量**上叠加的噪声标准差。它存在的理由是一个真实
    的坑：若全盲分量输出**精确常数**（如 0.5），校准器会把这条常数映射到
    anchor 上该分量的**平均人类评分**（如 0.85）—— 一条「常数完美预言」，
    于是这个本该被无视的 provider 在该分量上拿到 ρ = 1.00，冒充神谕。
    给盲区一点噪声，这条映射就退化成「低方差但无信息」的噪声，ρ ≈ 0。
    """
    id: str
    kind: str
    label: str
    vision: tuple[float, ...]
    sigma: float
    scale: str
    declared_direction: int
    blind_noise: float = 0.0
    length_bias: float = 0.0
    family_bias: Mapping[str, float] = field(default_factory=dict)
    cost_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "vision": {k: round(v, 4) for k, v in zip(COMPONENTS, self.vision)},
            "sigma": round(self.sigma, 6),
            "blind_noise": round(self.blind_noise, 6),
            "scale": self.scale,
            "declared_direction": self.declared_direction,
            "length_bias": round(self.length_bias, 6),
            "family_bias": dict(self.family_bias),
            "cost_ms": round(self.cost_ms, 4),
            "note": self.note,
        }


@dataclass(frozen=True)
class ProviderOutput:
    """provider 的**原始**输出 —— 统一接口 `score(prompt, response)` 的返回体。

    `components[k]` 为 `None` 表示「看不见」（盲区），**不是 0**。
    `raw_total` 是 provider 自己尺度上的标量（logit / 0-1 / 1-10 / 三档），
    它只被**基线**使用；本项目的组合走分量，不走这个标量。
    """

    provider_id: str
    candidate_id: str
    components: tuple[float | None, ...]
    raw_total: float
    self_uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "candidate_id": self.candidate_id,
            "components": {
                k: (None if v is None else round(v, 6))
                for k, v in zip(COMPONENTS, self.components)
            },
            "raw_total": round(self.raw_total, 6),
            "self_uncertainty": round(self.self_uncertainty, 6),
        }


@dataclass(frozen=True)
class ProviderScore:
    """归一化之后：方向已对齐、盲区已标出、尺度族已记录。

    **归一化不改数值，只改符号与标记** —— 把 logit 拉成 0-1 是 `Calibrate` 的活。
    这里做的是「把它变成一个可比的方向一致量」，而不是「把它变成真值尺度」。
    """

    provider_id: str
    candidate_id: str
    components: tuple[float | None, ...]
    raw_total: float
    scale: str
    direction: int
    self_uncertainty: float

    @property
    def flipped(self) -> bool:
        return self.direction < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "candidate_id": self.candidate_id,
            "components": {
                k: (None if v is None else round(v, 6))
                for k, v in zip(COMPONENTS, self.components)
            },
            "raw_total": round(self.raw_total, 6),
            "scale": self.scale,
            "direction": self.direction,
            "self_uncertainty": round(self.self_uncertainty, 6),
        }


@dataclass(frozen=True)
class CalibratedScore:
    """校准到「与真值同尺度」的分量分。

    `reliability` 是该 (provider, 分量) 在 anchor 上实测的可靠性 ∈ [0,1]：
    `1 − 2·MAE`（MAE 是分量尺度的绝对误差，乘 2 是因为随机猜的 MAE ≈ 0.25
    ⇒ `1 − 2·0.25 = 0.5`，正好把「完全没用」映射到 0.5 附近再由下界截住）。
    **可靠性是测出来的，不是配置里声明的** —— 声明一个 provider 有多可靠，
    正是这个项目要消灭的那种做法。
    """

    provider_id: str
    candidate_id: str
    components: tuple[float | None, ...]
    cal_total: float
    reliability: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "candidate_id": self.candidate_id,
            "components": {
                k: (None if v is None else round(v, 6))
                for k, v in zip(COMPONENTS, self.components)
            },
            "cal_total": round(self.cal_total, 6),
            "reliability": {k: round(v, 4) for k, v in zip(COMPONENTS, self.reliability)},
        }


@dataclass(frozen=True)
class CalibrationMap:
    """一个 (provider, 分量) 上的校准映射。

    `points` 是分段线性插值的断点 `[(x, y), …]`，`x` 单调递增。
    **五种校准法最终都落成同一张分段线性表** —— 这是刻意的：产物
    `calibration.json` 必须自足，拿到它的人不该需要再跑一遍 PAVA 才能应用它。

    `mae_after` 是**交叉验证**出来的误差（4 折，按 x 分层），`mae_fit` 是
    拟合集上的误差。两个都要报：只报 `mae_fit` 会让 `isotonic` 看起来无敌
    （它能把 anchor 拟合到几乎零残差），而那个数在 holdout 上不成立。
    """

    provider_id: str
    component: str
    method: str
    points: tuple[tuple[float, float], ...]
    n_anchor: int
    mae_before: float
    mae_after: float
    reliability: float
    mae_fit: float = 0.0
    n_folds: int = 1
    blind: bool = False

    def apply(self, x: float) -> float:
        """分段线性插值。端点外做**常数外推**（不外插 —— 外插会让极端值爆炸，
        而极端值恰恰是最需要保守的地方）。"""
        pts = self.points
        if not pts:
            return 0.5
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid][0] <= x:
                lo = mid
            else:
                hi = mid
        x0, y0 = pts[lo]
        x1, y1 = pts[hi]
        if x1 <= x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "component": self.component,
            "method": self.method,
            "n_anchor": self.n_anchor,
            "n_points": len(self.points),
            "points": [[round(x, 6), round(y, 6)] for x, y in self.points],
            "mae_before": round(self.mae_before, 6),
            "mae_after": round(self.mae_after, 6),
            "mae_fit": round(self.mae_fit, 6),
            "n_folds": self.n_folds,
            "reliability": round(self.reliability, 6),
            "blind": self.blind,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationMap":
        return cls(
            provider_id=str(raw["provider_id"]),
            component=str(raw["component"]),
            method=str(raw["method"]),
            points=tuple((float(x), float(y)) for x, y in raw["points"]),
            n_anchor=int(raw["n_anchor"]),
            mae_before=float(raw["mae_before"]),
            mae_after=float(raw["mae_after"]),
            reliability=float(raw["reliability"]),
            mae_fit=float(raw.get("mae_fit", 0.0)),
            n_folds=int(raw.get("n_folds", 1)),
            blind=bool(raw.get("blind", False)),
        )


@dataclass(frozen=True)
class MixedScore:
    """组合后的分数。**产物里的主角。**

    `uncertainty` 是**判定用的**那一项（纯分量项），它**参与判定**：
    两个候选的分数差落在 `κ·(u_a + u_b)` 以内就判平局。所以它不是一个装饰性的
    字段 —— 估小了会强迫在含糊的对上硬选，估大了会把所有对都判平，两个方向都掉一致性。

    `u_weight` 是权重不稳导致的那部分（bootstrap 下同一个候选分数的散布），
    **只报不判**。理由见 `uncertainty.py`：算它需要 `w`，而求 `w` 的搜索又需要
    用不确定度判平局 —— 折进去就成环了。诚实的做法是把它报出来，而不是假装
    它不存在，也不是用一个自我指涉的口径把它硬塞进判定。
    """

    candidate_id: str
    prompt_id: str
    total: float
    components: tuple[float, ...]
    uncertainty: float
    u_components: tuple[float, ...]
    provider_contrib: Mapping[str, float]
    u_weight: float = 0.0

    @property
    def u_total(self) -> float:
        """报告用的合成不确定度（判定不用它）。"""
        return math.sqrt(self.uncertainty * self.uncertainty + self.u_weight * self.u_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prompt_id": self.prompt_id,
            "total": round(self.total, 6),
            "components": {k: round(v, 6) for k, v in zip(COMPONENTS, self.components)},
            "uncertainty": round(self.uncertainty, 6),
            "u_weight": round(self.u_weight, 6),
            "u_total": round(self.u_total, 6),
            "u_components": {k: round(v, 6) for k, v in zip(COMPONENTS, self.u_components)},
            "provider_contrib": {k: round(v, 6) for k, v in self.provider_contrib.items()},
        }


@dataclass(frozen=True)
class RewardEvidence:
    """**对外统一接口的返回体** —— README 第四节那个 `RewardEvidence`。

    这是本项目的 M0 兑现点：任何来源（RM 模型 / LLM Judge / 规则）都返回它，
    而且它带**分量**与**不确定度**，不是一个裸标量。
    """

    candidate_id: str
    prompt_id: str
    total_score: float
    components: Mapping[str, float]
    uncertainty: float
    providers: Mapping[str, float]
    evidence: tuple[str, ...]
    stub: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prompt_id": self.prompt_id,
            "total_score": round(self.total_score, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "uncertainty": round(self.uncertainty, 6),
            "providers": {k: round(v, 6) for k, v in self.providers.items()},
            "evidence": list(self.evidence),
            "stub": self.stub,
        }


@dataclass(frozen=True)
class Pair:
    """成对标注。`label` 是人类给的（带噪声），`delta_quality` 是**真值**差（只在评测里用）。

    `noisy` 标记这一条被噪声翻过 —— 它让「标签噪声率」可核对，
    也让「哪些错是标签的错、哪些是 reward 的错」分得开。
    """

    id: str
    prompt_id: str
    family: str
    split: str
    a: str
    b: str
    label: int
    delta_quality: float
    noisy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "family": self.family,
            "split": self.split,
            "a": self.a,
            "b": self.b,
            "label": self.label,
            "delta_quality": round(self.delta_quality, 6),
            "noisy": self.noisy,
        }


@dataclass(frozen=True)
class PairVerdict:
    """某一对上的判定结果。`pred` 是混合器给的（`TIE` 表示「差距在不确定度以内」）。"""

    pair_id: str
    split: str
    family: str
    label: int
    pred: int
    delta_r: float
    u_pair: float
    correct: bool
    decisive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "split": self.split,
            "family": self.family,
            "label": self.label,
            "pred": self.pred,
            "delta_r": round(self.delta_r, 6),
            "u_pair": round(self.u_pair, 6),
            "correct": self.correct,
            "decisive": self.decisive,
        }


@dataclass(frozen=True)
class SearchTrial:
    """搜索器的一次评估。`w` 存成 tuple，`agreement` 是目标值。"""

    searcher: str
    step: int
    w: tuple[float, ...]
    agreement: float
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "step": self.step,
            "w": [round(x, 6) for x in self.w],
            "agreement": round(self.agreement, 6),
            "cached": self.cached,
        }


@dataclass(frozen=True)
class SearchResult:
    """一个搜索器的结果。`trace` 是**完整**轨迹，不是只有最优的那一步 ——
    `forward` 的轨迹就是「单源 → 组合」那条路径，是本项目最主要的图。"""

    searcher: str
    best_w: tuple[float, ...]
    best_agreement: float
    n_evals_new: int
    n_evals_cached: int
    trace: tuple[SearchTrial, ...]

    @property
    def n_evals(self) -> int:
        return self.n_evals_new + self.n_evals_cached

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(i for i, x in enumerate(self.best_w) if x > 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "best_w": {k: round(v, 6) for k, v in zip(COMPONENTS, self.best_w)},
            "best_agreement": round(self.best_agreement, 6),
            "n_evals_new": self.n_evals_new,
            "n_evals_cached": self.n_evals_cached,
            "n_trials": len(self.trace),
            "support": [COMPONENTS[i] for i in self.support],
        }


@dataclass(frozen=True)
class WeightEstimate:
    """某个分量权重的 bootstrap 分布摘要。"""

    component: str
    mean: float
    std: float
    lo: float
    hi: float
    ci_level: float
    truth: float

    @property
    def covers_truth(self) -> bool:
        return self.lo <= self.truth <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "ci_level": round(self.ci_level, 4),
            "width": round(self.width, 6),
            "truth": round(self.truth, 6),
            "covers_truth": self.covers_truth,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """权重的不确定度 —— 验收标准 ② 的兑现物。

    `estimates` 逐分量给 CI，`n_covered` 是「CI 覆盖了注入真值」的分量个数。
    **点估计偏离真值是正常的**（人类标签有噪声、样本有限），
    所以判据是**覆盖率**而不是「点估计等不等于真值」——
    后者会逼着实现去偷看真值。

    `samples` 是 200 条重采样出来的权重向量，**只在内存里活着，不进产物**
    （`to_dict` 刻意不导出它）：它服务的是「权重项不确定度」`u_w`，
    而 `u_w` 是每个候选一个数，把 200 个向量原样印出去只是把噪声搬了个地方。
    `bagged_w` 是 `samples` 的均值 —— **它才是本项目交付的目标函数**，
    所以它进产物，而 `samples` 不进。
    """

    n_resamples: int
    ci_level: float
    searcher: str
    estimates: tuple[WeightEstimate, ...]
    agreement_lo: float
    agreement_hi: float
    agreement_std: float
    point_agreement: float
    samples: tuple[tuple[float, ...], ...] = ()
    bagged_w: tuple[float, ...] = ()
    argmax_w: tuple[float, ...] = ()

    @property
    def n_covered(self) -> int:
        return sum(1 for e in self.estimates if e.covers_truth)

    @property
    def n_components(self) -> int:
        return len(self.estimates)

    @property
    def coverage(self) -> float:
        return self.n_covered / self.n_components if self.n_components else 0.0

    @property
    def widest(self) -> WeightEstimate | None:
        return max(self.estimates, key=lambda e: e.width) if self.estimates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "ci_level": round(self.ci_level, 4),
            "searcher": self.searcher,
            "estimates": [e.to_dict() for e in self.estimates],
            "n_covered": self.n_covered,
            "n_components": self.n_components,
            "coverage": round(self.coverage, 6),
            "agreement_lo": round(self.agreement_lo, 6),
            "agreement_hi": round(self.agreement_hi, 6),
            "agreement_std": round(self.agreement_std, 6),
            "point_agreement": round(self.point_agreement, 6),
            "argmax_w": {k: round(v, 6) for k, v in zip(COMPONENTS, self.argmax_w)} if self.argmax_w else None,
            "bagged_w": {k: round(v, 6) for k, v in zip(COMPONENTS, self.bagged_w)} if self.bagged_w else None,
        }


@dataclass(frozen=True)
class ProviderReport:
    """一个 provider 的实测档案。**「声明」与「实测」分两栏**是本项目的核心纪律：
    配置里说它有多可靠不算数，anchor 上测出来的才算。"""

    provider_id: str
    kind: str
    scale: str
    declared_direction: int
    detected_direction: int
    blind_components: tuple[str, ...]
    coverage: float
    mae_before: float
    mae_after: float
    reliability: float
    solo_agreement: float
    solo_agreement_cal: float
    specific_error: int
    shared_error: int
    n_pairs: int
    degenerate: bool
    note: str = ""

    @property
    def direction_correct(self) -> bool:
        return self.detected_direction == self.declared_direction

    @property
    def calibration_helps(self) -> bool:
        return self.mae_after <= self.mae_before

    @property
    def specific_error_rate(self) -> float:
        return self.specific_error / self.n_pairs if self.n_pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "kind": self.kind,
            "scale": self.scale,
            "declared_direction": self.declared_direction,
            "detected_direction": self.detected_direction,
            "direction_correct": self.direction_correct,
            "blind_components": list(self.blind_components),
            "coverage": round(self.coverage, 6),
            "mae_before": round(self.mae_before, 6),
            "mae_after": round(self.mae_after, 6),
            "reliability": round(self.reliability, 6),
            "solo_agreement": round(self.solo_agreement, 6),
            "solo_agreement_cal": round(self.solo_agreement_cal, 6),
            "specific_error": self.specific_error,
            "shared_error": self.shared_error,
            "specific_error_rate": round(self.specific_error_rate, 6),
            "n_pairs": self.n_pairs,
            "degenerate": self.degenerate,
            "note": self.note,
        }


@dataclass(frozen=True)
class MethodScore:
    """一个方法在 holdout 上的成绩。

    四个数必须一起看，**单看任何一个都会被骗**：

    | 列 | 能被怎么刷 |
    | --- | --- |
    | `agreement` | 与人类标签逐对相等（平局算一个标签）。人类只判 5% 平局，所以任何谨慎的方法都会被它重罚 |
    | `tie_rate` | 判平局就能躲开所有难题 |
    | `agreement_decisive` | 只在最有把握时下结论 ⇒ 接近 1 |
    | `agreement_truth` | **对干净标签（`sign(Δq)`）的一致性，平局算半分** |

    `agreement_truth` 是唯一**不带人类标签噪声**的一列（用的是真值 `Δq`，
    只在 `benchmark` 这个第三方里算得出来），也是唯一**对「判平局」公平**的一列
    （平局拿半分，既不是满分也不是零分）。本项目的主结论以它为准。

    `agreement_decisive` 允许是 `nan`（一列果断判决都没有 ⇒ 这个数**无定义**）。
    落盘时写成 `null`，因为 `NaN` 不是合法 JSON，而「无定义」在报告里必须
    印成 `—` 而不是 `0.0%` —— 后者会被读成「一果断就全错」，那是没测过的话。
    """

    method: str
    label: str
    agreement: float
    tie_rate: float
    decisive_rate: float
    agreement_decisive: float
    n_pairs: int
    agreement_truth: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        ad = float(self.agreement_decisive)
        at = float(self.agreement_truth)
        return {
            "method": self.method,
            "label": self.label,
            "agreement": round(self.agreement, 6),
            "tie_rate": round(self.tie_rate, 6),
            "decisive_rate": round(self.decisive_rate, 6),
            "agreement_decisive": (None if ad != ad else round(ad, 6)),
            "agreement_truth": (None if at != at else round(at, 6)),
            "n_pairs": self.n_pairs,
        }


@dataclass(frozen=True)
class ErrorDecomposition:
    """误差分解 —— **「组合为什么有用」的机制表**。

    用**干净标签**（`sign(Δquality)`，剔除近平局）算，于是「不可约误差」才有意义：

    - `shared`：所有 provider 都错的那些对 ⇒ 真值里那块谁也读不到的残差 + 共有盲区；
    - `specific[p]`：p 错而多数其他 provider 对的那些对 ⇒ p 的**特有偏差**。
      组合的价值就是把这一列压下去 —— 它压不下去的部分才是真·不可约。
    """

    n_pairs: int
    shared: int
    specific: Mapping[str, int]
    mixture_specific: int
    mixture_shared: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "shared": self.shared,
            "shared_rate": round(self.shared / self.n_pairs, 6) if self.n_pairs else 0.0,
            "specific": {k: int(v) for k, v in self.specific.items()},
            "mixture_specific": self.mixture_specific,
            "mixture_shared": self.mixture_shared,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """holdout 上的全部成绩。"""

    n_pairs: int
    n_pairs_clean: int
    label_pos_rate: float
    label_tie_rate: float
    label_noise_measured: float
    methods: tuple[MethodScore, ...]
    providers: tuple[ProviderReport, ...]
    decomposition: ErrorDecomposition
    ece: float
    uncertainty_mean: float
    uncertainty_mae_correlation: float
    per_family: Mapping[str, Mapping[str, float]]
    bootstrap: BootstrapResult
    train_agreement: float
    overfit_gap: float
    #: 最高四分位 u 的 `mean|误差|` ÷ 最低四分位的。**比秩相关稳健**：
    #: u 与误差的关系是弱单调的（误差由模型偏差主导，u 只看得到测量噪声），
    #: 秩相关在 160 个点上会被噪声吃掉。
    uncertainty_lift: float = 1.0
    #: `mean(R − quality)`。**一个恒定的偏移不是不确定度。** `R = Σ w·C` 里
    #: 各分量的均值并不相等，所以只要 `w ≠ w_true` 就会出现一个常数错位
    #: （实测 −0.23）。ECE 量的是「u 与误差**散布**是否匹配」，把常数错位算进去
    #: 等于让不确定度去背一个偏差的账 —— 所以 ECE 用去偏后的误差，而偏移本身
    #: 单独报出来，因为它是一条结论：**reward 只有序意义，尺度要靠校准。**
    reward_bias: float = 0.0

    def method(self, name: str) -> MethodScore | None:
        for m in self.methods:
            if m.method == name:
                return m
        return None

    def provider(self, pid: str) -> ProviderReport | None:
        for p in self.providers:
            if p.provider_id == pid:
                return p
        return None

    @property
    def best_single_calibrated(self) -> ProviderReport | None:
        return max(self.providers, key=lambda p: p.solo_agreement_cal) if self.providers else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_pairs_clean": self.n_pairs_clean,
            "label_pos_rate": round(self.label_pos_rate, 6),
            "label_tie_rate": round(self.label_tie_rate, 6),
            "label_noise_measured": round(self.label_noise_measured, 6),
            "methods": [m.to_dict() for m in self.methods],
            "providers": [p.to_dict() for p in self.providers],
            "decomposition": self.decomposition.to_dict(),
            "ece": round(self.ece, 6),
            "reward_bias": round(self.reward_bias, 6),
            "uncertainty_mean": round(self.uncertainty_mean, 6),
            "uncertainty_mae_correlation": round(self.uncertainty_mae_correlation, 6),
            "uncertainty_lift": round(self.uncertainty_lift, 6),
            "per_family": {k: {m: round(v, 6) for m, v in d.items()} for k, d in self.per_family.items()},
            "bootstrap": self.bootstrap.to_dict(),
            "train_agreement": round(self.train_agreement, 6),
            "overfit_gap": round(self.overfit_gap, 6),
        }


@dataclass(frozen=True)
class ExpectationRow:
    """一条**事先声明**的设计期望 × 实测。

    期望必须写在 `configs/default.json` 里（跑之前就写死），否则「期望被推翻」
    这件事就变成事后编故事。被推翻的那几条是附录 B 里最值钱的部分。
    """

    name: str
    claim: str
    expected: bool
    observed: bool
    held: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "claim": self.claim,
            "expected": self.expected,
            "observed": self.observed,
            "held": self.held,
            "overturned": self.expected and not self.observed,
            "note": self.note,
        }


@dataclass(frozen=True)
class AcceptanceCheck:
    """一条验收项。`observed` 是实测值，`threshold` 是配置里的门槛。"""

    name: str
    threshold: float
    observed: float
    passed: bool
    comparator: str = ">="
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "observed": self.observed,
            "passed": self.passed,
            "comparator": self.comparator,
            "note": self.note,
        }


@dataclass(frozen=True)
class AdversarialCase:
    """对抗用例。**不依赖产物** —— 「检测器还能不能工作」和「这一批产物对不对」
    是两件事，产物过期时前者仍然必须能跑。"""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class VerifyResult:
    """验收 + 对抗的总结果。"""

    checks: tuple[AcceptanceCheck, ...]
    adversarial: tuple[AdversarialCase, ...]

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_checks if self.n_checks else 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def adversarial_passed(self) -> int:
        return sum(1 for c in self.adversarial if c.passed)

    def failing(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "adversarial": [a.to_dict() for a in self.adversarial],
            "n_passed": self.n_passed,
            "n_checks": self.n_checks,
            "pass_rate": round(self.pass_rate, 6),
            "all_passed": self.all_passed,
            "adversarial_passed": self.adversarial_passed,
            "n_adversarial": len(self.adversarial),
        }


def to_dicts(items: Iterable[Any]) -> list[dict[str, Any]]:
    """把一串带 `to_dict` 的对象转成 dict 列表（产物序列化用）。"""
    return [it.to_dict() for it in items]
