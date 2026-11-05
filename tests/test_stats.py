"""stats：三条约定的冒烟测试（平均秩、线性插值分位数、常数列不返回 NaN）。"""
from __future__ import annotations

import pytest

from rewardfactory.stats import (
    norm_cdf,
    pearson,
    quantile,
    ranks,
    spearman,
    variance,
)


def test_quantile_linear_interpolation() -> None:
    xs = [0.0, 10.0, 20.0, 30.0]
    assert quantile(xs, 0.0) == 0.0
    assert quantile(xs, 1.0) == 30.0
    assert quantile(xs, 0.5) == 15.0
    assert quantile(xs, 0.25) == 7.5


def test_ranks_average_ties() -> None:
    assert ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]


def test_pearson_constant_is_zero() -> None:
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_spearman_monotonic_is_one() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_norm_cdf_center() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)


def test_variance_sample_vs_population() -> None:
    xs = [1.0, 2.0, 3.0]
    assert variance(xs, sample=True) == pytest.approx(1.0)
    assert variance(xs, sample=False) == pytest.approx(2.0 / 3.0)
