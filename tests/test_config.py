"""config：默认配置必须通过校验；几条最容易悄悄失效的约束必须拒绝坏输入。"""
from __future__ import annotations

import copy

import pytest

from rewardfactory.config import DEFAULTS, ConfigError, deep_merge, validate_config


def test_defaults_are_valid() -> None:
    validate_config(copy.deepcopy(DEFAULTS))


def test_split_sum_must_match_n_prompts() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["world"]["anchor_prompts"] += 1
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_blind_rho_tol_must_exceed_direction_tol() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["normalize"]["blind_rho_tol"] = cfg["normalize"]["direction_rho_tol"]
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_deep_merge_replaces_lists() -> None:
    assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


def test_deep_merge_keeps_untouched_keys() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
