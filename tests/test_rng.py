"""rng：可复现性的地基。种子派生、规范化 JSON、Dirichlet 单纯形。"""
from __future__ import annotations

import math

from rewardfactory.rng import Rng, canonical_json, derive_seed


def test_same_seed_same_sequence() -> None:
    a = Rng("rewardfactory")
    b = Rng("rewardfactory")
    assert [a.random() for _ in range(8)] == [b.random() for _ in range(8)]


def test_derive_seed_avoids_boundary_collision() -> None:
    assert derive_seed("ab", "c") != derive_seed("a", "bc")


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})


def test_dirichlet_lives_on_simplex() -> None:
    xs = Rng("simplex").dirichlet1(5)
    assert len(xs) == 5
    assert all(x > 0.0 for x in xs)
    assert math.isclose(math.fsum(xs), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_sample_does_not_mutate_input() -> None:
    src = [1, 2, 3, 4]
    out = Rng("sample").sample(src, 2)
    assert len(out) == 2
    assert src == [1, 2, 3, 4]
