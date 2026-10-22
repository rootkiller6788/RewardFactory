"""`score --text` 契约：同一文本两次打分逐位一致，不同文本给出不同分数。

这条测试守的是「产物是一个函数」——校准映射与 w* 只从 `out/` 读取，
latent 分量由 sha256(text) 确定性派生，因此结果不依赖进程状态。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"


def run_score(text: str) -> dict:
    r = subprocess.run(
        [sys.executable, str(MAIN), "score", "--text", text, "--json"],
        cwd=ROOT,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
    return json.loads(r.stdout.decode("utf-8"))


def test_same_text_is_reproducible() -> None:
    text = "RewardFactory 的同一个输入必须给出同一个分数。"
    a = run_score(text)
    b = run_score(text)
    assert a["stub"] is True
    assert a["total_score"] == b["total_score"]
    assert a["uncertainty"] == b["uncertainty"]
    assert a["components"] == b["components"]


def test_different_text_gives_different_score() -> None:
    a = run_score("alpha")
    b = run_score("omega")
    assert a["total_score"] != b["total_score"]
