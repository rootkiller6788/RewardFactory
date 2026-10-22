"""`--json` 契约：stdout 上**恰好**是一段可解析的 UTF-8 JSON。

不依赖 conftest 夹具；子进程的字节显式按 UTF-8 解码，避免 Windows 代码页干扰。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

COMMANDS = [
    "summary",
    "weights",
    "search",
    "benchmark",
    "acceptance",
    "check",
    "lineage",
]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_json_is_one_parseable_document(cmd: str) -> None:
    r = subprocess.run(
        [sys.executable, str(MAIN), cmd, "--json"],
        cwd=ROOT,
        capture_output=True,
    )
    stderr = r.stderr.decode("utf-8", "replace")
    assert r.returncode == 0, f"{cmd} rc={r.returncode}\n{stderr[-2000:]}"
    payload = json.loads(r.stdout.decode("utf-8"))
    assert isinstance(payload, dict)
