"""``python -m rewardfactory`` 入口（等价于 ``python main.py``）。

零安装场景仍需让解释器看见 ``src/``（例如 ``PYTHONPATH=src python -m rewardfactory``），
或先 ``pip install -e .``。``main.py`` 的存在就是为了省掉这一步。
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
