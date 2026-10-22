# Changelog

本项目遵循「每次提交只做一件事、且这件事可被验收」的约定。

## 2026-10-22

### Added

- `check` 子命令：验收清单的退出码版本（全过 rc=0，否则 rc=1），可直接当 CI 闸门。
- `python -m rewardfactory` 入口，与 `python main.py` 等价。
- `tests/test_json_contract.py`：断言 `--json` 的 stdout 恰好是一段可解析的 UTF-8 JSON。
- `tests/test_scoring.py`：断言同文本两次 `score --text` 逐位一致、不同文本给出不同分数。
- `Makefile`：收拢 `run` / `visualize` / `test` / `acceptance` / `check` / `clean`。
- GitHub Actions 工作流：在 Python 3.12 / 3.13 上跑流水线、验收闸门与 pytest。
- `.editorconfig` / `.gitattributes`：统一编码、换行与缩进。

### Fixed

- Windows 默认 GBK 控制台下，`summary` / `benchmark` / `acceptance` / `pairs` / `search`
  因 `−`、`⇒`、`✓`、`✗` 等字符抛 `UnicodeEncodeError`。现在保留当前代码页并对不可编码字符
  降级替换，排版符号改用 GBK 可编码的等价字符。
- `--json` 输出显式写 UTF-8 字节，管道（`jq` / `json.loads`）不再受本机代码页影响。
- `summary` 不再把「组合超过真值权重上界」打印成「还差 -2.88%」，改为分别表述超出/落后。

### Docs

- README 修正 `tests/` 为空的过时说明，并补充 14 个只读命令的口径。
- `tests/conftest.py` 不再引用不存在的 `diagnostics/` 目录。
