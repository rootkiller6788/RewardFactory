# Contributing

## 环境

核心链路只依赖标准库（Python >= 3.12），不需要 `pip install`：

```bash
python main.py run        # 整条流水线 + 16 个产物
python main.py check      # 验收闸门：全过 rc=0，否则 rc=1
python -m pytest -q       # 入口级契约测试
```

Windows 默认 GBK 控制台若仍有显示问题，可用 `set PYTHONUTF8=1`（或 `export PYTHONUTF8=1`）。

## 改动前先问两个问题

1. **这条阈值该不该进 `configs/default.json`？** 代码里不硬编码阈值；新增口径同时补校验。
2. **这条纪律能不能被断言？** 例如「搜索器不读真值」「`normalize.py` 里不出现 `invert`」
   都有对抗用例读源码守着。靠约定不如靠断言。

## 提交

- 一个提交只做一件事，消息用 `feat:` / `fix:` / `docs:` / `test:` / `chore:` 前缀。
- 提交前至少跑 `python main.py check` 和 `python -m pytest -q`。
- 改动了产物格式时，同步更新 `README.md` 的附录 A 与 `CHANGELOG.md`。
- `out/` 是生成物，不入库。

## 禁止

- 不要在 `search/**`、`aggregate.py`、`calibrate.py` 里读 `latent_weights`。
- 不要在 `normalize.py`、`calibrate.py` 里读 `invert` 或 `declared_direction`。
- 不要给核心链路引入第三方依赖；重型接入点写成抛 `NotImplementedError` 的扩展点。
