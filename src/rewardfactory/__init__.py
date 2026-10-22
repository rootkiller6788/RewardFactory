"""RewardFactory —— 把「什么才算好」变成一个可搜索、可校准、带不确定度的目标函数。

README 的核心命题：**Reward 不应该只有一个模型。** 单一 RM 会把自己的偏差
变成训练目标的偏差。但「多来几个」不是答案 —— 五个尺度互不可比的打分器
等权平均，比最好的那一个还差。答案是三步：

```
Normalize（方向 + 盲区） → Calibrate（拉到同一把尺子） → Aggregate（按可靠性加权）
                       ↘ Search（连权重本身也是搜出来的）
```

模块地图：

```
rng         确定性随机（只用 random.random()）+ derive_seed + Dirichlet
stats       小统计工具箱（Spearman 平均秩 / 线性插值分位数 / 正态 CDF）
schema      对象链与常量表（**唯一裁决处**）
config      口径与门槛 + validate_config + 三个指纹
world       真值层：latent 分量 + 注入权重 + 带噪声的人类标注 + 不可观测残差
providers   五个不可比的打分器（模拟物理，**唯一允许读 invert 的地方**）
normalize   方向检测 + 盲区检测（**只从 anchor 测，不读配置**）
calibrate   五种校准法 → 统一的分段线性表 + 交叉验证可靠性
aggregate   可靠性加权 → C[k] → R + u（平局判定用 u）
uncertainty 分量项 + 权重项 + ECE + 偏保守比
search/     四个搜索器搜权重 w（**一行真值都不读**）
benchmark   **第三方**评测：读真值算上界、CI 覆盖率、不可约误差
lineage     阶段级血缘 DAG + 三条被断言的不变量
pipeline    编排 + 17 个产物 + 回读（报告从产物字典渲染，不从对象渲染）
report      Markdown 报告（纯函数：产物字典 → 文本）
visualize   自包含 HTML（零外部资源、浅色主题、合法 XML）
acceptance  验收（依赖产物）+ 对抗（**不依赖产物**）+ 读源码的两条墙
cli         15 个命令，其中 13 个只读且绝不写盘
```

一句话：**搜权重的人不给自己打分，打分的人不搜权重。**
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
