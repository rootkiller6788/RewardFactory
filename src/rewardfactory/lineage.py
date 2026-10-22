"""血缘 DAG：把「这个 Reward 是怎么来的」画成一张可查的图。

本项目的血缘是**阶段级**的，不是逐记录级的。理由：候选有 480 个，
逐候选建节点会得到一张 3000 节点的图 —— 那种图不能回答问题，只能证明
「我们有血缘」。阶段级 + 逐 provider 扇出才是这个形状该有的粒度：

```
truth ── pool ──┬── prov:rule ──── norm:rule ──── cal:rule ────┐
                ├── prov:rm_small ─ norm:rm_small ─ cal:rm_small ┤
                ├── prov:rm_large ─ norm:rm_large ─ cal:rm_large ┼── mix ──┐
                ├── prov:llm_judge ─ norm:llm_judge ─ cal:llm_judge ┤        ├── reward ── bench ── expect ── accept
                └── prov:rm_legacy ─ norm:rm_legacy ─ cal:rm_legacy ┘        │
                                    search:grid ──┐                          │
                                    search:random ┼── w* ─────────────────────┘
                                    search:coord. ┤
                                    search:forward┘
```

## 三条被断言的不变量

1. **`mix` 的父节点恰好是全部 `cal:*`**，一个不多一个不少。
   「一个不少」保证每个 provider 的校准结果都被组合接住了；
   「一个不多」保证图没把落选的东西偷偷接进来 —— 于是「到底组合了哪几个
   provider」看一张图就知道，不必翻报告。
2. **`w*` 的父节点恰好是全部 `search:*`**。少一个就说明那个搜索器跑了但结果没进
   决策，而报告里还印着它的成绩 —— 那种图是骗人的。
3. **`reward` 的父节点恰好是 `mix` 与 `w*`**。这两样是构成目标函数的全部输入，
   多一样就说明有东西绕过了流水线。

这三条都是 `acceptance.py` 里真的去查的，不是写在文档里的愿望。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .rng import canonical_json, sha256_text

__all__ = [
    "Node",
    "Edge",
    "Lineage",
    "build_lineage",
    "TRUTH_NODE",
    "POOL_NODE",
    "MIX_NODE",
    "WEIGHT_NODE",
    "REWARD_NODE",
    "BENCH_NODE",
    "EXPECT_NODE",
    "ACCEPT_NODE",
]

TRUTH_NODE = "truth"
POOL_NODE = "pool"
MIX_NODE = "mix"
WEIGHT_NODE = "w*"
REWARD_NODE = "reward"
BENCH_NODE = "bench"
EXPECT_NODE = "expect"
ACCEPT_NODE = "accept"


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "meta": dict(self.meta)}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str = "data"

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind}


class LineageError(ValueError):
    """血缘图不合法。**图是产物的一部分，它错了不能静默过去。**"""


class Lineage:
    """有向无环图。`add_node` 拒重复，`add_edge` 校验两端存在。"""

    __slots__ = ("_nodes", "_edges", "_out", "_in")

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}

    # ---- 构建

    def add_node(self, node: Node) -> Node:
        if node.id in self._nodes:
            raise LineageError(f"节点已存在：{node.id!r}（血缘节点 id 必须唯一）")
        self._nodes[node.id] = node
        self._out[node.id] = []
        self._in[node.id] = []
        return node

    def add_edge(self, src: str, dst: str, kind: str = "data") -> Edge:
        if src not in self._nodes:
            raise LineageError(f"边的起点不存在：{src!r}")
        if dst not in self._nodes:
            raise LineageError(f"边的终点不存在：{dst!r}")
        e = Edge(src=src, dst=dst, kind=kind)
        self._edges.append(e)
        self._out[src].append(dst)
        self._in[dst].append(src)
        return e

    # ---- 查询

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)

    def node(self, nid: str) -> Node:
        if nid not in self._nodes:
            raise LineageError(f"没有这个节点：{nid!r}")
        return self._nodes[nid]

    def has(self, nid: str) -> bool:
        return nid in self._nodes

    def parents(self, nid: str) -> tuple[str, ...]:
        self.node(nid)
        return tuple(self._in[nid])

    def children(self, nid: str) -> tuple[str, ...]:
        self.node(nid)
        return tuple(self._out[nid])

    def roots(self) -> tuple[str, ...]:
        return tuple(n for n in self._nodes if not self._in[n])

    def leaves(self) -> tuple[str, ...]:
        return tuple(n for n in self._nodes if not self._out[n])

    def by_kind(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(n for n, nd in self._nodes.items() if nd.kind == kind))

    def depth(self, nid: str) -> int:
        """从任一根到该节点的**最长**路径长度。"""
        memo: dict[str, int] = {}

        def go(n: str, stack: set[str]) -> int:
            if n in memo:
                return memo[n]
            if n in stack:
                raise LineageError(f"血缘图里有环：{n!r}")
            stack.add(n)
            ps = self._in[n]
            d = 0 if not ps else 1 + max(go(p, stack) for p in ps)
            stack.discard(n)
            memo[n] = d
            return d

        return go(nid, set())

    def path_to_root(self, nid: str) -> tuple[str, ...]:
        """到最近的一条根路径（倒序：从根到 `nid`）。确定性：平手按 id 排序取小。"""
        cur = nid
        chain = [cur]
        seen = {cur}
        while self._in[cur]:
            nxt = sorted(self._in[cur])[0]
            if nxt in seen:
                raise LineageError(f"血缘图里有环：{nxt!r}")
            seen.add(nxt)
            chain.append(nxt)
            cur = nxt
        return tuple(reversed(chain))

    def topological(self) -> tuple[str, ...]:
        """拓扑序。**入度为 0 时按 id 排序取下一个** —— 用 `set` 的迭代顺序
        会让同一张图在不同进程里给出不同的节点顺序，进而让 `dot` 产物不稳定。"""
        indeg = {n: len(self._in[n]) for n in self._nodes}
        ready = sorted(n for n, d in indeg.items() if d == 0)
        out: list[str] = []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for m in self._out[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort()
        if len(out) != len(self._nodes):
            raise LineageError("血缘图里有环：拓扑排序未覆盖全部节点")
        return tuple(out)

    # ---- 产物

    def fingerprint(self) -> str:
        """图指纹。**按节点 id / 边端点排序** —— 直接 `sorted(dict)` 会抛
        `TypeError`（dict 之间没有序），而换成「按插入顺序」又会让指纹依赖构建顺序。"""
        nodes = [self._nodes[n].to_dict() for n in sorted(self._nodes)]
        edges = [e.to_dict() for e in sorted(self._edges, key=lambda e: (e.src, e.dst, e.kind))]
        return sha256_text(canonical_json({"nodes": nodes, "edges": edges}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "nodes": [self._nodes[n].to_dict() for n in self.topological()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def to_dot(self, *, include: Sequence[str] | None = None) -> str:
        """DOT。`include` 给节点 id 前缀白名单（例如只导出因果链那一段）。"""
        keep = set()
        if include:
            for nid in self._nodes:
                if any(nid == p or nid.startswith(p) for p in include):
                    keep.add(nid)
        else:
            keep = set(self._nodes)
        lines = ["digraph rewardfactory {", '  rankdir="LR";', '  node [shape=box, fontname="Helvetica"];']
        for nid in self.topological():
            if nid not in keep:
                continue
            nd = self._nodes[nid]
            lines.append(f'  "{nid}" [label="{nd.label}", kind="{nd.kind}"];')
        for e in self._edges:
            if e.src in keep and e.dst in keep:
                lines.append(f'  "{e.src}" -> "{e.dst}" [kind="{e.kind}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self) -> str:
        """一张按深度分层的 ASCII 概览。给终端看，不必装 graphviz。"""
        by_depth: dict[int, list[str]] = {}
        for nid in self._nodes:
            by_depth.setdefault(self.depth(nid), []).append(nid)
        out = []
        for d in sorted(by_depth):
            out.append(f"  [{d}] " + "  ".join(sorted(by_depth[d])))
        return "\n".join(out)


def build_lineage(
    *,
    providers: Sequence[str],
    searchers: Sequence[str],
    searched: str,
    n_candidates: int,
    n_pairs: Mapping[str, int],
    mix_inputs: Sequence[str],
    weight_inputs: Sequence[str],
    extra_meta: Mapping[str, Any] | None = None,
) -> Lineage:
    """按流水线的实际形态建图。

    这个函数**不吃结果对象**，只吃「拓扑事实」（有哪几个 provider、哪几个搜索器、
    谁被组合进去了）。理由：让它依赖 `BenchmarkResult` 会形成一个「要出报告才能建图、
    要建图才能出报告」的环。拓扑事实是构建时就确定的。
    """
    lin = Lineage()
    lin.add_node(Node(TRUTH_NODE, "truth", "真值世界（latent 分量 + 注入权重 + 标注）",
                      {"n_candidates": n_candidates, "pairs": dict(n_pairs)}))
    lin.add_node(Node(POOL_NODE, "pool", f"provider 池（{len(providers)} 个）",
                      {"providers": list(providers)}))
    lin.add_edge(TRUTH_NODE, POOL_NODE, "simulate")

    for pid in providers:
        pn = f"prov:{pid}"
        nn = f"norm:{pid}"
        cn = f"cal:{pid}"
        lin.add_node(Node(pn, "provider", f"原始输出 · {pid}", {}))
        lin.add_node(Node(nn, "normalize", f"归一化（方向 + 盲区）· {pid}", {}))
        lin.add_node(Node(cn, "calibrate", f"校准（anchor 拟合）· {pid}", {}))
        lin.add_edge(POOL_NODE, pn, "score")
        lin.add_edge(pn, nn, "normalize")
        lin.add_edge(nn, cn, "calibrate")

    lin.add_node(Node(MIX_NODE, "mix", "可靠性加权 → 分量 C[k] → 混合", {}))
    for pid in mix_inputs:
        lin.add_edge(f"cal:{pid}", MIX_NODE, "aggregate")

    for name in searchers:
        sn = f"search:{name}"
        lin.add_node(Node(sn, "search", f"权重搜索 · {name}", {}))
        lin.add_edge(MIX_NODE, sn, "objective")

    lin.add_node(Node(WEIGHT_NODE, "weight", f"最优权重 w*（来自 {searched}）", {}))
    for name in weight_inputs:
        lin.add_edge(f"search:{name}", WEIGHT_NODE, "select")

    lin.add_node(Node(REWARD_NODE, "reward", "Reward 函数 R = Σ w*·C[k] + u", {}))
    lin.add_edge(MIX_NODE, REWARD_NODE, "instantiate")
    lin.add_edge(WEIGHT_NODE, REWARD_NODE, "instantiate")

    lin.add_node(Node(BENCH_NODE, "benchmark", "holdout 一致性 / 误差分解 / bootstrap",
                      dict(extra_meta or {})))
    lin.add_edge(REWARD_NODE, BENCH_NODE, "evaluate")
    # 评测是**第三方**：它也要读真值世界。这条边必须画出来 —— 它是
    # 「搜权重的人不给自己打分」这个分工在图上的样子。
    lin.add_edge(TRUTH_NODE, BENCH_NODE, "ground-truth")

    lin.add_node(Node(EXPECT_NODE, "expectations", "九条声明的期望 × 实测", {}))
    lin.add_edge(BENCH_NODE, EXPECT_NODE, "check")

    lin.add_node(Node(ACCEPT_NODE, "acceptance", "验收门槛 + 对抗用例", {}))
    lin.add_edge(EXPECT_NODE, ACCEPT_NODE, "gate")
    lin.add_edge(BENCH_NODE, ACCEPT_NODE, "gate")
    return lin
