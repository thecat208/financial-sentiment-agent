"""
NetworkX 后端（知识图谱的降级实现）。

当 Neo4j 不可用时，用 NetworkX 在内存里维护同样的图，并持久化到 JSON 文件，
提供与 Neo4j 后端一致的查询接口（get_neighbors / get_industry_members / find_path ...），
上层代码不感知后端差异。单机数据量（几百~几万节点）下查询同样是毫秒级。
"""
import json
import os
import threading

import networkx as nx

from knowledge_graph import data_model as dm
from config.settings import settings


class NetworkXStore:
    """NetworkX 有向多重图。节点key=f"{type}:{name}"，节点attrs含 type/label/aliases"""

    def __init__(self, file_path: str = None):
        self._g = nx.MultiDiGraph()
        self._file_path = file_path or settings.KG_FILE_PATH
        self._built = False
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------
    def is_available(self) -> bool:
        return True

    def is_built(self) -> bool:
        return self._built

    def load_nodes(self, nodes: list):
        for n in nodes:
            self._g.add_node(n["key"], type=n["type"], label=n["label"], aliases=n.get("aliases") or [])

    def load_edges(self, edges: list):
        for e in edges:
            self._g.add_edge(
                e["src"], e["dst"], rel=e["rel"],
                label=dm.RELATION_LABELS.get(e["rel"], e["rel"]),
                **(e.get("props") or {}),
            )

    def save(self):
        if not self._file_path:
            return
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self._g), f, ensure_ascii=False)

    def load_from_file(self) -> bool:
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
            self._g = nx.node_link_graph(data)
            self._built = True
            return True
        except Exception:
            return False

    def mark_built(self):
        self._built = True

    # ---------- 查询 ----------
    def _resolve_key(self, name: str) -> str:
        """名字(可能带type前缀) -> 节点key。优先精确key，其次按label匹配"""
        if self._g.has_node(name):
            return name
        for node, attrs in self._g.nodes(data=True):
            if attrs.get("label") == name:
                return node
        return name

    def find_entities(self, text: str) -> list:
        """在正文里找图中出现的实体，返回 [(type, canonical_label), ...]。别名匹配到则返回规范名"""
        found, seen = [], set()
        for _node, attrs in self._g.nodes(data=True):
            label = attrs.get("label", "")
            if not label:
                continue
            aliases = attrs.get("aliases") or []
            matched = label in text or any(a and a != label and a in text for a in aliases)
            if matched and label not in seen:
                seen.add(label)
                found.append((attrs.get("type", ""), label))
        return found

    def get_neighbors(self, name: str, rel_types: list = None) -> list:
        """返回 [(rel, rel_label, target_label, props), ...]"""
        key = self._resolve_key(name)
        results = []
        for _src, dst, data in self._g.edges(key, data=True):
            rel = data.get("rel")
            if rel_types and rel not in rel_types:
                continue
            results.append((rel, data.get("label", rel), self._g.nodes[dst].get("label", dst), data))
        return results

    def get_industry_members(self, industry: str) -> list:
        members = []
        ikey = self._resolve_key(f"industry:{industry}")
        for src, dst, data in self._g.edges(data=True):
            if dst == ikey and data.get("rel") == "belongs_to":
                members.append(self._g.nodes[src].get("label", src))
        return members

    def get_relations_between(self, name_a: str, name_b: str) -> list:
        """返回 [(rel, rel_label, props), ...]，含双向"""
        ka, kb = self._resolve_key(name_a), self._resolve_key(name_b)
        out = []
        # 注意：MultiDiGraph.edges(ka, kb) 的第二个位置参数会被当成 data，
        # 所以用 edges(ka, data=True) 遍历后按目标节点过滤
        for _s, dst, data in self._g.edges(ka, data=True):
            if dst == kb:
                out.append((data.get("rel"), data.get("label", data.get("rel")), data))
        for _s, dst, data in self._g.edges(kb, data=True):
            if dst == ka:
                out.append((data.get("rel"), f"{data.get('label', data.get('rel'))}（反向）", data))
        return out

    def find_path(self, name_a: str, name_b: str, max_hops: int = 3) -> list:
        ka, kb = self._resolve_key(name_a), self._resolve_key(name_b)
        if not self._g.has_node(ka) or not self._g.has_node(kb):
            return []
        try:
            path = nx.shortest_path(self._g, ka, kb)
        except nx.NetworkXNoPath:
            return []
        if len(path) - 1 > max_hops or len(path) - 1 == 0:
            return []
        return [self._g.nodes[k].get("label", k) for k in path]

    def stats(self) -> dict:
        return {
            "backend": "networkx",
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }
