"""
Neo4j 后端（知识图谱主后端）。

使用 Cypher 存储与查询，提供与 NetworkX 后端一致的查询接口。
未安装 neo4j 驱动、或连不上服务时，is_available() 返回 False，
由 graph_manager 自动降级到 NetworkX（对应项目一贯的优雅降级风格）。

部署：docker run -p 7687:7687 -p 7474:7474 neo4j:5  （默认账号 neo4j/neo4j，首次登录改密）
"""
from config.settings import settings
from knowledge_graph import data_model as dm

try:
    from neo4j import GraphDatabase
    _NEO4J_DRIVER_OK = True
except ImportError:
    _NEO4J_DRIVER_OK = False


class Neo4jStore:
    def __init__(self):
        self._driver = None
        self._built = False

    # ---------- 生命周期 ----------
    def _connect(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
        self._driver.verify_connectivity()
        return self._driver

    def is_available(self) -> bool:
        if not _NEO4J_DRIVER_OK:
            return False
        try:
            self._connect()
            return True
        except Exception:
            return False

    def is_built(self) -> bool:
        return self._built

    def load_nodes(self, nodes: list):
        driver = self._connect()
        with driver.session() as s:
            for n in nodes:
                s.run(
                    "MERGE (x:Node {key: $key}) "
                    "SET x.type=$type, x.label=$label, x.aliases=$aliases",
                    key=n["key"], type=n["type"], label=n["label"], aliases=n.get("aliases") or [],
                )
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.label)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.key)")

    def load_edges(self, edges: list):
        driver = self._connect()
        with driver.session() as s:
            for e in edges:
                s.run(
                    "MATCH (a:Node {key:$src}), (b:Node {key:$dst}) "
                    "MERGE (a)-[r:REL {rel:$rel}]->(b) SET r.label=$label",
                    src=e["src"], dst=e["dst"],
                    rel=e["rel"], label=dm.RELATION_LABELS.get(e["rel"], e["rel"]),
                )

    def save(self):
        """Neo4j即写即存，无需额外持久化"""
        pass

    def load_from_file(self) -> bool:
        return False  # Neo4j 不读本地文件

    def mark_built(self):
        self._built = True

    # ---------- 查询 ----------
    def _all_labels(self) -> list:
        """拉取全部节点 label/aliases/type（实体集合小，供实体识别用）"""
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (n:Node) RETURN n.label AS label, n.aliases AS aliases, n.type AS type"
            ).data()
        return [(r["type"], r["label"], r.get("aliases") or []) for r in rows]

    def find_entities(self, text: str) -> list:
        found, seen = [], set()
        for ntype, label, aliases in self._all_labels():
            matched = label in text or any(a and a != label and a in text for a in aliases)
            if matched and label not in seen:
                seen.add(label)
                found.append((ntype, label))
        return found

    def get_neighbors(self, name: str, rel_types: list = None) -> list:
        extra = "AND r.rel IN $rels" if rel_types else ""
        with self._driver.session() as s:
            rows = s.run(
                f"MATCH (n:Node {{label:$name}})-[r:REL]->(m:Node) "
                f"WHERE m.label IS NOT NULL {extra} "
                f"RETURN r.rel AS rel, r.label AS rlabel, m.label AS target",
                name=name, rels=rel_types or [],
            ).data()
        return [(r["rel"], r["rlabel"], r["target"], {}) for r in rows]

    def get_industry_members(self, industry: str) -> list:
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (c:Node)-[r:REL {rel:'belongs_to'}]->(i:Node {label:$industry}) "
                "RETURN c.label AS company",
                industry=industry,
            ).data()
        return [r["company"] for r in rows if r.get("company")]

    def get_relations_between(self, name_a: str, name_b: str) -> list:
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (a:Node {label:$a})-[r:REL]->(b:Node {label:$b}) "
                "RETURN r.rel AS rel, r.label AS rlabel, 'out' AS dir "
                "UNION "
                "MATCH (a:Node {label:$a})<-[r:REL]-(b:Node {label:$b}) "
                "RETURN r.rel AS rel, r.label AS rlabel, 'in' AS dir",
                a=name_a, b=name_b,
            ).data()
        return [
            (r["rel"], f"{r['rlabel']}（反向）" if r["dir"] == "in" else r["rlabel"], {})
            for r in rows
        ]

    def find_path(self, name_a: str, name_b: str, max_hops: int = 3) -> list:
        with self._driver.session() as s:
            rows = s.run(
                "MATCH p = shortestPath((a:Node {label:$a})-[*..$hops]->(b:Node {label:$b})) "
                "RETURN [n IN nodes(p) | n.label] AS labels",
                a=name_a, b=name_b, hops=max_hops,
            ).data()
        return (rows[0]["labels"] if rows and rows[0].get("labels") else []) or []

    def stats(self) -> dict:
        with self._driver.session() as s:
            nodes = s.run("MATCH (n:Node) RETURN count(n) AS c").single()["c"]
            edges = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        return {"backend": "neo4j", "nodes": nodes, "edges": edges}
