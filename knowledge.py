# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 知识图谱模块
第一性原理：知识 = 节点(概念) + 边(关系)，一切经验都可以表示为图
"""

import os
import json
import sqlite3
import time
import logging
import threading
import networkx as nx
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("agent.knowledge")


# ============================================================
# 数据结构 - 知识的基本原子
# ============================================================
@dataclass
class KnowledgeNode:
    """知识节点 - 一个概念/经验/事实"""
    id: str = ""
    title: str = ""
    content: str = ""
    category: str = "general"  # general, experience, method, fact, task, error
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    importance: float = 0.5  # 0-1, 越高越重要
    
    def __post_init__(self):
        if not self.id:
            self.id = f"n_{int(time.time()*1000)}_{hash(self.title) % 10000}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class KnowledgeEdge:
    """知识边 - 两个概念之间的关系"""
    source: str = ""
    target: str = ""
    relation: str = "related_to"  # related_to, caused_by, solved_by, improved_by, part_of
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# ============================================================
# 知识图谱核心 - SQLite持久化 + NetworkX图计算
# ============================================================
class KnowledgeGraph:
    """
    知识图谱
    第一性原理：
    - 存储层(SQLite)：保证持久化，重启不丢失
    - 计算层(NetworkX)：保证图查询效率
    - 两者通过双向同步保持一致
    """
    
    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._graph = nx.DiGraph()
        self._lock = threading.Lock()  # 线程安全锁
        self._conn = None
        self._init_db()
        self._load_from_db()
        logger.info(f"知识图谱初始化完成: {self._graph.number_of_nodes()} 节点, {self._graph.number_of_edges()} 边")
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（线程安全）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT DEFAULT '',
                category TEXT DEFAULT 'general',
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT,
                importance REAL DEFAULT 0.5
            );
            
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT DEFAULT 'related_to',
                weight REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id),
                UNIQUE(source, target, relation)
            );
            
            CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category);
            CREATE INDEX IF NOT EXISTS idx_nodes_tags ON nodes(tags);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
        """)
        conn.commit()
        logger.info("数据库表结构初始化完成")
    
    def _load_from_db(self):
        """从数据库加载所有数据到内存图"""
        conn = self._get_conn()
        
        # 加载节点
        cursor = conn.execute("SELECT * FROM nodes")
        for row in cursor:
            node = KnowledgeNode(
                id=row['id'], title=row['title'], content=row['content'],
                category=row['category'], tags=json.loads(row['tags']),
                metadata=json.loads(row['metadata']),
                created_at=row['created_at'], updated_at=row['updated_at'],
                importance=row['importance']
            )
            self._graph.add_node(node.id, **asdict(node))
        
        # 加载边
        cursor = conn.execute("SELECT * FROM edges")
        for row in cursor:
            self._graph.add_edge(
                row['source'], row['target'],
                relation=row['relation'], weight=row['weight'],
                metadata=json.loads(row['metadata']),
                created_at=row['created_at']
            )
    
    # ---- 节点操作 ----
    
    def add_node(self, node: KnowledgeNode) -> str:
        """添加知识节点（线程安全）"""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO nodes 
                   (id, title, content, category, tags, metadata, created_at, updated_at, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node.id, node.title, node.content, node.category,
                 json.dumps(node.tags, ensure_ascii=False),
                 json.dumps(node.metadata, ensure_ascii=False),
                 node.created_at, node.updated_at, node.importance)
            )
            conn.commit()
            self._graph.add_node(node.id, **asdict(node))
            logger.debug(f"添加节点: {node.title} [{node.category}]")
            return node.id
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """获取节点"""
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None
    
    def search_nodes(self, query: str, category: Optional[str] = None, 
                     limit: int = 20) -> List[Dict]:
        """搜索节点 - 基于关键词匹配"""
        results = []
        query_lower = query.lower()
        
        for node_id, data in self._graph.nodes(data=True):
            if category and data.get('category') != category:
                continue
            # 标题、内容、标签匹配
            score = 0
            if query_lower in data.get('title', '').lower():
                score += 3
            if query_lower in data.get('content', '').lower():
                score += 1
            for tag in data.get('tags', []):
                if query_lower in tag.lower():
                    score += 2
            if score > 0:
                results.append((score, dict(data)))
        
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:limit]]
    
    def get_nodes_by_category(self, category: str) -> List[Dict]:
        """按类别获取节点"""
        return [
            dict(data) for _, data in self._graph.nodes(data=True)
            if data.get('category') == category
        ]
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点及其关联边（线程安全）"""
        with self._lock:
            if node_id not in self._graph:
                return False
            conn = self._get_conn()
            conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()
            self._graph.remove_node(node_id)
            logger.debug(f"删除节点: {node_id}")
            return True
    
    # ---- 边操作 ----
    
    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """添加关系边（线程安全）"""
        with self._lock:
            if edge.source not in self._graph or edge.target not in self._graph:
                logger.warning(f"边引用了不存在的节点: {edge.source} -> {edge.target}")
                return False
            
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO edges (source, target, relation, weight, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (edge.source, edge.target, edge.relation, edge.weight,
                     json.dumps(edge.metadata, ensure_ascii=False), edge.created_at)
                )
                conn.commit()
                self._graph.add_edge(
                    edge.source, edge.target,
                    relation=edge.relation, weight=edge.weight,
                    metadata=edge.metadata, created_at=edge.created_at
                )
                logger.debug(f"添加边: {edge.source} -[{edge.relation}]-> {edge.target}")
                return True
            except Exception as e:
                logger.error(f"添加边失败: {e}")
                return False
    
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Dict]:
        """获取邻居节点"""
        if node_id not in self._graph:
            return []
        neighbors = []
        for neighbor_id in self._graph.neighbors(node_id):
            edge_data = self._graph.edges[node_id, neighbor_id]
            if relation and edge_data.get('relation') != relation:
                continue
            neighbor_data = dict(self._graph.nodes[neighbor_id])
            neighbor_data['_relation'] = edge_data.get('relation')
            neighbor_data['_weight'] = edge_data.get('weight', 1.0)
            neighbors.append(neighbor_data)
        return neighbors
    
    # ---- 图查询 ----
    
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """查找两个节点之间的最短路径"""
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> Dict:
        """获取与某节点相关的知识子图"""
        if node_id not in self._graph:
            return {"nodes": [], "edges": []}
        
        # BFS 获取子图
        subgraph_nodes = set()
        queue = [(node_id, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue
            subgraph_nodes.add(current)
            for neighbor in self._graph.neighbors(current):
                if neighbor not in subgraph_nodes:
                    queue.append((neighbor, depth + 1))
        
        nodes = [dict(self._graph.nodes[n]) for n in subgraph_nodes]
        edges = [
            {
                "source": u, "target": v,
                "relation": self._graph.edges[u, v].get('relation'),
                "weight": self._graph.edges[u, v].get('weight', 1.0)
            }
            for u, v in self._graph.edges(subgraph_nodes)
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    def get_stats(self) -> Dict:
        """获取图谱统计信息"""
        categories = {}
        for _, data in self._graph.nodes(data=True):
            cat = data.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "categories": categories,
            "db_size_bytes": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        }
    
    # ---- 经验管理（高级接口）----
    
    def store_experience(self, title: str, content: str, 
                         category: str = "experience",
                         tags: Optional[List[str]] = None,
                         related_to: Optional[List[str]] = None,
                         importance: float = 0.5) -> str:
        """存储一条经验 - 高级接口"""
        node = KnowledgeNode(
            title=title, content=content,
            category=category,
            tags=tags or [],
            importance=importance
        )
        node_id = self.add_node(node)
        
        # 建立关联
        if related_to:
            for rel_id in related_to:
                if rel_id in self._graph:
                    self.add_edge(KnowledgeEdge(
                        source=node_id, target=rel_id,
                        relation="related_to"
                    ))
        
        return node_id
    
    def query_experience(self, query: str, limit: int = 10) -> List[Dict]:
        """查询相关经验"""
        return self.search_nodes(query, category="experience", limit=limit)
    
    def store_task_result(self, task_name: str, result: str, 
                          success: bool = True,
                          error: str = "",
                          method: str = "") -> str:
        """存储任务执行结果作为经验"""
        category = "experience" if success else "error"
        content = f"任务: {task_name}\n结果: {result}\n方法: {method}"
        if error:
            content += f"\n错误: {error}"
        
        node = KnowledgeNode(
            title=f"任务记录: {task_name}",
            content=content,
            category=category,
            tags=["task", task_name, "success" if success else "error"],
            importance=0.7 if success else 0.9,  # 失败经验更重要
            metadata={"task_name": task_name, "success": success, "method": method}
        )
        return self.add_node(node)
    
    def store_optimization(self, original_method: str, optimized_method: str,
                           reason: str, improvement: str) -> str:
        """存储优化记录"""
        node = KnowledgeNode(
            title=f"优化: {original_method[:30]}",
            content=f"原始方法: {original_method}\n优化方法: {optimized_method}\n原因: {reason}\n改进: {improvement}",
            category="method",
            tags=["optimization", "method"],
            importance=0.8,
            metadata={
                "original": original_method,
                "optimized": optimized_method,
                "reason": reason,
                "improvement": improvement
            }
        )
        return self.add_node(node)
    
    def get_relevant_context(self, task_description: str) -> str:
        """为任务获取相关上下文 - Agent 执行前的关键步骤"""
        # 搜索相关经验
        experiences = self.search_nodes(task_description, limit=5)
        methods = self.search_nodes(task_description, category="method", limit=3)
        errors = self.search_nodes(task_description, category="error", limit=3)
        
        context_parts = []
        
        if experiences:
            context_parts.append("=== 相关经验 ===")
            for exp in experiences:
                context_parts.append(f"- {exp.get('title')}: {exp.get('content', '')[:200]}")
        
        if methods:
            context_parts.append("\n=== 相关方法 ===")
            for m in methods:
                context_parts.append(f"- {m.get('title')}: {m.get('content', '')[:200]}")
        
        if errors:
            context_parts.append("\n=== 历史错误(避免重犯) ===")
            for e in errors:
                context_parts.append(f"- {e.get('title')}: {e.get('content', '')[:200]}")
        
        return "\n".join(context_parts) if context_parts else "暂无相关经验"
    
    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("知识图谱数据库连接已关闭")
