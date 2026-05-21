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
        self._lock = threading.RLock()  # 线程安全锁（可重入，支持同一线程嵌套获取）
        self._conn = None
        self._last_sync_version = 0  # 用于多实例同步
        self._init_db()
        self._load_from_db()
        self._last_sync_version = self._get_db_version()
        logger.info(f"知识图谱初始化完成: {self._graph.number_of_nodes()} 节点, {self._graph.number_of_edges()} 边")
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（线程安全）"""
        # [修复5] 在 _get_conn() 中添加 self._lock 保护，防止多线程同时创建连接导致竞态条件
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
                self._conn.row_factory = sqlite3.Row
                # 启用 WAL 模式提高并发性能
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            return self._conn
    
    def _get_db_version(self) -> int:
        """获取数据库版本号（用于多实例同步），包含节点和边的变更"""
        try:
            conn = self._get_conn()
            cursor = conn.execute("SELECT COUNT(*) as cnt, MAX(id) as max_id FROM nodes")
            row = cursor.fetchone()
            node_count = row['cnt'] if row else 0
            max_node_id = row['max_id'] if row and row['max_id'] else ""
            version = node_count * 1000000
            if max_node_id:
                try:
                    nums = [int(s) for s in max_node_id.replace('n_', '').split('_') if s.isdigit()]
                    for n in nums:
                        version += n
                except (ValueError, AttributeError):
                    pass

            # 包含边的变更信息：边数量 + 最大边ID
            cursor = conn.execute("SELECT COUNT(*) as cnt, MAX(id) as max_id FROM edges")
            row = cursor.fetchone()
            edge_count = row['cnt'] if row else 0
            max_edge_id = row['max_id'] if row and row['max_id'] else 0
            version += edge_count * 10000 + (max_edge_id or 0)

            return version
        except Exception as e:
            logger.warning(f"获取数据库版本号失败: {e}")
            return 0
    
    def _check_need_sync(self) -> int:
        """无锁检查版本号，返回当前数据库版本（调用方无需持有锁）"""
        return self._get_db_version()

    def _do_sync(self):
        """执行实际同步（调用方必须已持有 self._lock）"""
        current_version = self._get_db_version()
        if current_version == self._last_sync_version:
            return  # 无需同步

        conn = self._get_conn()

        # 获取数据库中所有节点ID
        db_node_ids = set()
        cursor = conn.execute("SELECT id FROM nodes")
        for row in cursor:
            db_node_ids.add(row['id'])

        # 获取内存图中的节点ID
        mem_node_ids = set(self._graph.nodes())

        # 添加新节点（分批处理，避免超过 SQLITE_MAX_VARIABLE_NUMBER）
        new_ids = db_node_ids - mem_node_ids
        if new_ids:
            new_ids_list = list(new_ids)
            batch_size = 900
            for i in range(0, len(new_ids_list), batch_size):
                batch = new_ids_list[i:i + batch_size]
                placeholders = ','.join('?' * len(batch))
                cursor = conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", batch)
                for row in cursor:
                    node = KnowledgeNode(
                        id=row['id'], title=row['title'], content=row['content'],
                        category=row['category'], tags=json.loads(row['tags']),
                        metadata=json.loads(row['metadata']),
                        created_at=row['created_at'], updated_at=row['updated_at'],
                        importance=row['importance']
                    )
                    self._graph.add_node(node.id, **asdict(node))

        # 删除已移除的节点
        removed_ids = mem_node_ids - db_node_ids
        for node_id in removed_ids:
            if node_id in self._graph:
                self._graph.remove_node(node_id)

        # 同步边
        db_edges = set()
        cursor = conn.execute("SELECT source, target, relation FROM edges")
        for row in cursor:
            db_edges.add((row['source'], row['target'], row['relation']))

        mem_edges = set()
        for u, v, data in self._graph.edges(data=True):
            mem_edges.add((u, v, data.get('relation', 'related_to')))

        # 添加新边
        for source, target, relation in db_edges - mem_edges:
            if source in self._graph and target in self._graph:
                cursor = conn.execute(
                    "SELECT * FROM edges WHERE source=? AND target=? AND relation=?",
                    (source, target, relation)
                )
                row = cursor.fetchone()
                if row:
                    self._graph.add_edge(
                        source, target,
                        relation=row['relation'], weight=row['weight'],
                        metadata=json.loads(row['metadata']),
                        created_at=row['created_at']
                    )

        self._last_sync_version = current_version
        if new_ids or removed_ids or (db_edges != mem_edges):
            logger.debug(f"知识图谱同步: +{len(new_ids)}节点 -{len(removed_ids)}节点, 边变化{len(db_edges - mem_edges)}")

    def _sync_and_lock(self):
        """上下文管理器：获取锁 -> 检查版本 -> 如需同步则同步 -> 遍历图 -> 释放锁
        消除 TOCTOU 竞态：整个同步和图遍历在同一个锁保护下完成"""
        class _SyncLockContext:
            def __init__(self, kg):
                self._kg = kg
            def __enter__(self):
                self._kg._lock.acquire()
                self._kg._do_sync()
                return self._kg
            def __exit__(self, exc_type, exc_val, exc_tb):
                self._kg._lock.release()
                return False
        return _SyncLockContext(self)

    def _sync_from_db(self):
        """从数据库同步最新数据到内存图（多实例共享时调用，向后兼容）"""
        with self._lock:
            self._do_sync()
    
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
        """添加知识节点（线程安全），已存在的节点不会被覆盖"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT OR IGNORE INTO nodes 
                   (id, title, content, category, tags, metadata, created_at, updated_at, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node.id, node.title, node.content, node.category,
                 json.dumps(node.tags, ensure_ascii=False),
                 json.dumps(node.metadata, ensure_ascii=False),
                 node.created_at, node.updated_at, node.importance)
            )
            conn.commit()
            # 更新版本号，确保后续 _sync_and_lock 能检测到变化
            self._last_sync_version = self._get_db_version()
            if cursor.rowcount > 0:
                # 只有实际插入了新行才更新内存图
                self._graph.add_node(node.id, **asdict(node))
                logger.debug(f"添加节点: {node.title} [{node.category}]")
            else:
                # 节点已存在，从SQLite同步最新数据到内存图
                row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone()
                if row:
                    existing = KnowledgeNode(
                        id=row['id'], title=row['title'], content=row['content'],
                        category=row['category'], tags=json.loads(row['tags']),
                        metadata=json.loads(row['metadata']),
                        created_at=row['created_at'], updated_at=row['updated_at'],
                        importance=row['importance']
                    )
                    self._graph.add_node(existing.id, **asdict(existing))
                    logger.debug(f"节点已存在，同步: {existing.title}")
            return node.id
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """获取节点"""
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
            if node_id in self._graph:
                return dict(self._graph.nodes[node_id])
            return None
    
    def search_nodes(self, query: str, category: Optional[str] = None, 
                     limit: int = 20) -> List[Dict]:
        """搜索节点 - 基于关键词匹配"""
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
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
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
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
            # 更新版本号
            self._last_sync_version = self._get_db_version()
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
                # 更新版本号
                self._last_sync_version = self._get_db_version()
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
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
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
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
            try:
                return nx.shortest_path(self._graph, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
    
    def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> Dict:
        """获取与某节点相关的知识子图"""
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
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
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        with self._sync_and_lock():
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
        """存储一条经验 - 高级接口（原子操作，单次锁获取）"""
        node = KnowledgeNode(
            title=title, content=content,
            category=category,
            tags=tags or [],
            importance=importance
        )

        with self._lock:
            conn = self._get_conn()

            # 直接写入 SQLite（与 add_node 内部逻辑一致）
            conn.execute(
                """INSERT OR IGNORE INTO nodes
                   (id, title, content, category, tags, metadata, created_at, updated_at, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node.id, node.title, node.content, node.category,
                 json.dumps(node.tags, ensure_ascii=False),
                 json.dumps(node.metadata, ensure_ascii=False),
                 node.created_at, node.updated_at, node.importance)
            )

            # 直接写入 NetworkX（与 add_node 内部逻辑一致）
            self._graph.add_node(node.id, **asdict(node))

            # 建立关联边（直接操作，不调用 add_edge）
            if related_to:
                for rel_id in related_to:
                    if rel_id in self._graph:
                        edge = KnowledgeEdge(
                            source=node.id, target=rel_id,
                            relation="related_to"
                        )
                        try:
                            conn.execute(
                                """INSERT OR REPLACE INTO edges (source, target, relation, weight, metadata, created_at)
                                   VALUES (?, ?, ?, ?, ?, ?)""",
                                (edge.source, edge.target, edge.relation, edge.weight,
                                 json.dumps(edge.metadata, ensure_ascii=False), edge.created_at)
                            )
                            self._graph.add_edge(
                                edge.source, edge.target,
                                relation=edge.relation, weight=edge.weight,
                                metadata=edge.metadata, created_at=edge.created_at
                            )
                        except Exception as e:
                            logger.error(f"添加边失败: {e}")

            conn.commit()

            # 写入改变了数据库版本，更新同步版本号
            self._last_sync_version = self._get_db_version()

            logger.debug(f"存储经验: {node.title} [{node.category}]")
            return node.id
    
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
        # [修复1] 使用 _sync_and_lock 上下文管理器保护读操作
        # 注意：不能在锁内调用 self.search_nodes()，因为 search_nodes 也会
        # 调用 _sync_and_lock() 尝试获取同一个不可重入锁，会导致死锁。
        # 因此直接在锁内执行搜索逻辑。
        with self._sync_and_lock():
            query_lower = task_description.lower()

            def _do_search(category=None, limit=5):
                results = []
                for node_id, data in self._graph.nodes(data=True):
                    if category and data.get('category') != category:
                        continue
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

            experiences = _do_search(limit=5)
            methods = _do_search(category="method", limit=3)
            errors = _do_search(category="error", limit=3)
            
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
        # [修复4] 在 close() 中获取 self._lock，防止与其他线程的读写操作并发
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("知识图谱数据库连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
