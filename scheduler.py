# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 保活调度器（核心引擎）
第一性原理：调度器 = 无限循环 + 任务队列 + 错误恢复 + 三线并行
三线并行: 执行任务 / 优化反思 / 知识积累 同时进行
"""

import time
import signal
import logging
import threading
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime

try:
    from .executor import Task, TaskExecutor, TaskStatus
    from .optimizer import Optimizer
    from .knowledge import KnowledgeGraph
except ImportError:
    from executor import Task, TaskExecutor, TaskStatus
    from optimizer import Optimizer
    from knowledge import KnowledgeGraph

logger = logging.getLogger("agent.scheduler")


class KeepAliveScheduler:
    """
    保活调度器 - Agent 的心脏
    第一性原理：
    1. 永不停歇：主循环 + 异常自动恢复
    2. 三线并行：执行、优化、存知识同步推进
    3. 自我修复：任何模块崩溃都能自动重启
    4. 无空闲时刻：没有任务时执行自我改进
    """
    
    def __init__(self, config, knowledge_graph: KnowledgeGraph, 
                 executor: TaskExecutor, optimizer: Optimizer):
        self.config = config
        self.kg = knowledge_graph
        self.executor = executor
        self.optimizer = optimizer
        
        # 任务队列（优先级队列用 deque + 排序模拟）
        self._task_queue: deque = deque()
        self._completed_tasks: List[Dict] = []
        self._completed_lock = threading.Lock()  # 保护已完成任务列表
        
        # 运行状态
        self._running = threading.Event()
        self._paused = threading.Event()
        self._cycle_count = 0
        self._start_time = None
        self._last_optimize_cycle = 0
        
        # 统计
        self._stats = {
            "total_cycles": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "optimizations": 0,
            "uptime_seconds": 0,
            "auto_recoveries": 0
        }
        
        # 线程锁
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # 注册信号处理
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            logger.warning("信号处理器注册失败（非主线程），请通过stop()方法优雅关闭")
    
    def _signal_handler(self, signum, frame):
        """优雅关闭"""
        logger.info(f"收到信号 {signum}，正在优雅关闭...")
        self.stop()
    
    # ---- 任务管理 ----
    
    def add_task(self, name: str, description: str = "", 
                 task_type: str = "general",
                 params: Optional[Dict] = None,
                 priority: int = 0) -> Task:
        """添加任务到队列"""
        task = Task(
            name=name, description=description,
            task_type=task_type,
            params=params or {},
            priority=priority
        )
        with self._lock:
            self._task_queue.append(task)
            # 按优先级排序（高优先级在前）
            self._task_queue = deque(
                sorted(self._task_queue, key=lambda t: -t.priority)
            )
        logger.info(f"📋 任务入队: {name} (类型: {task_type}, 优先级: {priority})")
        return task
    
    def add_tasks(self, tasks: List[Dict]) -> int:
        """批量添加任务"""
        count = 0
        with self._lock:
            for t in tasks:
                task = Task(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    task_type=t.get("task_type", "general"),
                    params=t.get("params", {}),
                    priority=t.get("priority", 0)
                )
                self._task_queue.append(task)
                count += 1
            # 批量添加后只排序一次
            self._task_queue = deque(
                sorted(self._task_queue, key=lambda t: -t.priority)
            )
        return count
    
    def get_queue_size(self) -> int:
        with self._lock:
            return len(self._task_queue)
    
    def get_next_task(self) -> Optional[Task]:
        """从队列取出下一个任务"""
        with self._lock:
            if self._task_queue:
                return self._task_queue.popleft()
        return None
    
    # ---- 核心循环 ----
    
    def start(self):
        """启动 Agent - 开始永不停歇的运转"""
        if self._running.is_set():
            logger.warning("Agent 已在运行中")
            return
        
        self._running.set()
        self._paused.clear()
        self._start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("🚀 极简全能持续运行 Agent 启动!")
        logger.info(f"   名称: {self.config.name}")
        logger.info(f"   版本: {self.config.version}")
        with self._lock:
            queue_size = len(self._task_queue)
        logger.info(f"   任务队列: {queue_size} 个任务")
        logger.info("=" * 60)
        
        try:
            self._main_loop()
        except Exception as e:
            logger.critical(f"主循环崩溃: {e}", exc_info=True)
            self._auto_recover(e)
    
    def stop(self):
        """停止 Agent - 确保所有子线程也停止"""
        self._running.clear()
        self._paused.clear()
        if self._start_time:
            with self._stats_lock:
                self._stats["uptime_seconds"] = time.time() - self._start_time
        # 等待一小段时间让当前执行中的任务完成
        logger.info("🛑 Agent 已停止")
    
    def pause(self):
        """暂停"""
        self._paused.set()
        logger.info("⏸ Agent 已暂停")
    
    def resume(self):
        """恢复"""
        self._paused.clear()
        logger.info("▶ Agent 已恢复运行")
    
    def _main_loop(self):
        """
        主循环 - Agent 的心跳
        修复：执行线紧密循环，优化线异步执行
        """
        while self._running.is_set():
            cycle_start = time.time()
            self._cycle_count += 1
            with self._stats_lock:
                self._stats["total_cycles"] += 1
            
            try:
                if self._paused.is_set():
                    logger.debug("暂停中，等待恢复...")
                    time.sleep(1)
                    continue
                
                # ====== 第一线：执行任务（紧密循环，不等待） ======
                self._execute_cycle()
                
                # ====== 第二线：优化反思（异步，不阻塞执行） ======
                self._optimize_cycle_async()
                
                # ====== 第三线：知识积累 ======
                self._knowledge_cycle()
                
            except Exception as e:
                logger.error(f"循环 #{self._cycle_count} 异常: {e}", exc_info=True)
                with self._stats_lock:
                    self._stats["auto_recoveries"] += 1
            
            # 只在无任务时才等待（有任务时立即处理下一个）
            if self.get_queue_size() == 0:
                interval = self.config.scheduler.cycle_interval
                if interval > 0:
                    time.sleep(interval)
        
        # 清理
        self._shutdown()
    
    def _execute_cycle(self):
        """执行线 - 从队列取任务并执行"""
        task = self.get_next_task()
        
        if task:
            # 有任务：执行它
            logger.info(f"━━━ 执行线 ━━━ 处理任务: {task.name}")
            result = self.executor.execute_with_retry(
                task,
                max_retries=self.config.scheduler.retry_max,
                retry_delay=self.config.scheduler.retry_delay,
                should_stop=lambda: not self._running.is_set()  # 传入停止信号
            )
            
            # 记录结果
            task_dict = result.to_dict()
            with self._completed_lock:
                self._completed_tasks.append(task_dict)
                if len(self._completed_tasks) > 1000:
                    self._completed_tasks = self._completed_tasks[-500:]
            
            if result.status == TaskStatus.SUCCESS.value:
                with self._stats_lock:
                    self._stats["tasks_completed"] += 1
            else:
                with self._stats_lock:
                    self._stats["tasks_failed"] += 1
            
            # 立即反思（优化线的一部分）
            self.optimizer.reflect_on_task(task_dict)
        else:
            # 无任务：执行自我改进（无空闲时刻！）
            self._idle_improve()
    
    def _optimize_cycle_async(self):
        """优化线 - 异步执行，不阻塞主循环"""
        # 每执行 N 个任务做一次深度反思
        if (self._cycle_count - self._last_optimize_cycle) >= self.config.scheduler.optimize_interval:
            self._last_optimize_cycle = self._cycle_count
            # 在后台线程执行，不阻塞执行线
            def _do_reflect():
                try:
                    logger.info("━━━ 优化线 ━━━ 执行深度反思...")
                    reflection = self.optimizer.deep_reflection()
                    # Bug修复: 后台线程修改 _stats 时使用 _stats_lock 保护，防止竞态条件
                    with self._stats_lock:
                        self._stats["optimizations"] += 1
                    logger.info(f"深度反思完成:\n{reflection[:500]}")
                except Exception as e:
                    logger.error(f"深度反思失败: {e}")
            threading.Thread(target=_do_reflect, daemon=True).start()
    
    def _knowledge_cycle(self):
        """知识线 - 确保知识图谱持续更新"""
        # 知识积累在执行线和优化线中已经自动进行
        # 这里做周期性的知识整理
        if self._cycle_count % 50 == 0 and self._cycle_count > 0 and self.kg:
            try:
                stats = self.kg.get_stats()
                logger.debug(
                    f"━━━ 知识线 ━━━ 图谱状态: "
                    f"{stats.get('total_nodes', 0)} 节点, {stats.get('total_edges', 0)} 边"
                )
            except Exception as e:
                logger.debug(f"知识线检查失败: {e}")
    
    def _idle_improve(self):
        """
        空闲改进 - 无任务时保持运转
        第一性原理：Agent 永远不闲着
        """
        # 每隔几轮空闲才执行一次改进（避免过于频繁）
        if self._cycle_count % 5 != 0:
            return
        
        logger.debug("━━━ 空闲改进 ━━━ 无待执行任务，进行自我提升...")
        
        # 轮流执行不同的自我改进任务
        idle_actions = [
            ("health_check", "自我健康检查"),
            ("knowledge_query", "知识库回顾"),
            ("self_improve", "自我改进分析"),
        ]
        
        action_type, action_name = idle_actions[self._cycle_count % len(idle_actions)]
        
        task = Task(
            name=action_name,
            description=action_name,
            task_type=action_type,
            params={"limit": 5}
        )
        
        try:
            result = self.executor.execute(task)
            if result.status == TaskStatus.SUCCESS.value:
                logger.debug(f"  空闲改进完成: {action_name}")
        except Exception as e:
            logger.debug(f"  空闲改进失败: {e}")
    
    def _auto_recover(self, error: Exception):
        """
        自动恢复 - 崩溃后自动重启
        Bug修复: 将递归调用 _main_loop 改为 while True 循环，防止栈溢出
        Bug修复: 在设置 _running=True 前检查是否已被外部停止（如信号处理调用了 stop()）
        """
        logger.warning("🔄 触发自动恢复机制...")
        with self._stats_lock:
            self._stats["auto_recoveries"] += 1

        # Bug修复: 使用 while True 循环代替递归调用 _main_loop，避免栈溢出
        while True:
            # 短暂等待
            time.sleep(3)

            # Bug修复: 恢复前检查是否已被外部停止（例如 stop() 已被调用）
            if not self._running.is_set():
                logger.info("  自动恢复取消: Agent 已被外部停止")
                return

            # 尝试重新初始化关键模块并重启主循环
            try:
                if self.kg:
                    logger.info("  重新连接知识图谱...")
                    # 知识图谱已有持久化，无需重新初始化

                logger.info("  恢复完成，重新启动主循环...")
                self._running.set()
                self._main_loop()
                # _main_loop 正常返回（_running 被设为 False），退出恢复循环
                return
            except Exception as e2:
                logger.critical(f"自动恢复失败: {e2}")
                # Bug修复: 循环重试而非递归，检查是否应继续恢复
                if not self._running.is_set():
                    logger.info("  自动恢复终止: Agent 已被外部停止")
                    return
                # 继续循环，下一次迭代将再次尝试恢复
    
    def _shutdown(self):
        """优雅关闭 - 不关闭共享的知识图谱（由创建者管理生命周期）"""
        logger.info("正在执行优雅关闭...")
        with self._stats_lock:
            self._stats["uptime_seconds"] = time.time() - self._start_time if self._start_time else 0
        
        # 保存最终统计
        if self.kg:
            try:
                self.kg.store_experience(
                    title=f"运行统计 #{self._cycle_count}",
                    content=f"循环次数: {self._cycle_count}, "
                           f"完成任务: {self._stats['tasks_completed']}, "
                           f"失败任务: {self._stats['tasks_failed']}, "
                           f"优化次数: {self._stats['optimizations']}, "
                           f"运行时间: {self._stats['uptime_seconds']:.0f}s",
                    category="experience",
                    tags=["stats", "runtime"],
                    importance=0.3
                )
            except Exception as e:
                logger.warning(f"保存运行统计失败: {e}")
        
        # 不关闭知识图谱 - 它是共享资源，由创建者负责关闭
        logger.info("优雅关闭完成")
    
    # ---- 状态查询 ----
    
    def get_stats_snapshot(self) -> Dict:
        """获取统计快照（线程安全）"""
        with self._stats_lock:
            return dict(self._stats)

    def get_status(self) -> Dict:
        """获取 Agent 当前状态"""
        uptime = time.time() - self._start_time if self._start_time else 0
        with self._completed_lock:
            completed_count = len(self._completed_tasks)
        return {
            "running": self._running.is_set(),
            "paused": self._paused.is_set(),
            "name": self.config.name,
            "version": self.config.version,
            "cycle_count": self._cycle_count,
            "queue_size": self.get_queue_size(),
            "completed_tasks": completed_count,
            "uptime_seconds": round(uptime, 1),
            "stats": self.get_stats_snapshot(),
            "executor_stats": self.executor.get_stats(),
            "optimizer_stats": self.optimizer.get_stats()
        }
    
    def get_recent_tasks(self, limit: int = 20) -> List[Dict]:
        """获取最近完成的任务"""
        limit = max(1, min(limit, 1000))
        with self._completed_lock:
            return list(self._completed_tasks[-limit:])
    
    def get_pending_tasks(self) -> List[Dict]:
        """获取待执行任务"""
        with self._lock:
            return [t.to_dict() for t in self._task_queue]
