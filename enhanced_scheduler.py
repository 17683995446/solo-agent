# -*- coding: utf-8 -*-
"""
SoloAgent 增强调度器 v2.0
第一性原理：调度 = 任务的有序执行，依赖 = 执行的约束条件
"""

import time
import uuid
import logging
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
import heapq

logger = logging.getLogger("agent.enhanced_scheduler")


class TaskState(Enum):
    """任务状态"""
    PENDING = auto()      # 等待依赖完成
    READY = auto()        # 依赖完成，可以执行
    RUNNING = auto()      # 执行中
    SUCCESS = auto()      # 成功完成
    FAILED = auto()       # 失败
    CANCELLED = auto()    # 取消
    TIMEOUT = auto()      # 超时


@dataclass
class Task:
    """增强版任务定义 - 兼容 executor.py 的 Task 接口"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    task_type: str = ""
    params: Dict = field(default_factory=dict)
    
    # 兼容 executor.Task 的字段
    description: str = ""  # executor.py 依赖此字段
    result: str = ""       # executor.py 写入此字段
    error: str = ""        # executor.py 写入此字段
    method: str = ""       # executor.py 写入此字段
    status: str = "pending"  # executor.py 读取此字段
    metadata: Dict = field(default_factory=dict)
    
    # 依赖管理
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID列表
    dependents: List[str] = field(default_factory=list)    # 依赖此任务的任务ID列表
    
    # 执行控制
    priority: int = 0
    timeout: Optional[int] = None  # 秒，None表示使用全局设置
    max_retries: int = 0
    retry_count: int = 0
    
    # 定时执行
    scheduled_at: Optional[datetime] = None  # 计划执行时间
    cron_expression: Optional[str] = None    # Cron表达式（周期性任务）
    
    # 状态（内部使用）
    state: TaskState = TaskState.PENDING
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = f"task_{self.id}"
    
    def to_dict(self) -> Dict:
        def fmt_dt(dt):
            if dt is None:
                return None
            if isinstance(dt, datetime):
                return dt.isoformat()
            return str(dt)  # 兼容字符串格式
        
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "params": self.params,
            "dependencies": self.dependencies,
            "state": self.state.name,
            "status": self.status,
            "priority": self.priority,
            "result": self.result[:500] if self.result else None,
            "error": self.error,
            "created_at": fmt_dt(self.created_at),
            "started_at": fmt_dt(self.started_at),
            "completed_at": fmt_dt(self.completed_at),
        }


class DAGBuilder:
    """
    DAG构建器
    用于构建复杂的工作流依赖图
    """
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._edges: List[tuple] = []  # (from_id, to_id)
    
    def add_task(self, task: Task) -> "DAGBuilder":
        """添加任务"""
        self._tasks[task.id] = task
        return self
    
    def add_edge(self, from_task_id: str, to_task_id: str) -> "DAGBuilder":
        """添加依赖边：from_task 必须在 to_task 之前完成"""
        self._edges.append((from_task_id, to_task_id))
        
        # 更新任务的依赖关系
        if to_task_id in self._tasks:
            if from_task_id not in self._tasks[to_task_id].dependencies:
                self._tasks[to_task_id].dependencies.append(from_task_id)
        
        if from_task_id in self._tasks:
            if to_task_id not in self._tasks[from_task_id].dependents:
                self._tasks[from_task_id].dependents.append(to_task_id)
        
        return self
    
    def chain(self, *task_ids: str) -> "DAGBuilder":
        """创建任务链：task1 -> task2 -> task3"""
        for i in range(len(task_ids) - 1):
            self.add_edge(task_ids[i], task_ids[i + 1])
        return self
    
    def parallel(self, *task_ids: str, then: str = None) -> "DAGBuilder":
        """
        创建并行任务：多个任务并行执行，然后执行 then 任务
        parallel(task1, task2, task3, then=task4)
        表示 task1, task2, task3 并行，都完成后执行 task4
        """
        if then:
            for task_id in task_ids:
                self.add_edge(task_id, then)
        return self
    
    def validate(self) -> tuple[bool, str]:
        """验证DAG无环"""
        # 拓扑排序检测环
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for from_id, to_id in self._edges:
            graph[from_id].append(to_id)
            in_degree[to_id] += 1
        
        # 找到所有入度为0的节点
        queue = deque([tid for tid in self._tasks if in_degree[tid] == 0])
        visited = 0
        
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if visited != len(self._tasks):
            return False, "DAG存在环，请检查依赖关系"
        
        return True, ""
    
    def build(self) -> List[Task]:
        """构建并返回任务列表"""
        valid, error = self.validate()
        if not valid:
            raise ValueError(error)
        return list(self._tasks.values())
    
    def get_execution_order(self) -> List[List[str]]:
        """
        获取按层级的执行顺序
        返回：[[第一层任务ID], [第二层任务ID], ...]
        """
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for from_id, to_id in self._edges:
            graph[from_id].append(to_id)
            in_degree[to_id] += 1
        
        result = []
        current_level = [tid for tid in self._tasks if in_degree[tid] == 0]
        
        while current_level:
            result.append(current_level)
            next_level = []
            for node in current_level:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)
            current_level = next_level
        
        return result


class EnhancedScheduler:
    """
    增强版任务调度器
    支持：DAG依赖、定时任务、超时控制、优先级队列
    """
    
    def __init__(self, executor, config=None):
        self.executor = executor
        self.config = config or {}
        
        # 任务存储
        self._tasks: Dict[str, Task] = {}
        self._ready_queue: List[tuple] = []  # (-priority, created_at, task_id) for heapq
        self._running_tasks: Dict[str, threading.Thread] = {}
        
        # 状态
        self._running = False
        self._paused = False
        self._shutdown_event = threading.Event()
        
        # 统计
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "total_cancelled": 0,
        }
        
        # 锁
        self._lock = threading.RLock()
        
        # 定时任务检查间隔
        self._schedule_check_interval = 0.1  # ★ 性能优化：从 1.0 降低到 0.1
    
    def submit(self, task: Task) -> str:
        """提交任务"""
        with self._lock:
            self._tasks[task.id] = task
            self._stats["total_submitted"] += 1
            
            # 检查是否立即可执行（无依赖或依赖已完成）
            if self._is_ready(task):
                task.state = TaskState.READY
                self._add_to_ready_queue(task)
                logger.debug(f"任务 {task.name} 已就绪")
            else:
                logger.debug(f"任务 {task.name} 等待依赖: {task.dependencies}")
            
            return task.id
    
    def submit_dag(self, tasks: List[Task]) -> List[str]:
        """提交DAG任务组"""
        task_ids = []
        with self._lock:
            for task in tasks:
                self._tasks[task.id] = task
                self._stats["total_submitted"] += 1
                task_ids.append(task.id)
            
            # 重新评估所有任务的就绪状态
            for task in tasks:
                if self._is_ready(task):
                    task.state = TaskState.READY
                    self._add_to_ready_queue(task)
        
        return task_ids
    
    def _is_ready(self, task: Task) -> bool:
        """检查任务是否就绪（所有依赖已完成）"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self._tasks:
                return False
            dep_task = self._tasks[dep_id]
            if dep_task.state != TaskState.SUCCESS:
                return False
        
        return True
    
    def _add_to_ready_queue(self, task: Task):
        """添加到就绪队列"""
        # 使用堆实现优先级队列
        heapq.heappush(self._ready_queue, (-task.priority, task.created_at.timestamp(), task.id))
    
    def _get_next_task(self) -> Optional[Task]:
        """获取下一个要执行的任务"""
        with self._lock:
            # 检查定时任务
            now = datetime.now()
            for task in self._tasks.values():
                if task.state == TaskState.PENDING and task.scheduled_at:
                    if now >= task.scheduled_at and self._is_ready(task):
                        task.state = TaskState.READY
                        self._add_to_ready_queue(task)
            
            # 从就绪队列取任务
            while self._ready_queue:
                _, _, task_id = heapq.heappop(self._ready_queue)
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    if task.state == TaskState.READY:
                        return task
            
            return None
    
    def _on_task_completed(self, task: Task, success: bool, result: Any, error: str = ""):
        """任务完成回调"""
        with self._lock:
            task.completed_at = datetime.now()
            
            if success:
                task.state = TaskState.SUCCESS
                task.status = "success"  # 同步 executor 兼容字段
                # [Bug修复] result=0, result=[], result=False 等 falsy 值不应被转为空字符串，
                # 只有 result=None 才表示无结果
                task.result = str(result)[:5000] if result is not None else ""
                self._stats["total_completed"] += 1
                logger.info(f"✓ 任务 {task.name} 完成")
                
                # 通知依赖此任务的其他任务
                for dependent_id in task.dependents:
                    if dependent_id in self._tasks:
                        dependent = self._tasks[dependent_id]
                        if dependent.state == TaskState.PENDING:
                            if self._is_ready(dependent):
                                dependent.state = TaskState.READY
                                self._add_to_ready_queue(dependent)
                                logger.debug(f"任务 {dependent.name} 依赖已满足，进入就绪队列")
            else:
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.state = TaskState.READY
                    task.status = "retrying"
                    self._add_to_ready_queue(task)
                    logger.info(f"↻ 任务 {task.name} 重试 ({task.retry_count}/{task.max_retries})")
                    return
                else:
                    task.state = TaskState.FAILED
                    task.status = "failed"
                    task.error = error
                    self._stats["total_failed"] += 1
                    logger.error(f"✗ 任务 {task.name} 失败: {error}")
                    
                    # ★ Bug修复：失败传播 - 取消所有下游依赖任务
                    self._propagate_failure(task)
    
    def _propagate_failure(self, failed_task: Task):
        """
        失败传播：递归取消所有下游依赖任务
        当一个任务最终失败时，所有依赖它的任务都不可能成功执行
        """
        # BFS遍历所有下游任务
        queue = list(failed_task.dependents)
        cancelled_count = 0
        
        while queue:
            dep_id = queue.pop(0)
            if dep_id not in self._tasks:
                continue
            dep = self._tasks[dep_id]
            
            # 只取消尚未开始执行的任务
            if dep.state in (TaskState.PENDING, TaskState.READY):
                dep.state = TaskState.CANCELLED
                dep.status = "cancelled"
                dep.error = f"上游依赖任务 [{failed_task.name}] 失败，自动取消"
                self._stats["total_cancelled"] += 1
                cancelled_count += 1
                logger.warning(f"⊘ 任务 {dep.name} 因上游失败被取消 (依赖: {failed_task.name})")
                # 继续传播到更下游
                queue.extend(dep.dependents)
            elif dep.state == TaskState.RUNNING:
                # 正在运行的任务标记为取消（线程无法强制终止）
                logger.warning(f"⚠ 任务 {dep.name} 正在运行中，上游已失败，标记为待取消")
        
        if cancelled_count > 0:
            logger.info(f"失败传播: {failed_task.name} → {cancelled_count} 个下游任务被取消")

    def _execute_task_locked(self, task: Task):
        """在锁内启动任务执行（仅注册线程，不等待）"""
        self._execute_task(task)

    def _execute_task(self, task: Task):
        """执行任务（非阻塞，立即返回）"""
        task.state = TaskState.RUNNING
        task.status = "running"  # 同步 executor 兼容字段
        task.started_at = datetime.now()

        # [Bug修复] timeout=0 是合法值（表示立即超时），不能用 `or` 判断，
        # 因为 0 是 falsy 值会被跳过，导致 timeout=0 被错误当作"无超时"
        timeout = task.timeout if task.timeout is not None else self.config.get("task_timeout", 300)

        # ★ Bug修复：使用Event协调超时，避免阻塞调度循环
        completion_event = threading.Event()

        def run_with_timeout():
            try:
                # 调用执行器
                self.executor.execute(task)

                # ★ Bug修复：检查是否已被超时标记，避免覆盖TIMEOUT状态
                with self._lock:
                    if task.state == TaskState.TIMEOUT:
                        logger.debug(f"任务 {task.name} 已超时，忽略延迟返回的结果")
                        return

                # 同步 executor 状态到增强调度器状态
                if task.status == "success":
                    self._on_task_completed(task, True, task.result)
                else:
                    self._on_task_completed(task, False, None, task.error)
            except Exception as e:
                with self._lock:
                    if task.state == TaskState.TIMEOUT:
                        return
                self._on_task_completed(task, False, None, str(e))
            finally:
                completion_event.set()
                with self._lock:
                    if task.id in self._running_tasks:
                        del self._running_tasks[task.id]

        thread = threading.Thread(target=run_with_timeout, name=f"Task-{task.name}")
        thread.daemon = True

        with self._lock:
            self._running_tasks[task.id] = thread

        thread.start()

        # ★ Bug修复：启动超时监控线程，不阻塞调度循环
        def timeout_watcher():
            if not completion_event.wait(timeout=timeout):
                # 超时 - 检查任务是否仍在运行
                with self._lock:
                    if task.state == TaskState.RUNNING:
                        task.state = TaskState.TIMEOUT
                        task.status = "timeout"
                        task.error = f"任务超时 ({timeout}s)"
                        self._stats["total_timeout"] += 1
                        logger.warning(f"⏱ 任务 {task.name} 超时 ({timeout}s)")
                        self._propagate_failure(task)

        watcher = threading.Thread(target=timeout_watcher, name=f"Timeout-{task.name}", daemon=True)
        watcher.start()
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        logger.info("增强调度器已启动")
        
        # 主调度循环
        while not self._shutdown_event.is_set():
            if self._paused:
                time.sleep(0.1)
                continue

            # 获取并发限制
            max_concurrent = self.config.get("max_concurrent_tasks", 3)

            # ★ Bug修复：在锁内检查并发数并启动任务，避免竞态
            with self._lock:
                if len(self._running_tasks) < max_concurrent:
                    task = self._get_next_task()
                    if task:
                        self._execute_task_locked(task)
                    else:
                        # 没有就绪任务，释放锁后等待
                        pass
                else:
                    task = None

            if task is None:
                # ★ Bug修复：_running=False 且无任务时才退出，确保 ready_queue 清空
                if not self._running:
                    with self._lock:
                        if not self._ready_queue and not self._running_tasks:
                            break
                # ★ 性能优化：有运行中任务时用短间隔，空闲时用长间隔
                with self._lock:
                    has_running = len(self._running_tasks) > 0
                    has_pending = len(self._ready_queue) > 0 or any(
                        t.state == TaskState.PENDING for t in self._tasks.values()
                    )
                if has_running:
                    time.sleep(0.01)  # 有任务运行中，快速检查
                elif has_pending:
                    time.sleep(self._schedule_check_interval)  # 有待处理任务
                else:
                    time.sleep(self._schedule_check_interval)  # 真正空闲
            # 如果启动了任务，立即回到循环顶部检查是否还能启动更多
        
        logger.info("增强调度器已停止")
    
    def stop(self, wait: bool = True, timeout: float = 30.0):
        """停止调度器"""
        self._running = False
        self._shutdown_event.set()

        if wait:
            # ★ Bug修复：等待所有非终态任务完成或就绪队列清空
            start = time.time()
            while time.time() - start < timeout:
                with self._lock:
                    running = len(self._running_tasks)
                    ready = len(self._ready_queue)
                    if running == 0 and ready == 0:
                        break
                time.sleep(0.1)
    
    def pause(self):
        """暂停调度"""
        self._paused = True
        logger.info("调度器已暂停")
    
    def resume(self):
        """恢复调度"""
        self._paused = False
        logger.info("调度器已恢复")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            if task.state in [TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED]:
                return False
            
            task.state = TaskState.CANCELLED
            self._stats["total_cancelled"] += 1
            logger.info(f"⊘ 任务 {task.name} 已取消")

            # [Bug修复] cancel_task 只修改 state 但未从 _ready_queue 移除被取消的任务，
            # 导致队列积累无效条目。重建就绪队列，过滤掉已取消/非就绪状态的任务。
            self._ready_queue = [
                entry for entry in self._ready_queue
                if entry[2] != task_id  # entry = (-priority, created_at, task_id)
            ]

            return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self.get_task(task_id)
        return task.to_dict() if task else None
    
    def list_tasks(self, state: TaskState = None) -> List[Dict]:
        """列出任务"""
        with self._lock:
            tasks = self._tasks.values()
            if state:
                tasks = [t for t in tasks if t.state == state]
            return [t.to_dict() for t in tasks]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            return dict(self._stats)


# ============================================================
# 便捷函数
# ============================================================

def create_simple_task(name: str, task_type: str, params: Dict, priority: int = 0) -> Task:
    """创建简单任务"""
    return Task(name=name, task_type=task_type, params=params, priority=priority)


def create_scheduled_task(name: str, task_type: str, params: Dict, 
                          scheduled_at: datetime, priority: int = 0) -> Task:
    """创建定时任务"""
    return Task(
        name=name, 
        task_type=task_type, 
        params=params, 
        scheduled_at=scheduled_at,
        priority=priority
    )
