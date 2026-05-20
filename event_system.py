# -*- coding: utf-8 -*-
"""
SoloAgent 事件系统 v2.0
第一性原理：系统 = 事件的流动，组件 = 事件的处理器
"""

import asyncio
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger("agent.events")


class EventType(Enum):
    """系统事件类型"""
    # 任务相关
    TASK_CREATED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_CANCELLED = auto()
    
    # 系统相关
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_PAUSE = auto()
    SYSTEM_RESUME = auto()
    SYSTEM_ERROR = auto()
    
    # 知识相关
    KNOWLEDGE_ADDED = auto()
    KNOWLEDGE_UPDATED = auto()
    KNOWLEDGE_QUERIED = auto()
    
    # 优化相关
    OPTIMIZATION_TRIGGERED = auto()
    OPTIMIZATION_COMPLETED = auto()
    
    # 插件相关
    PLUGIN_LOADED = auto()
    PLUGIN_UNLOADED = auto()
    PLUGIN_ERROR = auto()
    
    # 自定义
    CUSTOM = auto()


@dataclass
class Event:
    """事件对象"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    priority: int = 0  # 越高越优先处理
    
    def __post_init__(self):
        # Bug修复 #4: 在覆盖 self.type 之前先保存原始自定义类型字符串
        if isinstance(self.type, str):
            custom_type_str = self.type
            self.type = EventType.CUSTOM
            self.data["custom_type"] = custom_type_str


class EventHandler(ABC):
    """事件处理器接口"""
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """是否能处理该事件"""
        pass
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件"""
        pass


class EventBus:
    """
    事件总线 - 系统的神经系统
    支持同步和异步事件处理
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._async_handlers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()

        # 事件统计
        self._stats = {
            "published": 0,
            "handled": 0,
            "dropped": 0,
            "errors": 0
        }
    
    def start(self):
        """启动事件总线"""
        if self._running:
            return
        
        self._running = True
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        # 启动事件处理器
        self._processor_task = self._loop.create_task(self._process_events())
        logger.info("事件总线已启动")
    
    def stop(self):
        """停止事件总线"""
        if not self._running:
            return
        
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                self._loop.run_until_complete(self._processor_task)
            except asyncio.CancelledError:
                pass
        
        logger.info("事件总线已停止")
    
    async def _process_events(self):
        """事件处理循环"""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"事件处理异常: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
    
    async def _dispatch(self, event: Event):
        """分发事件到处理器"""
        handlers = []
        async_handlers = []
        
        with self._lock:
            handlers = self._handlers.get(event.type, []).copy()
            async_handlers = self._async_handlers.get(event.type, []).copy()
        
        # 同步处理
        for handler in handlers:
            try:
                handler(event)
                with self._stats_lock:
                    self._stats["handled"] += 1
            except Exception as e:
                logger.error(f"同步处理器异常: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
        
        # 异步处理
        if async_handlers:
            tasks = []
            for handler in async_handlers:
                tasks.append(self._run_async_handler(handler, event))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_async_handler(self, handler: Callable, event: Event):
        """运行异步处理器"""
        try:
            await handler(event)
            with self._stats_lock:
                self._stats["handled"] += 1
        except Exception as e:
            logger.error(f"异步处理器异常: {e}")
            with self._stats_lock:
                self._stats["errors"] += 1
    
    def subscribe(self, event_type: EventType, handler: Callable, async_mode: bool = False):
        """
        订阅事件
        handler: 同步或异步函数
        async_mode: 是否异步处理
        """
        with self._lock:
            target_list = self._async_handlers if async_mode else self._handlers
            if event_type not in target_list:
                target_list[event_type] = []
            if handler not in target_list[event_type]:  # 防止重复订阅
                target_list[event_type].append(handler)
        
        handler_name = getattr(handler, '__name__', handler.__class__.__name__)
        logger.debug(f"订阅 {event_type.name}: {handler_name}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """取消订阅"""
        with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
            if event_type in self._async_handlers and handler in self._async_handlers[event_type]:
                self._async_handlers[event_type].remove(handler)
    
    def publish(self, event: Event) -> bool:
        """
        发布事件
        同步接口，内部异步处理
        """
        if not self._running:
            with self._stats_lock:
                self._stats["dropped"] += 1
            return False
        
        try:
            # 使用 call_soon_threadsafe 确保线程安全
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    self._event_queue.put_nowait, 
                    event
                )
            else:
                self._event_queue.put_nowait(event)
            
            with self._stats_lock:
                self._stats["published"] += 1
            return True
        except Exception as e:
            logger.error(f"发布事件失败: {e}")
            with self._stats_lock:
                self._stats["dropped"] += 1
            return False
    
    def publish_sync(self, event: Event):
        """同步发布并等待处理完成（用于关键事件）"""
        # ★ Bug修复：publish_sync 是直接同步调用，不需要检查 _running 状态
        # 直接调用同步处理器
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
                with self._stats_lock:
                    self._stats["handled"] += 1
            except Exception as e:
                logger.error(f"同步处理器异常: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
    
    def get_stats(self) -> Dict:
        """获取事件统计"""
        with self._stats_lock:
            return dict(self._stats)


# ============================================================
# 预置事件处理器
# ============================================================

class LoggingEventHandler:
    """日志记录处理器"""
    
    def __init__(self, level: int = logging.INFO):
        self.level = level
    
    def __call__(self, event: Event):
        logger.log(self.level, f"[{event.type.name}] {event.data}")


class MetricsEventHandler:
    """指标收集处理器"""
    
    def __init__(self):
        self.metrics: Dict[str, List] = {}
    
    def __call__(self, event: Event):
        event_name = event.type.name
        if event_name not in self.metrics:
            self.metrics[event_name] = []

        self.metrics[event_name].append({
            "timestamp": event.timestamp,
            "data": event.data
        })

        if len(self.metrics[event_name]) > 1000:
            self.metrics[event_name] = self.metrics[event_name][-500:]
    
    def get_metrics(self, event_type: EventType = None) -> Dict:
        if event_type:
            return self.metrics.get(event_type.name, [])
        return dict(self.metrics)


class AuditEventHandler:
    """审计日志处理器 - 记录所有重要操作"""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        self._audit_logger = logging.getLogger("agent.audit")
        
        # Bug修复 #5: 在创建 FileHandler 前确保 logs/ 目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 配置审计日志处理器
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self._audit_logger.addHandler(handler)
        self._audit_logger.setLevel(logging.INFO)
    
    def __call__(self, event: Event):
        # 只记录重要事件
        important_types = {
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
            EventType.SYSTEM_ERROR,
            EventType.KNOWLEDGE_ADDED,
            EventType.PLUGIN_LOADED,
            EventType.PLUGIN_ERROR
        }
        
        if event.type in important_types:
            self._audit_logger.info(
                f"{event.type.name} | source={event.source} | "
                f"data={str(event.data)[:200]}"
            )


# ============================================================
# 事件驱动的调度器扩展
# ============================================================

class EventDrivenSchedulerMixin:
    """
    为调度器添加事件驱动能力
    使用混入模式，保持向后兼容
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """设置默认事件处理器"""
        # 日志处理器
        self.event_bus.subscribe(
            EventType.TASK_COMPLETED, 
            LoggingEventHandler(logging.INFO)
        )
        self.event_bus.subscribe(
            EventType.TASK_FAILED, 
            LoggingEventHandler(logging.ERROR)
        )
        
        # 审计处理器
        self.event_bus.subscribe(
            EventType.TASK_COMPLETED,
            AuditEventHandler()
        )
    
    def emit_task_created(self, task_id: str, task_name: str, task_type: str):
        """发送任务创建事件"""
        self.event_bus.publish(Event(
            type=EventType.TASK_CREATED,
            data={"task_id": task_id, "name": task_name, "type": task_type},
            source="scheduler"
        ))
    
    def emit_task_completed(self, task_id: str, task_name: str, result: Any, duration: float):
        """发送任务完成事件"""
        self.event_bus.publish(Event(
            type=EventType.TASK_COMPLETED,
            data={
                "task_id": task_id, 
                "name": task_name, 
                "result": str(result)[:500],
                "duration": duration
            },
            source="scheduler"
        ))
    
    def emit_task_failed(self, task_id: str, task_name: str, error: str):
        """发送任务失败事件"""
        self.event_bus.publish(Event(
            type=EventType.TASK_FAILED,
            data={"task_id": task_id, "name": task_name, "error": error[:500]},
            source="scheduler"
        ))
    
    def emit_system_start(self):
        """发送系统启动事件"""
        self.event_bus.publish(Event(
            type=EventType.SYSTEM_START,
            data={"timestamp": datetime.now().isoformat()},
            source="scheduler"
        ))
    
    def emit_system_stop(self):
        """发送系统停止事件"""
        self.event_bus.publish(Event(
            type=EventType.SYSTEM_STOP,
            data={"timestamp": datetime.now().isoformat()},
            source="scheduler"
        ))
