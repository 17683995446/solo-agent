# -*- coding: utf-8 -*-
"""
SoloAgent 配置中心 v2.0
第一性原理：配置 = 系统的可变性，热更新 = 不重启的改变
"""

import os
import json
import yaml
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger("agent.config_center")


@dataclass
class ConfigSchema:
    """配置项 schema"""
    key: str
    type: str  # string, int, float, bool, list, dict
    default: Any = None
    description: str = ""
    validator: Optional[Callable] = None
    hot_reload: bool = True  # 是否支持热更新


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_type(value: Any, expected_type: str) -> tuple[bool, str]:
        """验证类型"""
        type_map = {
            "string": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict
        }
        
        if expected_type not in type_map:
            return False, f"未知类型: {expected_type}"
        
        expected = type_map[expected_type]
        if not isinstance(value, expected):
            return False, f"类型错误: 期望 {expected_type}, 实际 {type(value).__name__}"
        
        return True, ""
    
    @staticmethod
    def validate_range(value: Any, min_val: Any = None, max_val: Any = None) -> tuple[bool, str]:
        """验证范围"""
        if min_val is not None and value < min_val:
            return False, f"值 {value} 小于最小值 {min_val}"
        if max_val is not None and value > max_val:
            return False, f"值 {value} 大于最大值 {max_val}"
        return True, ""
    
    @staticmethod
    def validate_enum(value: Any, allowed: List[Any]) -> tuple[bool, str]:
        """验证枚举值"""
        if value not in allowed:
            return False, f"值 {value} 不在允许列表 {allowed} 中"
        return True, ""


class ConfigChangeEvent:
    """配置变更事件"""
    def __init__(self, key: str, old_value: Any, new_value: Any, source: str):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.source = source
        self.timestamp = time.time()


class ConfigCenter:
    """
    配置中心
    支持：文件加载、环境变量覆盖、热更新、变更监听
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._schema: Dict[str, ConfigSchema] = {}
        self._listeners: Dict[str, List[Callable]] = {}
        self._global_listeners: List[Callable] = []
        self._lock = threading.RLock()
        self._last_modified = 0
        self._observer: Optional[Observer] = None
        self._watching = False
        
        # 初始化 schema
        self._init_default_schema()
    
    def _init_default_schema(self):
        """初始化默认配置 schema"""
        default_schemas = [
            ConfigSchema("name", "string", "SoloAgent", "Agent名称"),
            ConfigSchema("version", "string", "2.0.0", "版本号", hot_reload=False),
            ConfigSchema("log_level", "string", "INFO", "日志级别", 
                        validator=lambda v: ConfigValidator.validate_enum(v, ["DEBUG", "INFO", "WARNING", "ERROR"])),
            ConfigSchema("scheduler.cycle_interval", "float", 1.0, "调度循环间隔(秒)",
                        validator=lambda v: ConfigValidator.validate_range(v, 0.1, 60.0)),
            ConfigSchema("scheduler.max_concurrent", "int", 3, "最大并发任务数",
                        validator=lambda v: ConfigValidator.validate_range(v, 1, 100)),
            ConfigSchema("llm.timeout", "int", 60, "LLM请求超时(秒)",
                        validator=lambda v: ConfigValidator.validate_range(v, 5, 300)),
            ConfigSchema("llm.temperature", "float", 0.7, "LLM温度参数",
                        validator=lambda v: ConfigValidator.validate_range(v, 0.0, 2.0)),
            ConfigSchema("features.plugin_system", "bool", True, "启用插件系统"),
            ConfigSchema("features.event_system", "bool", True, "启用事件系统"),
            ConfigSchema("features.hot_reload", "bool", True, "启用配置热更新"),
        ]
        
        for schema in default_schemas:
            self._schema[schema.key] = schema
    
    def load(self) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认值")
                self._apply_defaults()
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    loaded = yaml.safe_load(f) or {}
                else:
                    loaded = json.load(f)
            
            # 验证并应用配置
            self._validate_and_apply(loaded)
            
            self._last_modified = os.path.getmtime(self.config_path)
            logger.info(f"配置加载成功: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._apply_defaults()
            return False
    
    def _validate_and_apply(self, config: Dict):
        """验证并应用配置"""
        with self._lock:
            # 扁平化配置（支持嵌套 key 如 "scheduler.cycle_interval"）
            flat_config = self._flatten(config)
            
            for key, value in flat_config.items():
                if key in self._schema:
                    schema = self._schema[key]
                    
                    # 类型验证
                    valid, error = ConfigValidator.validate_type(value, schema.type)
                    if not valid:
                        logger.warning(f"配置项 {key} 验证失败: {error}，使用默认值")
                        value = schema.default
                    
                    # 自定义验证
                    if schema.validator:
                        valid, error = schema.validator(value)
                        if not valid:
                            logger.warning(f"配置项 {key} 验证失败: {error}，使用默认值")
                            value = schema.default
                
                # 应用配置
                old_value = self._get_nested(self._config, key)
                self._set_nested(self._config, key, value)
                
                # 触发变更事件
                if old_value != value:
                    self._notify_change(key, old_value, value, "file")
    
    def _flatten(self, config: Dict, parent_key: str = "") -> Dict[str, Any]:
        """将嵌套配置扁平化"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_nested(self, config: Dict, key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value
    
    def _set_nested(self, config: Dict, key: str, value: Any):
        """设置嵌套配置值"""
        keys = key.split(".")
        target = config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    
    def _apply_defaults(self):
        """应用默认配置"""
        with self._lock:
            for key, schema in self._schema.items():
                self._set_nested(self._config, key, schema.default)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            value = self._get_nested(self._config, key)
            return value if value is not None else default
    
    def set(self, key: str, value: Any, source: str = "api") -> bool:
        """
        设置配置值（支持热更新）
        如果配置项不支持热更新，返回 False
        """
        with self._lock:
            # 检查 schema
            if key in self._schema:
                schema = self._schema[key]
                if not schema.hot_reload:
                    logger.warning(f"配置项 {key} 不支持热更新")
                    return False
                
                # 类型验证
                valid, error = ConfigValidator.validate_type(value, schema.type)
                if not valid:
                    logger.error(f"配置项 {key} 验证失败: {error}")
                    return False
                
                # 自定义验证（如范围检查）
                if schema.validator:
                    valid, error = schema.validator(value)
                    if not valid:
                        logger.warning(f"配置项 {key} 验证失败: {error}")
                        return False
            
            old_value = self._get_nested(self._config, key)
            self._set_nested(self._config, key, value)
            
            # 触发变更事件
            if old_value != value:
                self._notify_change(key, old_value, value, source)
            
            return True
    
    def register_listener(self, key: str, callback: Callable):
        """注册特定配置项的变更监听器"""
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(callback)
    
    def register_global_listener(self, callback: Callable):
        """注册全局配置变更监听器"""
        self._global_listeners.append(callback)
    
    def _notify_change(self, key: str, old_value: Any, new_value: Any, source: str):
        """通知配置变更"""
        event = ConfigChangeEvent(key, old_value, new_value, source)
        
        # 通知特定监听器
        if key in self._listeners:
            for callback in self._listeners[key]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"配置变更监听器异常: {e}")
        
        # 通知全局监听器
        for callback in self._global_listeners:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"全局配置变更监听器异常: {e}")
        
        logger.info(f"配置变更: {key} = {new_value} (来源: {source})")
    
    def start_hot_reload(self):
        """启动配置文件热更新监听"""
        if self._watching:
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_center):
                self.config_center = config_center
                self.last_reload = 0
            
            def on_modified(self, event):
                if event.src_path == self.config_center.config_path:
                    # 防抖：避免重复加载
                    now = time.time()
                    if now - self.last_reload < 1.0:
                        return
                    self.last_reload = now
                    
                    logger.info("配置文件变更，重新加载...")
                    self.config_center.load()
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        watch_dir = os.path.dirname(os.path.abspath(self.config_path)) or "."
        self._observer.schedule(handler, watch_dir, recursive=False)
        self._observer.start()
        self._watching = True
        
        logger.info(f"配置热更新监听已启动: {self.config_path}")
    
    def stop_hot_reload(self):
        """停止热更新监听"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._watching = False
            logger.info("配置热更新监听已停止")
    
    def save(self) -> bool:
        """保存配置到文件"""
        try:
            with self._lock:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
                    else:
                        json.dump(self._config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"配置已保存: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def get_all(self) -> Dict:
        """获取所有配置"""
        with self._lock:
            return dict(self._config)
    
    def get_schema(self, key: str = None) -> Optional[ConfigSchema]:
        """获取配置 schema"""
        if key:
            return self._schema.get(key)
        return dict(self._schema)


# ============================================================
# 与现有系统的集成
# ============================================================

class ConfigurableMixin:
    """
    可配置混入类
    为组件添加配置中心支持
    """
    
    def __init__(self):
        self._config_center: Optional[ConfigCenter] = None
        self._config_listeners: List[str] = []
    
    def attach_config_center(self, config_center: ConfigCenter):
        """附加配置中心"""
        self._config_center = config_center
    
    def config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if self._config_center:
            return self._config_center.get(key, default)
        return default
    
    def watch_config(self, key: str, callback: Callable):
        """监听配置变更"""
        if self._config_center:
            self._config_center.register_listener(key, callback)
            self._config_listeners.append(key)
    
    def unwatch_all(self):
        """取消所有配置监听"""
        # 简化实现：实际应该取消注册
        self._config_listeners.clear()


# 全局配置中心实例
_global_config_center: Optional[ConfigCenter] = None


def get_config_center() -> Optional[ConfigCenter]:
    """获取全局配置中心"""
    return _global_config_center


def init_config_center(config_path: str = "config.yaml") -> ConfigCenter:
    """初始化全局配置中心"""
    global _global_config_center
    _global_config_center = ConfigCenter(config_path)
    _global_config_center.load()
    return _global_config_center
