# -*- coding: utf-8 -*-
"""
SoloAgent 配置中心 v2.0
第一性原理：配置 = 系统的可变性，热更新 = 不重启的改变
"""

import os
import json
import time
import logging
import tempfile
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from filelock import FileLock

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
        # [R17-修复] 添加进程级文件锁，解决多进程并发访问问题
        self._file_lock_path = config_path + ".lock"
        self._file_lock = FileLock(self._file_lock_path, timeout=30)
        
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
        """加载配置文件（进程安全）"""
        # [R17-修复] 使用文件锁保护多进程并发读取
        with self._file_lock:
            try:
                if not os.path.exists(self.config_path):
                    logger.warning(f"配置文件不存在: {self.config_path}，使用默认值")
                    self._apply_defaults()
                    return False
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        import yaml
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
    
    def _flatten_config(self) -> Dict[str, Any]:
        """扁平化当前配置"""
        return self._flatten(self._config)

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
        """获取配置值，通过检查 key 是否存在来区分 None 值和缺失 key"""
        with self._lock:
            flat = self._flatten_config()
            if key in flat:
                return flat[key]
            return default
    
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
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典，递归合并嵌套的字典值"""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def set_and_save(self, key: str, value: Any, source: str = "api") -> bool:
        """
        原子性设置配置并保存到文件（进程安全）
        [R17-修复] 解决多进程配置更新丢失问题
        """
        # 使用文件锁确保整个操作原子性
        with self._file_lock:
            # 先重新加载配置，确保基于最新状态
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                            import yaml
                            latest_config = yaml.safe_load(f) or {}
                        else:
                            latest_config = json.load(f)
                    # 深度合并最新配置，避免浅合并覆盖嵌套配置
                    with self._lock:
                        self._config = self._deep_merge(self._config, latest_config)
                except Exception as e:
                    logger.warning(f"重新加载配置失败: {e}")
            
            # 设置新值
            if not self.set(key, value, source):
                return False
            
            # 立即保存
            return self.save()
    
    def register_listener(self, key: str, callback: Callable):
        """注册特定配置项的变更监听器"""
        with self._lock:
            if key not in self._listeners:
                self._listeners[key] = []
            self._listeners[key].append(callback)
    
    def register_global_listener(self, callback: Callable):
        """注册全局配置变更监听器"""
        with self._lock:
            self._global_listeners.append(callback)
    
    def _notify_change(self, key: str, old_value: Any, new_value: Any, source: str):
        """通知配置变更。注意：此方法必须在 _lock 保护下调用，调用者负责持锁。"""
        event = ConfigChangeEvent(key, old_value, new_value, source)
        
        # 在锁内收集需要通知的回调（调用者已持有 _lock，此处不再重复获取）
        callbacks = list(self._listeners.get(key, []))
        global_callbacks = list(self._global_listeners)
        
        # 释放锁后再调用回调
        # 注意：由于使用 RLock，同一线程可重入，但回调中不应依赖锁
        # 为避免回调中修改 listeners 导致问题，先释放锁再调用回调
        self._lock.release()
        try:
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"配置变更监听器异常: {e}")
            for callback in global_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"全局配置监听器异常: {e}")
        finally:
            self._lock.acquire()
        
        logger.info(f"配置变更: {key} = {new_value} (来源: {source})")
    
    def start_hot_reload(self):
        """启动配置文件热更新监听"""
        if self._watching:
            return
        
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_center):
                self.config_center = config_center
                self.last_reload = 0
                self._config_path = os.path.abspath(config_center.config_path)

            def on_modified(self, event):
                if os.path.abspath(event.src_path) == self._config_path:
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
        """保存配置到文件（原子写入+进程安全）"""
        # [R17-修复] 使用文件锁保护多进程并发写入
        with self._file_lock:
            try:
                with self._lock:
                    config_copy = dict(self._config)

                # [R17-修复] 使用唯一临时文件名，避免多进程冲突
                import uuid as uuid_mod
                tmp_path = self.config_path + f'.tmp.{uuid_mod.uuid4().hex}'
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        import yaml
                        yaml.dump(config_copy, f, allow_unicode=True, default_flow_style=False)
                    else:
                        json.dump(config_copy, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.config_path)

                logger.info(f"配置已保存: {self.config_path}")
                return True
            except Exception as e:
                logger.error(f"配置保存失败: {e}")
                # 清理可能残留的临时文件
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
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

    def register_schema(self, key: str, schema: ConfigSchema):
        """注册配置 schema"""
        with self._lock:
            self._schema[key] = schema
            logger.info(f"已注册配置 schema: {key}")


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
_global_cc_lock = threading.Lock()


def get_config_center() -> Optional[ConfigCenter]:
    """获取全局配置中心"""
    return _global_config_center


def init_config_center(config_path: str = "config.yaml") -> ConfigCenter:
    """初始化全局配置中心"""
    global _global_config_center
    with _global_cc_lock:
        if _global_config_center is None:
            _global_config_center = ConfigCenter(config_path)
            _global_config_center.load()
        return _global_config_center
