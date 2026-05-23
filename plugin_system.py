# -*- coding: utf-8 -*-
"""
SoloAgent 插件系统 v2.0
第一性原理：功能 = 插件的集合，核心 = 插件的容器
"""

import os
import sys
import json
import logging
import threading
import importlib
import importlib.util
import types  # ★ 新增：用于热更新时创建模块
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import traceback

logger = logging.getLogger("agent.plugins")


@dataclass
class PluginInfo:
    """插件元数据"""
    name: str
    version: str
    description: str
    author: str
    entry_point: str  # 入口类名
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    enabled: bool = True


class PluginInterface(ABC):
    """
    插件接口 - 所有插件必须实现
    第一性原理：插件 = 输入处理函数 + 生命周期管理
    """
    
    def __init__(self, config: Dict = None):
        # Bug修复 #3: config={} 被丢弃 — 使用 is not None 而非 or，保留空字典
        self.config = config if config is not None else {}
        self._initialized = False
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """返回插件元数据"""
        pass
    
    @abstractmethod
    def execute(self, params: Dict, context: str = "") -> Any:
        """
        执行插件功能
        params: 任务参数
        context: 知识图谱上下文
        return: 执行结果
        """
        pass
    
    def initialize(self) -> bool:
        """初始化插件 - 子类可覆盖"""
        self._initialized = True
        return True
    
    def shutdown(self):
        """关闭插件 - 子类可覆盖"""
        self._initialized = False
    
    def get_config(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    def validate_params(self, params: Dict) -> tuple[bool, str]:
        """
        验证参数 - 子类可覆盖
        return: (是否有效, 错误信息)
        """
        return True, ""


class PluginManager:
    """
    插件管理器
    职责：发现、加载、管理插件生命周期
    """
    
    def __init__(self, plugin_dirs: List[str] = None):
        self._lock = threading.RLock()
        self.plugin_dirs = plugin_dirs if plugin_dirs is not None else ["plugins", "~/.solo_agent/plugins"]
        self._plugins: Dict[str, PluginInterface] = {}  # name -> instance
        self._handlers: Dict[str, Callable] = {}  # task_type -> execute function
        self._metadata: Dict[str, PluginInfo] = {}  # name -> info
        self._module_names: Dict[str, str] = {}  # name -> module_name (for hot reload cleanup)
        
        # 展开用户目录
        self.plugin_dirs = [
            os.path.expanduser(d) for d in self.plugin_dirs
        ]
    
    def discover_plugins(self) -> List[Path]:
        """发现所有可用插件"""
        with self._lock:
            discovered = []
            for plugin_dir in self.plugin_dirs:
                if not os.path.exists(plugin_dir):
                    continue
                for item in Path(plugin_dir).iterdir():
                    if item.is_dir() and (item / "plugin.json").exists():
                        discovered.append(item)
                    elif item.suffix == ".py" and item.stem != "__init__":
                        discovered.append(item)
            return discovered
    
    def load_plugin(self, plugin_path) -> Optional[str]:
        """
        加载单个插件
        plugin_path: Path对象或字符串路径
        return: 插件名称或None
        """
        try:
            # ★ Bug修复：兼容字符串路径
            plugin_path = Path(plugin_path) if not isinstance(plugin_path, Path) else plugin_path
            if plugin_path.is_dir():
                return self._load_package_plugin(plugin_path)
            else:
                return self._load_single_file_plugin(plugin_path)
        except Exception as e:
            logger.error(f"加载插件失败 {plugin_path}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _load_package_plugin(self, plugin_dir: Path) -> Optional[str]:
        """加载包格式插件（目录 + plugin.json）"""
        manifest_path = plugin_dir / "plugin.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        info = PluginInfo(**manifest)
        
        with self._lock:
            if info.name in self._plugins:
                logger.warning(f"插件 {info.name} 已存在，跳过")
                return None
        
        # 动态导入
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            init_file = plugin_dir / f"{info.entry_point}.py"
        
        spec = importlib.util.spec_from_file_location(
            f"plugin_{info.name}", init_file
        )
        # Bug修复 #2: 入口文件不存在时 spec 为 None，需要防御性检查
        if spec is None:
            logger.error(f"无法为插件 {info.name} 创建模块规格，入口文件: {init_file}")
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        
        # Bug修复 #6: 模块加载失败后清理sys.modules
        try:
            spec.loader.exec_module(module)
            
            # 实例化插件类
            plugin_class = getattr(module, info.entry_point)
        except Exception as e:
            # 加载失败时清理sys.modules中已注册的模块
            if spec.name in sys.modules:
                del sys.modules[spec.name]
            logger.error(f"插件 {info.name} 模块加载失败: {e}")
            logger.debug(traceback.format_exc())
            return None
        instance = plugin_class()
        
        if not instance.initialize():
            logger.error(f"插件 {info.name} 初始化失败")
            # 初始化失败后清理 sys.modules
            sys.modules.pop(spec.name, None)
            return None
        
        with self._lock:
            self._plugins[info.name] = instance
            self._metadata[info.name] = info
            self._module_names[info.name] = spec.name
        
        # 注册任务处理器
        self._register_handlers(instance)
        
        logger.info(f"✓ 插件加载成功: {info.name} v{info.version}")
        return info.name
    
    def _load_single_file_plugin(self, plugin_file: Path) -> Optional[str]:
        """加载单文件插件（简化格式）"""
        module_name = f"plugin_{plugin_file.stem}"
        
        # ★ Bug修复：使用exec()直接执行代码，避免importlib的文件缓存问题
        # 这对于热更新场景至关重要
        module = types.ModuleType(module_name)
        module.__file__ = str(plugin_file)
        sys.modules[module_name] = module
        
        # 读取并执行文件内容（耗时操作，不在锁内执行）
        # 安全警告：exec()执行外部代码存在风险，请确保插件来源可信
        logger.warning(f"正在加载外部插件: {plugin_file}，请确保来源可信")
        try:
            with open(plugin_file, 'r', encoding='utf-8') as f:
                code = f.read()
            exec(compile(code, str(plugin_file), 'exec'), module.__dict__)
        except Exception as e:
            # exec() 失败后清理 sys.modules
            sys.modules.pop(module_name, None)
            logger.error(f"插件文件 {plugin_file} 执行失败: {e}")
            logger.debug(traceback.format_exc())
            return None
        
        # 查找插件类（继承PluginInterface）
        plugin_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, PluginInterface) and 
                obj is not PluginInterface):
                plugin_class = obj
                break
        
        if not plugin_class:
            logger.warning(f"在 {plugin_file} 中未找到插件类")
            return None
        
        instance = plugin_class()
        info = instance.get_info()
        
        with self._lock:
            if info.name in self._plugins:
                logger.warning(f"插件 {info.name} 已存在，跳过")
                return None
        
        if not instance.initialize():
            logger.error(f"插件 {info.name} 初始化失败")
            return None
        
        with self._lock:
            self._plugins[info.name] = instance
            self._metadata[info.name] = info
            self._module_names[info.name] = module_name  # ★ 存储模块名用于热更新清理
        
        # 注册任务处理器
        self._register_handlers(instance)
        
        logger.info(f"✓ 插件加载成功: {info.name} v{info.version}")
        return info.name
    
    def _register_handlers(self, plugin: PluginInterface):
        """注册插件的任务处理器"""
        info = plugin.get_info()
        with self._lock:
            # 使用插件名作为任务类型
            self._handlers[info.name] = plugin.execute
            
            # 插件可以注册多个别名
            aliases = plugin.config.get("aliases", [])
            for alias in aliases:
                self._handlers[alias] = plugin.execute
    
    def unload_plugin(self, name: str) -> bool:
        """卸载插件"""
        with self._lock:
            if name not in self._plugins:
                return False
            
            plugin = self._plugins[name]
            info = plugin.get_info()
            
            # 注销处理器
            handlers_to_remove = [info.name] + plugin.config.get("aliases", [])
            for handler_name in handlers_to_remove:
                if handler_name in self._handlers:
                    del self._handlers[handler_name]
            
            # 关闭插件
            plugin.shutdown()
            
            # ★ Bug修复：清理sys.modules中的模块缓存，确保热更新时重新加载
            # 使用实际存储的模块名（单文件插件是 plugin_{stem}，包插件是 plugin_{name}）
            module_name = self._module_names.get(name)
            if module_name and module_name in sys.modules:
                sys.modules.pop(module_name, None)
            # 同时清理可能的变体
            # Bug修复 #14: 使用pop替代del避免遍历删除时的竞态条件
            for key in list(sys.modules.keys()):
                if key == module_name or (module_name and key.startswith(module_name + "_")):
                    sys.modules.pop(key, None)
            
            # 清理
            del self._plugins[name]
            del self._metadata[name]
            if name in self._module_names:
                del self._module_names[name]
        
        logger.info(f"✓ 插件已卸载: {name}")
        return True
    
    def reload_plugin(self, name: str) -> bool:
        """热重载插件"""
        if name not in self._plugins:
            return False
        
        # Bug修复 #15: 卸载前保存插件路径信息，防止重新加载失败后插件丢失
        plugin = self._plugins[name]
        info = plugin.get_info()
        saved_plugin_path = None
        
        # 尝试找到插件的原始路径
        for plugin_path in self.discover_plugins():
            plugin_path = Path(plugin_path) if not isinstance(plugin_path, Path) else plugin_path
            if plugin_path.is_dir():
                manifest_path = plugin_path / "plugin.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        if manifest.get('name') == name:
                            saved_plugin_path = plugin_path
                            break
                    except:
                        pass
            else:
                # 单文件插件：检查模块名是否匹配
                if f"plugin_{plugin_path.stem}" == self._module_names.get(name):
                    saved_plugin_path = plugin_path
                    break
        
        # 找到插件路径
        # 简化实现：先卸载再重新发现加载
        self.unload_plugin(name)
        
        for plugin_path in self.discover_plugins():
            loaded_name = self.load_plugin(plugin_path)
            if loaded_name == name:
                return True
        
        # Bug修复 #15: 重新加载失败时记录错误
        logger.error(f"插件 {name} 热重载失败：卸载后无法重新加载")
        if saved_plugin_path:
            logger.error(f"插件原始路径: {saved_plugin_path}，请手动重新加载")
        
        return False
    
    def get_handler(self, task_type: str) -> Optional[Callable]:
        """获取任务处理器"""
        with self._lock:
            return self._handlers.get(task_type)
    
    def list_plugins(self) -> List[Dict]:
        """列出所有插件"""
        with self._lock:
            return [
                {
                    "name": name,
                    "version": info.version,
                    "description": info.description,
                    "enabled": info.enabled,
                    "handlers": [name] + self._plugins[name].config.get("aliases", [])
                }
                for name, info in self._metadata.items()
            ]
    
    def load_all(self):
        """加载所有发现的插件"""
        discovered = self.discover_plugins()
        logger.info(f"发现 {len(discovered)} 个插件")
        
        loaded = 0
        for plugin_path in discovered:
            if self.load_plugin(plugin_path):
                loaded += 1
        
        logger.info(f"成功加载 {loaded}/{len(discovered)} 个插件")
        return loaded


# ============================================================
# 示例插件：HTTP请求插件
# ============================================================

class HttpRequestPlugin(PluginInterface):
    """HTTP请求插件 - 示例"""
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="http_request",
            version="1.0.0",
            description="发送HTTP请求并返回响应",
            author="SoloAgent",
            entry_point="HttpRequestPlugin",
            config_schema={
                "timeout": {"type": "number", "default": 30},
                "max_retries": {"type": "number", "default": 3}
            }
        )
    
    def execute(self, params: Dict, context: str = "") -> Any:
        import requests
        
        url = params.get("url")
        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        data = params.get("data")
        timeout = self.get_config("timeout", 30)
        
        if not url:
            return {"error": "缺少URL参数"}
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=timeout
            )
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:5000]  # 限制长度
            }
        except Exception as e:
            return {"error": str(e)}
    
    def validate_params(self, params: Dict) -> tuple[bool, str]:
        if "url" not in params:
            return False, "缺少必需参数: url"
        return True, ""


# ============================================================
# 示例插件：数据库操作插件
# ============================================================

class DatabasePlugin(PluginInterface):
    """数据库操作插件 - 示例"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self._connections = {}
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="database",
            version="1.0.0",
            description="执行SQL查询",
            author="SoloAgent",
            entry_point="DatabasePlugin"
        )
    
    def initialize(self) -> bool:
        """初始化数据库连接池"""
        # 这里可以建立连接池
        return True
    
    def execute(self, params: Dict, context: str = "") -> Any:
        import sqlite3
        
        db_path = params.get("db_path", "data/default.db")
        query = params.get("query")
        
        if not query:
            return {"error": "缺少查询语句"}
        
        # Bug修复 #20: 路径验证绕过风险 - 使用abspath和更严格的验证
        # 获取允许的基础目录的绝对路径
        base_dir = os.path.abspath("data")
        # 获取请求路径的绝对路径
        db_path = os.path.abspath(db_path)
        
        # 验证路径必须在允许的目录内
        if not db_path.startswith(base_dir + os.sep) and db_path != base_dir:
            return {"error": f"数据库路径不合法，只允许 data/ 目录下: {db_path}"}
        
        # SQL安全检查：只允许 SELECT 语句
        stripped = query.strip()
        if not stripped.upper().startswith("SELECT"):
            return {"error": "安全限制：只允许 SELECT 查询语句"}
        
        try:
            conn = None
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            
            results = [dict(row) for row in cursor.fetchall()]
            return {"results": results, "count": len(results)}
                
        except Exception as e:
            return {"error": str(e)}
        finally:
            if conn is not None:
                conn.close()
    
    def shutdown(self):
        """关闭所有连接"""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"关闭数据库连接失败: {e}")
        self._connections.clear()
