# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 配置模块
第一性原理：配置是系统行为的种子，一切从这里开始
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# ============================================================
# 日志配置 - 系统的眼睛
# ============================================================
def setup_logging(log_dir: str = "logs", level=logging.INFO):
    """配置日志系统"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "agent.log"), encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(console)
        root_logger.addHandler(file_handler)
    
    return root_logger


logger = logging.getLogger("agent.config")


# ============================================================
# 核心配置 - 用数据类保证类型安全
# ============================================================
@dataclass
class LLMConfig:
    """LLM 配置"""
    api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    api_key: str = ""
    model: str = "deepseek-ai/DeepSeek-V3"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class KnowledgeConfig:
    """知识图谱配置"""
    db_path: str = "data/knowledge.db"
    auto_save_interval: int = 30


@dataclass
class SchedulerConfig:
    """调度器配置"""
    task_queue_size: int = 100
    max_concurrent_tasks: int = 3
    task_timeout: int = 300
    retry_max: int = 3
    retry_delay: float = 2.0
    cycle_interval: float = 1.0
    optimize_interval: int = 5


@dataclass
class WebAPIConfig:
    """Web API 配置"""
    host: str = "0.0.0.0"
    port: int = 8899
    debug: bool = False


@dataclass
class AgentConfig:
    """Agent 总配置"""
    name: str = "SoloAgent"
    version: str = "1.0.0"
    workspace: str = "workspace"
    data_dir: str = "data"
    log_dir: str = "logs"
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    webapi: WebAPIConfig = field(default_factory=WebAPIConfig)


class ConfigManager:
    """配置管理器 - 让意图和执行保持同步"""
    
    DEFAULT_CONFIG_PATH = "config.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config = AgentConfig()
        self._load()
    
    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._apply_dict(data)
                logger.info(f"配置已从 {self.config_path} 加载")
            except Exception as e:
                logger.warning(f"配置加载失败，使用默认值: {e}")
        else:
            logger.info("未找到配置文件，使用默认配置")
            self.save()
    
    def _apply_dict(self, data: dict):
        for key in ['name', 'version', 'workspace', 'data_dir', 'log_dir']:
            if key in data:
                setattr(self._config, key, data[key])
        for section in ['llm', 'knowledge', 'scheduler', 'webapi']:
            if section in data:
                # 检查 data[section] 是否为字典
                if not isinstance(data[section], dict):
                    logger.warning(f"配置项 '{section}' 不是字典类型，跳过")
                    continue
                parent = getattr(self._config, section, None)
                if parent is None:
                    logger.warning(f"配置项 '{section}' 不存在，跳过")
                    continue
                for k, v in data[section].items():
                    if hasattr(parent, k):
                        setattr(parent, k, v)
    
    def save(self, path: Optional[str] = None):
        save_path = path or self.config_path
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self._config), f, ensure_ascii=False, indent=2)
        logger.info(f"配置已保存到 {save_path}")
    
    @property
    def config(self) -> AgentConfig:
        return self._config
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            parts = key.split('.')
            if len(parts) == 1 and hasattr(self._config, parts[0]):
                setattr(self._config, parts[0], value)
            elif len(parts) == 2:
                parent = getattr(self._config, parts[0], None)
                if parent and hasattr(parent, parts[1]):
                    setattr(parent, parts[1], value)
        logger.info(f"配置已更新: {kwargs}")
