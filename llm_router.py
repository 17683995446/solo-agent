# -*- coding: utf-8 -*-
"""
SoloAgent LLM 路由系统 v2.0
第一性原理：LLM = 可替换的推理引擎，路由 = 选择最优引擎
"""

import random
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import requests

logger = logging.getLogger("agent.llm_router")


class LLMProviderType(Enum):
    """LLM提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    SILICONFLOW = "siliconflow"
    AZURE = "azure"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """LLM配置"""
    name: str
    provider: LLMProviderType
    api_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    weight: int = 1  # 负载均衡权重
    priority: int = 0  # 优先级（越高越优先）
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    cost_per_1k_tokens: float = 0.0  # 成本估算


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    success: bool = True
    error: str = ""


class BaseLLMProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "errors": []
        }
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        pass
    
    def simple_chat(self, user_message: str, system_prompt: str = "") -> str:
        """简化版聊天接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages)
        # Bug修复 #15: response 可能为 None
        if response is None:
            return ""
        return response.content if response.success else ""
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._stats_lock:
            stats = dict(self._stats)
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_latency_ms"] = 0.0
        return stats
    
    def _update_stats(self, success: bool, tokens: int, latency_ms: float, error: str = ""):
        """更新统计"""
        with self._stats_lock:
            self._stats["total_requests"] += 1
            if success:
                self._stats["successful_requests"] += 1
                self._stats["total_tokens"] += tokens
                self._stats["total_latency_ms"] += latency_ms
            else:
                self._stats["failed_requests"] += 1
                if error:
                    self._stats["errors"].append({"time": time.time(), "error": error})
                    # 只保留最近10个错误
                    self._stats["errors"] = self._stats["errors"][-10:]


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI兼容格式的提供商（OpenAI、SiliconFlow、Azure等）"""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        # Bug修复 #16: 初始化 error 变量，防止 max_retries == 0 时未定义
        error = "未知错误"
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                # Bug修复 #6: 防御性检查 — 确保 choices 非空再访问
                choices = data.get('choices', [])
                if not choices or not choices[0].get('message'):
                    raise ValueError(f"API响应格式异常: choices为空或缺少message字段")
                content = choices[0]['message']['content']
                usage = data.get('usage', {})
                
                tokens_prompt = usage.get('prompt_tokens', 0)
                tokens_completion = usage.get('completion_tokens', 0)
                tokens_total = usage.get('total_tokens', tokens_prompt + tokens_completion)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # 计算成本（简化估算）
                cost = (tokens_total / 1000) * self.config.cost_per_1k_tokens
                
                self._update_stats(True, tokens_total, latency_ms)
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider=self.config.name,
                    tokens_used=tokens_total,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    latency_ms=latency_ms,
                    cost=cost,
                    success=True
                )
                
            except requests.exceptions.Timeout:
                error = f"请求超时 ({self.config.timeout}s)"
                
            except requests.exceptions.HTTPError as e:
                error = f"HTTP错误: {e.response.status_code}"
                # 某些错误不需要重试（如401认证失败）
                if e.response.status_code in [401, 403, 429]:
                    break
                    
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
            
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        # 所有重试失败
        latency_ms = (time.time() - start_time) * 1000
        self._update_stats(False, 0, latency_ms, error)
        
        return LLMResponse(
            content="",
            model=self.config.model,
            provider=self.config.name,
            latency_ms=latency_ms,
            success=False,
            error=error
        )


class OllamaProvider(BaseLLMProvider):
    """Ollama本地模型提供商"""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()
        
        # Ollama使用不同的API格式
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        }
        
        try:
            response = requests.post(
                f"{self.config.api_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            # Bug修复 #7: 防御性检查 — 确保 message 字段存在
            message = data.get('message')
            if not message or not message.get('content'):
                raise ValueError(f"Ollama API响应格式异常: 缺少message或content字段")
            content = message['content']
            
            # Ollama不返回token计数，估算
            tokens_prompt = sum(len(m['content']) // 4 for m in messages)
            tokens_completion = len(content) // 4
            tokens_total = tokens_prompt + tokens_completion
            
            latency_ms = (time.time() - start_time) * 1000
            
            self._update_stats(True, tokens_total, latency_ms)
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.name,
                tokens_used=tokens_total,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                latency_ms=latency_ms,
                cost=0.0,  # 本地模型无成本
                success=True
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error = f"{type(e).__name__}: {str(e)}"
            self._update_stats(False, 0, latency_ms, error)
            
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.name,
                latency_ms=latency_ms,
                success=False,
                error=error
            )


class LLMRouter:
    """
    LLM路由器
    支持：负载均衡、故障切换、成本优化、质量优先
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._strategy: str = "priority"  # priority, round_robin, weighted, cost, quality
        self._round_robin_index: int = 0
        self._fallback_enabled: bool = True
        self._health_check_interval: int = 60
        self._last_health_check: float = 0
        # Bug修复 #8: 添加锁保护 round_robin_index 的非原子操作
        self._select_lock = threading.Lock()
    
    def register_provider(self, config: LLMConfig) -> bool:
        """注册LLM提供商"""
        with self._select_lock:
            if config.name in self._providers:
                logger.warning(f"提供商 {config.name} 已存在，将覆盖")

            # 根据提供商类型创建对应实例
            if config.provider == LLMProviderType.OLLAMA:
                provider = OllamaProvider(config)
            else:
                provider = OpenAICompatibleProvider(config)

            self._providers[config.name] = provider
        logger.info(f"✓ 注册LLM提供商: {config.name} ({config.model})")
        return True

    def register_provider_instance(self, provider: BaseLLMProvider) -> bool:
        """
        ★ 新增：直接注册 provider 实例（用于测试和自定义 provider）
        """
        name = provider.config.name
        with self._select_lock:
            self._providers[name] = provider
        logger.info(f"✓ 注册LLM提供商实例: {name} ({provider.config.model})")
        return True

    def unregister_provider(self, name: str) -> bool:
        """注销LLM提供商"""
        with self._select_lock:
            if name in self._providers:
                del self._providers[name]
                logger.info(f"✓ 注销LLM提供商: {name}")
                return True
        return False
    
    def set_strategy(self, strategy: str):
        """
        设置路由策略
        - priority: 按优先级选择（默认）
        - round_robin: 轮询
        - weighted: 按权重
        - cost: 成本优先（选择最便宜的）
        - quality: 质量优先（选择成功率最高的）
        """
        valid_strategies = ["priority", "round_robin", "weighted", "cost", "quality"]
        if strategy not in valid_strategies:
            raise ValueError(f"无效的策略: {strategy}，可选: {valid_strategies}")
        self._strategy = strategy
        logger.info(f"路由策略设置为: {strategy}")
    
    def _select_provider(self) -> Optional[BaseLLMProvider]:
        """根据策略选择提供商"""
        # Bug修复 #8: 使用锁保护 round_robin_index 的读取和递增操作
        with self._select_lock:
            enabled_providers = [
                p for p in self._providers.values()
                if p.config.enabled
            ]
            
            if not enabled_providers:
                return None
            
            if self._strategy == "priority":
                # 选择优先级最高的
                return max(enabled_providers, key=lambda p: p.config.priority)
            
            elif self._strategy == "round_robin":
                # 轮询
                provider = enabled_providers[self._round_robin_index % len(enabled_providers)]
                self._round_robin_index += 1
                return provider
            
            elif self._strategy == "weighted":
                # 按权重随机选择
                weights = [p.config.weight for p in enabled_providers]
                total = sum(weights)
                r = random.uniform(0, total)
                cumulative = 0
                for provider in enabled_providers:
                    cumulative += provider.config.weight
                    if r <= cumulative:
                        return provider
                return enabled_providers[-1]
            
            elif self._strategy == "cost":
                # 选择成本最低的
                return min(enabled_providers, key=lambda p: p.config.cost_per_1k_tokens)
            
            elif self._strategy == "quality":
                # 选择成功率最高的
                def get_success_rate(p):
                    stats = p.get_stats()
                    return stats.get("success_rate", 0)
                return max(enabled_providers, key=get_success_rate)
            
            return enabled_providers[0]
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        发送聊天请求（带故障切换）
        """
        providers_to_try = []
        
        # 首先尝试选中的提供商
        primary = self._select_provider()
        if primary:
            providers_to_try.append(primary)
        
        # 添加其他启用的提供商作为备用
        if self._fallback_enabled:
            for provider in self._providers.values():
                if provider.config.enabled and provider not in providers_to_try:
                    providers_to_try.append(provider)
        
        if not providers_to_try:
            return LLMResponse(
                content="",
                model="",
                provider="",
                success=False,
                error="没有可用的LLM提供商"
            )
        
        # 尝试每个提供商
        last_error = ""
        for provider in providers_to_try:
            start_time = time.time()
            try:
                response = provider.chat(messages, **kwargs)
                if response.success:
                    return response
                last_error = response.error
                logger.warning(f"提供商 {provider.config.name} 失败: {last_error}，尝试下一个...")
            except Exception as e:
                # Bug修复 #22: 异常时更新统计
                latency_ms = (time.time() - start_time) * 1000
                provider._update_stats(False, 0, latency_ms, str(e))
                last_error = str(e)
                logger.warning(f"提供商 {provider.config.name} 异常: {e}，尝试下一个...")
        
        # 所有提供商都失败
        return LLMResponse(
            content="",
            model="",
            provider="",
            success=False,
            error=f"所有LLM提供商都失败，最后错误: {last_error}"
        )
    
    def simple_chat(self, user_message: str, system_prompt: str = "") -> str:
        """简化版聊天接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages)
        return response.content if response.success else ""
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """获取所有提供商的统计"""
        return {
            name: provider.get_stats()
            for name, provider in self._providers.items()
        }
    
    def get_provider_list(self) -> List[Dict]:
        """获取提供商列表"""
        return [
            {
                "name": p.config.name,
                "provider": p.config.provider.value,
                "model": p.config.model,
                "enabled": p.config.enabled,
                "priority": p.config.priority,
                "weight": p.config.weight,
                "cost_per_1k": p.config.cost_per_1k_tokens,
                "stats": p.get_stats()
            }
            for p in self._providers.values()
        ]


# ============================================================
# 便捷函数
# ============================================================

_global_router: Optional[LLMRouter] = None
_global_router_lock = threading.Lock()


def get_llm_router() -> Optional[LLMRouter]:
    """获取全局LLM路由器"""
    return _global_router


def init_llm_router() -> LLMRouter:
    """初始化全局LLM路由器"""
    global _global_router
    with _global_router_lock:
        if _global_router is None:
            _global_router = LLMRouter()
        return _global_router
