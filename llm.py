# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - LLM 接口模块
第一性原理：LLM 是一个函数 f(prompt) -> response，封装这个函数
"""

import json
import logging
import requests
from typing import Optional, List, Dict, Any

logger = logging.getLogger("agent.llm")


class LLMClient:
    """
    LLM 客户端
    第一性原理：LLM 就是一个文本到文本的变换器
    - 输入：系统提示 + 用户消息
    - 输出：文本响应
    - 附加：错误处理、重试、上下文管理
    """
    
    def __init__(self, api_url: str, api_key: str, model: str = "deepseek-ai/DeepSeek-V3",
                 temperature: float = 0.7, max_tokens: int = 2048, timeout: int = 60):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._total_calls = 0
        self._total_tokens = 0
        self._error_count = 0
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """
        发送聊天请求
        messages: [{"role": "system/user/assistant", "content": "..."}]
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False
        }
        
        try:
            logger.debug(f"LLM 请求: model={self.model}, messages={len(messages)}条")
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # 统计
            self._total_calls += 1
            usage = data.get('usage', {})
            self._total_tokens += usage.get('total_tokens', 0)
            
            logger.debug(f"LLM 响应: {len(content)}字符, tokens={usage.get('total_tokens', '?')}")
            return content
            
        except requests.exceptions.Timeout:
            self._error_count += 1
            logger.error(f"LLM 请求超时 ({self.timeout}s)")
            return ""
        except requests.exceptions.ConnectionError:
            self._error_count += 1
            logger.error("LLM 连接失败")
            return ""
        except KeyError as e:
            self._error_count += 1
            logger.error(f"LLM 响应格式异常: {e}")
            return ""
        except Exception as e:
            self._error_count += 1
            logger.error(f"LLM 请求失败: {e}")
            return ""
    
    def simple_chat(self, user_message: str, system_prompt: str = "") -> str:
        """简化版聊天接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages)
    
    def analyze(self, text: str, instruction: str = "请分析以下内容:") -> str:
        """分析文本"""
        return self.simple_chat(f"{instruction}\n\n{text}")
    
    def get_stats(self) -> Dict:
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "error_count": self._error_count,
            "model": self.model
        }
