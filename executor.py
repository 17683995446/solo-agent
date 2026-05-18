# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 执行引擎 (修复版)
第一性原理：执行 = 输入任务 -> 查经验 -> 执行 -> 记录结果
修复：安全漏洞、None参数、LLM错误处理、计算异常
"""

import time
import uuid
import logging
import traceback
import os
import re
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("agent.executor")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskExecutionError(Exception):
    """任务执行错误 - 用于标记不应重试的错误"""
    pass


class SecurityError(TaskExecutionError):
    """安全错误"""
    pass


@dataclass
class Task:
    """任务 - 执行的基本单元"""
    id: str = ""
    name: str = ""
    description: str = ""
    task_type: str = "general"
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: str = ""
    error: str = ""
    method: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    retry_count: int = 0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        # 确保 params 不为 None
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "task_type": self.task_type, "params": self.params,
            "status": self.status, "result": self.result,
            "error": self.error, "method": self.method,
            "created_at": self.created_at, "started_at": self.started_at,
            "completed_at": self.completed_at, "retry_count": self.retry_count,
            "priority": self.priority
        }


class TaskExecutor:
    """
    任务执行引擎 (修复版)
    修复内容：
    1. 安全：文件操作限制在工作目录
    2. 健壮：None参数安全处理
    3. 正确：LLM错误标记为失败
    4. 异常：计算错误抛出异常
    """
    
    # 安全配置
    ALLOWED_WORKSPACE = os.path.abspath("workspace")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_STRING_LENGTH = 100000  # 最大字符串长度
    
    def __init__(self, knowledge_graph=None, llm_client=None, config=None):
        self.kg = knowledge_graph
        self.llm = llm_client
        self.config = config
        self._handlers = {}
        self._stats = {
            "total_executed": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_retries": 0,
            "security_blocked": 0
        }
        self._register_builtin_handlers()
    
    def _register_builtin_handlers(self):
        """注册内置任务处理器"""
        self._handlers["echo"] = self._handle_echo
        self._handlers["compute"] = self._handle_compute
        self._handlers["llm_task"] = self._handle_llm_task
        self._handlers["file_write"] = self._handle_file_write
        self._handlers["file_read"] = self._handle_file_read
        self._handlers["research"] = self._handle_research
        self._handlers["summarize"] = self._handle_summarize
        self._handlers["health_check"] = self._handle_health_check
        self._handlers["knowledge_query"] = self._handle_knowledge_query
        self._handlers["self_improve"] = self._handle_self_improve
        self._handlers["pipeline"] = self._handle_pipeline
        logger.info(f"已注册 {len(self._handlers)} 个内置任务处理器")
    
    def register_handler(self, task_type: str, handler: Callable):
        """注册自定义任务处理器"""
        self._handlers[task_type] = handler
        logger.info(f"注册自定义处理器: {task_type}")
    
    def _safe_get_param(self, task: Task, key: str, default: Any = None) -> Any:
        """安全获取参数 - 处理None params"""
        if task.params is None:
            return default
        return task.params.get(key, default)
    
    def _validate_path(self, filepath: str) -> str:
        """
        验证文件路径安全性
        修复：防止路径穿越攻击 - 明确拒绝而非静默重定向
        """
        if not filepath:
            raise SecurityError("未指定文件路径")
        
        # 规范化路径
        abs_path = os.path.abspath(filepath)
        
        # 检查路径穿越模式
        if '..' in filepath or filepath.startswith('/') or filepath.startswith('~'):
            raise SecurityError(f"路径穿越被拒绝: {filepath}")
        
        # 只允许在工作目录内操作
        workspace = os.path.abspath("workspace")
        os.makedirs(workspace, exist_ok=True)
        
        # 检查最终路径是否在工作目录内
        if not abs_path.startswith(workspace):
            raise SecurityError(f"访问被拒绝: 只允许在workspace目录内操作")
        
        return abs_path
    
    def execute(self, task: Task) -> Task:
        """执行任务 - 核心方法"""
        task.status = TaskStatus.RUNNING.value
        task.started_at = datetime.now().isoformat()
        logger.info(f"▶ 开始执行任务 [{task.id}] {task.name} (类型: {task.task_type})")
        
        try:
            # 确保 params 不为 None
            if task.params is None:
                task.params = {}
            
            # 从知识图谱获取相关经验
            context = ""
            if self.kg:
                context = self.kg.get_relevant_context(task.description or task.name)
                if context != "暂无相关经验":
                    logger.info(f"  从知识图谱获取到相关上下文 ({len(context)}字符)")
                    task.metadata['kg_context'] = context
            
            # 查找并调用处理器
            handler = self._handlers.get(task.task_type)
            
            if handler:
                result = handler(task, context)
                
                # 检查结果是否为空（LLM错误情况）
                if result is None or (isinstance(result, str) and result == ""):
                    raise TaskExecutionError("处理器返回空结果")
                
                # 检查结果是否包含错误标记
                if isinstance(result, str) and result.startswith("错误:"):
                    raise TaskExecutionError(result)
                
                task.result = str(result)[:self.MAX_STRING_LENGTH]  # 限制长度
                task.status = TaskStatus.SUCCESS.value
                task.method = f"handler:{task.task_type}"
                self._stats["total_success"] += 1
            else:
                # 未知任务类型 - 不应重试
                raise TaskExecutionError(f"未找到任务类型 '{task.task_type}' 的处理器")
        
        except SecurityError as e:
            # 安全错误 - 记录但不重试
            task.error = f"安全错误: {str(e)}"
            task.status = TaskStatus.FAILED.value
            self._stats["total_failed"] += 1
            self._stats["security_blocked"] += 1
            logger.warning(f"  ⚠ 安全拦截: {task.error}")
        
        except TaskExecutionError as e:
            # 任务执行错误 - 不应重试的错误
            task.error = str(e)
            task.status = TaskStatus.FAILED.value
            self._stats["total_failed"] += 1
            logger.error(f"  ✗ {task.error}")
        
        except Exception as e:
            task.error = f"{type(e).__name__}: {str(e)}"
            task.status = TaskStatus.FAILED.value
            self._stats["total_failed"] += 1
            logger.error(f"  ✗ 任务执行异常: {task.error}\n{traceback.format_exc()}")
        
        finally:
            task.completed_at = datetime.now().isoformat()
            self._stats["total_executed"] += 1
            
            # 将结果存入知识图谱
            if self.kg:
                try:
                    self.kg.store_task_result(
                        task_name=task.name,
                        result=task.result[:500] if task.result else "",
                        success=(task.status == TaskStatus.SUCCESS.value),
                        error=task.error,
                        method=task.method
                    )
                except Exception as e:
                    logger.warning(f"  存储任务结果到知识图谱失败: {e}")
            
            status_icon = "✓" if task.status == TaskStatus.SUCCESS.value else "✗"
            logger.info(f"  {status_icon} 任务 [{task.id}] {task.name} -> {task.status}")
        
        return task
    
    def execute_with_retry(self, task: Task, max_retries: int = 3, 
                           retry_delay: float = 2.0, 
                           should_stop: Optional[Callable[[], bool]] = None) -> Task:
        """带重试的执行 - 智能重试策略，支持外部停止信号"""
        for attempt in range(max_retries + 1):
            # 检查外部停止信号
            if should_stop and should_stop():
                task.error = "任务被取消（系统停止）"
                task.status = TaskStatus.FAILED.value
                logger.info(f"  ⊘ 任务 [{task.id}] 被取消")
                return task
            
            if attempt > 0:
                task.retry_count = attempt
                task.status = TaskStatus.RETRYING.value
                self._stats["total_retries"] += 1
                logger.info(f"  ↻ 重试任务 [{task.id}] 第 {attempt} 次 (共 {max_retries} 次)")
                time.sleep(retry_delay * attempt)
            
            self.execute(task)
            
            if task.status == TaskStatus.SUCCESS.value:
                return task
            
            # 检查是否应该重试
            # 安全错误和任务执行错误不重试
            if "安全错误" in task.error or "未找到任务类型" in task.error:
                logger.info(f"  ⊘ 跳过重试: {task.error}")
                break
            
            # LLM认证错误不重试（401等）- 检查error和LLM错误计数
            if "401" in task.error or "Unauthorized" in task.error or "认证" in task.error:
                logger.info(f"  ⊘ 跳过重试: LLM认证错误")
                break
            
            # LLM连续失败不重试（API不可用时重试无意义）
            if self.llm and self.llm._error_count > 0:
                if "LLM 返回空结果" in task.error or "LLM" in task.error:
                    logger.info(f"  ⊘ 跳过重试: LLM服务不可用 (连续{self.llm._error_count}次错误)")
                    break
        
        if task.status != TaskStatus.SUCCESS.value:
            logger.error(f"  ✗ 任务 [{task.id}] 执行失败")
        return task
    
    # ---- 内置任务处理器 ----
    
    def _handle_echo(self, task: Task, context: str) -> str:
        """回声任务 - 返回输入内容"""
        msg = self._safe_get_param(task, "message", task.description or "")
        # 限制长度
        if len(str(msg)) > self.MAX_STRING_LENGTH:
            msg = str(msg)[:self.MAX_STRING_LENGTH] + "...(截断)"
        return f"Echo: {msg}"
    
    def _handle_compute(self, task: Task, context: str) -> str:
        """
        计算任务 - 安全执行数学表达式
        修复：正确处理计算错误
        """
        expr = self._safe_get_param(task, "expression", task.description)
        
        # 安全检查：只允许数学字符
        allowed = set("0123456789+-*/.() %^")
        if not all(c in allowed for c in str(expr)):
            raise SecurityError("不安全的表达式")
        
        # 额外检查：禁止危险模式
        dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'os', 'sys']
        for pattern in dangerous_patterns:
            if pattern in str(expr).lower():
                raise SecurityError(f"禁止的表达式模式: {pattern}")
        
        try:
            # 使用受限的eval
            result = eval(expr, {"__builtins__": {}}, {})
            return f"计算结果: {expr} = {result}"
        except ZeroDivisionError:
            raise TaskExecutionError("计算错误: 除零")
        except OverflowError:
            raise TaskExecutionError("计算错误: 数值溢出")
        except Exception as e:
            raise TaskExecutionError(f"计算错误: {type(e).__name__}: {e}")
    
    def _handle_llm_task(self, task: Task, context: str) -> str:
        """
        LLM 任务 - 使用大模型处理
        修复：正确处理LLM错误
        """
        if not self.llm:
            raise TaskExecutionError("LLM 客户端未配置")
        
        prompt = self._safe_get_param(task, "prompt", task.description)
        system = self._safe_get_param(task, "system_prompt", "你是一个全能助手，请高效完成任务。")
        
        # 注入知识图谱上下文
        if context and context != "暂无相关经验":
            prompt = f"参考经验:\n{context}\n\n当前任务:\n{prompt}"
        
        result = self.llm.simple_chat(prompt, system)
        
        # 检查LLM是否返回有效结果
        if not result:
            raise TaskExecutionError("LLM 返回空结果（可能是超时或API错误）")
        
        return result
    
    def _handle_file_write(self, task: Task, context: str) -> str:
        """
        文件写入任务
        修复：路径安全验证
        """
        filepath = self._safe_get_param(task, "path", "")
        content = self._safe_get_param(task, "content", "")
        
        # 安全验证路径
        filepath = self._validate_path(filepath)
        
        # 限制内容大小
        if len(content) > self.MAX_FILE_SIZE:
            raise TaskExecutionError(f"内容过大: {len(content)} 字节 (最大 {self.MAX_FILE_SIZE})")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"文件已写入: {filepath} ({len(content)} 字符)"
    
    def _handle_file_read(self, task: Task, context: str) -> str:
        """
        文件读取任务
        修复：路径安全验证，防止路径穿越
        """
        filepath = self._safe_get_param(task, "path", "")
        
        # 安全验证路径
        filepath = self._validate_path(filepath)
        
        try:
            # 检查文件大小
            if os.path.getsize(filepath) > self.MAX_FILE_SIZE:
                raise TaskExecutionError(f"文件过大: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:2000]
        except FileNotFoundError:
            raise TaskExecutionError(f"文件不存在: {filepath}")
        except PermissionError:
            raise SecurityError(f"无权限访问: {filepath}")
    
    def _handle_research(self, task: Task, context: str) -> str:
        """研究任务 - 使用 LLM 进行深度分析"""
        if not self.llm:
            raise TaskExecutionError("LLM 客户端未配置")
        
        topic = self._safe_get_param(task, "topic", task.description)
        depth = self._safe_get_param(task, "depth", "medium")
        
        system = f"你是一个研究助手。请对以下主题进行{depth}深度的分析。"
        if context and context != "暂无相关经验":
            system += f"\n\n已有相关知识:\n{context}"
        
        result = self.llm.simple_chat(
            f"请研究分析: {topic}\n\n要求:\n1. 概述\n2. 关键要点\n3. 结论和建议",
            system
        )
        
        if not result:
            raise TaskExecutionError("LLM 研究任务返回空结果")
        
        return result
    
    def _handle_summarize(self, task: Task, context: str) -> str:
        """摘要任务"""
        if not self.llm:
            raise TaskExecutionError("LLM 客户端未配置")
        
        text = self._safe_get_param(task, "text", task.description)
        result = self.llm.simple_chat(
            f"请简洁总结以下内容:\n\n{text}",
            "你是一个摘要助手，请提取关键信息，用简洁的语言总结。"
        )
        
        if not result:
            raise TaskExecutionError("LLM 摘要任务返回空结果")
        
        return result
    
    def _handle_health_check(self, task: Task, context: str) -> str:
        """健康检查任务"""
        import sys
        
        checks = {
            "系统运行时间": f"{time.time():.0f}s",
            "Python版本": sys.version.split()[0],
            "工作目录": os.getcwd(),
            "知识图谱": f"{'正常' if self.kg else '未配置'}",
            "LLM": f"{'正常' if self.llm else '未配置'}",
            "已执行任务": str(self._stats["total_executed"]),
            "成功率": f"{self._stats['total_success'] / max(1, self._stats['total_executed']) * 100:.1f}%",
            "安全拦截": str(self._stats["security_blocked"])
        }
        
        if self.kg:
            stats = self.kg.get_stats()
            checks["知识节点数"] = str(stats["total_nodes"])
            checks["知识边数"] = str(stats["total_edges"])
        
        if self.llm:
            llm_stats = self.llm.get_stats()
            checks["LLM调用次数"] = str(llm_stats["total_calls"])
            checks["LLM错误次数"] = str(llm_stats["error_count"])
        
        result = "\n".join(f"  {k}: {v}" for k, v in checks.items())
        return f"系统健康检查:\n{result}"
    
    def _handle_knowledge_query(self, task: Task, context: str) -> str:
        """知识查询任务"""
        if not self.kg:
            raise TaskExecutionError("知识图谱未配置")
        
        query = self._safe_get_param(task, "query", task.description)
        category = self._safe_get_param(task, "category")
        limit = self._safe_get_param(task, "limit", 10)
        
        if not query:
            return "提示: 请提供查询关键词"
        
        results = self.kg.search_nodes(query, category=category, limit=limit)
        
        if not results:
            return f"未找到与 '{query}' 相关的知识"
        
        output = [f"找到 {len(results)} 条相关知识:"]
        for i, r in enumerate(results, 1):
            output.append(f"  {i}. [{r.get('category', '?')}] {r.get('title', '无标题')}")
            content = r.get('content', '')
            if content:
                output.append(f"     {content[:150]}")
        
        return "\n".join(output)
    
    def _handle_self_improve(self, task: Task, context: str) -> str:
        """自我改进任务"""
        if not self.kg:
            raise TaskExecutionError("知识图谱未配置")
        
        experiences = self.kg.get_nodes_by_category("experience")
        errors = self.kg.get_nodes_by_category("error")
        methods = self.kg.get_nodes_by_category("method")
        
        if not self.llm:
            summary = f"经验数: {len(experiences)}, 错误数: {len(errors)}, 方法数: {len(methods)}"
            return f"自我改进分析(无LLM模式):\n  {summary}\n  建议: 配置LLM以获得更深入的改进分析"
        
        analysis_input = f"当前统计:\n- 成功经验: {len(experiences)}条\n- 失败记录: {len(errors)}条\n- 优化方法: {len(methods)}条\n"
        
        if errors:
            analysis_input += "\n最近的错误:\n"
            for e in errors[-5:]:
                analysis_input += f"- {e.get('title')}: {e.get('content', '')[:100]}\n"
        
        if methods:
            analysis_input += "\n已有的优化方法:\n"
            for m in methods[-5:]:
                analysis_input += f"- {m.get('title')}: {m.get('content', '')[:100]}\n"
        
        analysis_input += "\n请分析以上信息，提出具体的改进建议。"
        
        result = self.llm.simple_chat(
            analysis_input,
            "你是一个自我改进分析器。分析执行历史，找出模式和改进机会，给出具体可执行的建议。"
        )
        
        if not result:
            raise TaskExecutionError("自我改进分析返回空结果")
        
        return result
    
    def _handle_pipeline(self, task: Task, context: str) -> str:
        """
        链式任务 - 顺序执行多个步骤，前一步的输出作为后一步的输入
        params: {"steps": [{"type": "echo", "params": {"message": "hello"}}, ...]}
        """
        steps = self._safe_get_param(task, "steps", [])
        if not steps or not isinstance(steps, list):
            raise TaskExecutionError("pipeline 任务需要 steps 数组")
        
        if len(steps) > 20:
            raise TaskExecutionError("pipeline 步骤不能超过20个")
        
        results = []
        carry_over = ""  # 上一步的输出传递给下一步
        
        for i, step in enumerate(steps):
            step_type = step.get("type", "echo")
            step_params = step.get("params", {})
            
            # 将上一步结果注入参数
            if carry_over:
                step_params["_prev_result"] = carry_over[:2000]  # 限制长度
            
            step_task = Task(
                name=f"{task.name}_step{i}",
                task_type=step_type,
                params=step_params or {}
            )
            
            handler = self._handlers.get(step_type)
            if not handler:
                raise TaskExecutionError(f"pipeline 步骤{i}: 未知类型 '{step_type}'")
            
            try:
                result = handler(step_task, "")
                carry_over = str(result) if result else ""
                results.append({"step": i, "type": step_type, "status": "success", "result": carry_over[:200]})
            except Exception as e:
                raise TaskExecutionError(f"pipeline 步骤{i} ({step_type}) 失败: {e}")
        
        summary = "\n".join(
            f"  步骤{j['step']}: [{j['type']}] {j['status']}" 
            for j in results
        )
        return f"Pipeline 完成 ({len(results)}步):\n{summary}"
    
    def get_stats(self) -> Dict:
        return dict(self._stats)
    
    def get_registered_handlers(self) -> List[str]:
        return list(self._handlers.keys())
