# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 优化模块 (修复版)
第一性原理：优化 = 观察(结果) -> 分析(差距) -> 改进(方法) -> 验证(效果)
修复：减少不必要的LLM调用，提高执行效率
"""

import time
import random
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("agent.optimizer")


@dataclass
class OptimizationRecord:
    """优化记录"""
    id: str = ""
    target: str = ""
    original: str = ""
    improved: str = ""
    reason: str = ""
    improvement: str = ""
    applied: bool = False
    verified: bool = False
    timestamp: str = ""


class Optimizer:
    """
    自动优化模块 (修复版)
    
    修复内容：
    1. 减少LLM调用频率 - 只在失败时调用，成功时采样调用
    2. 批量反思 - 积累一定数量后再分析
    3. 轻量级记录 - 不调用LLM时只做简单记录
    """
    
    # 配置
    SUCCESS_REFLECT_RATE = 0.1  # 成功任务反思比例 (10%)
    BATCH_SIZE = 10  # 批量反思的阈值
    
    def __init__(self, knowledge_graph=None, llm_client=None):
        self.kg = knowledge_graph
        self.llm = llm_client
        self._optimization_history: List[OptimizationRecord] = []
        self._task_history: List[Dict] = []
        self._pending_successes: List[Dict] = []  # 待批量分析的成功任务
        self._stats = {
            "optimizations_proposed": 0,
            "optimizations_applied": 0,
            "optimizations_verified": 0,
            "total_reflections": 0,
            "llm_calls_saved": 0  # 节省的LLM调用次数
        }
    
    def reflect_on_task(self, task_result: Dict) -> Optional[OptimizationRecord]:
        """
        对单个任务结果进行反思 (优化版)
        策略：
        - 失败任务：立即分析（重要）
        - 成功任务：采样分析（10%概率）或批量分析
        """
        self._task_history.append(task_result)
        if len(self._task_history) > 100:
            self._task_history = self._task_history[-100:]
        
        self._stats["total_reflections"] += 1
        
        name = task_result.get("name", "")
        status = task_result.get("status", "")
        result = task_result.get("result", "")
        error = task_result.get("error", "")
        method = task_result.get("method", "")
        
        # 失败任务：立即分析（优先级最高）
        if status == "failed":
            return self._analyze_failure(name, error, method)
        
        # 成功任务：采样或批量分析
        if status == "success":
            # 采样分析
            if random.random() < self.SUCCESS_REFLECT_RATE:
                return self._extract_success_pattern(name, result, method)
            else:
                # 记录但不调用LLM
                self._pending_successes.append(task_result)
                self._stats["llm_calls_saved"] += 1
                
                # 批量分析
                if len(self._pending_successes) >= self.BATCH_SIZE:
                    batch_result = self._batch_analyze_successes()
                    self._pending_successes = []
                    return batch_result
        
        return None
    
    def _extract_success_pattern(self, name: str, result: str, method: str) -> Optional[OptimizationRecord]:
        """从成功中提取模式 - 简化版"""
        # 无LLM时跳过
        if not self.llm or not result:
            return None
        
        # 只对有意义的成功进行分析
        if len(result) < 50:
            return None
        
        try:
            prompt = f"""分析这个成功的任务执行，提取可复用的经验:

任务: {name}
结果摘要: {result[:300]}

用一句话回答: 这个成功的关键因素是什么？"""
            
            response = self.llm.simple_chat(prompt, "你是任务分析专家，极简回答。")
            
            if response and len(response) > 10:
                record = OptimizationRecord(
                    target=name,
                    original=method,
                    improved=f"成功模式: {name}",
                    reason="成功模式提取",
                    improvement=response[:200],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                self._optimization_history.append(record)
                self._stats["optimizations_proposed"] += 1
                
                # 存入知识图谱
                if self.kg:
                    self.kg.store_experience(
                        title=f"成功: {name}",
                        content=response[:300],
                        category="method",
                        tags=["success", name],
                        importance=0.5
                    )
                
                logger.info(f"  💡 成功模式提取: {name}")
                return record
        except Exception as e:
            logger.debug(f"  成功模式提取失败: {e}")
        
        return None
    
    def _batch_analyze_successes(self) -> Optional[OptimizationRecord]:
        """批量分析成功任务"""
        if not self.llm or not self._pending_successes:
            return None
        
        try:
            summary = f"批量分析 {len(self._pending_successes)} 个成功任务:\n"
            for t in self._pending_successes[-5:]:
                summary += f"- {t.get('name', '?')}: {t.get('task_type', '?')}\n"
            
            prompt = f"{summary}\n用一句话总结这些成功任务的共同特点。"
            response = self.llm.simple_chat(prompt, "你是分析专家，极简回答。")
            
            if response:
                record = OptimizationRecord(
                    target="批量成功分析",
                    original="",
                    improved="批量模式总结",
                    reason="批量反思",
                    improvement=response[:200],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                self._optimization_history.append(record)
                self._stats["optimizations_proposed"] += 1
                logger.info(f"  📊 批量成功分析完成")
                return record
        except Exception as e:
            logger.debug(f"  批量分析失败: {e}")
        
        return None
    
    def _analyze_failure(self, name: str, error: str, method: str) -> Optional[OptimizationRecord]:
        """分析失败原因 - 保持完整分析"""
        if not error:
            return None
        
        improvement = f"任务 '{name}' 失败: {error}"
        
        # 有LLM时做深入分析
        if self.llm:
            try:
                prompt = f"""分析失败任务:

任务: {name}
错误: {error}

用一句话回答: 如何避免这个错误？"""
                
                response = self.llm.simple_chat(prompt, "你是错误分析专家，极简回答。")
                if response:
                    improvement = response[:300]
            except Exception as e:
                logger.debug(f"  LLM失败分析失败: {e}")
        
        record = OptimizationRecord(
            target=name,
            original=method,
            improved=f"修复方案: {name}",
            reason=f"失败: {error[:50]}",
            improvement=improvement,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self._optimization_history.append(record)
        self._stats["optimizations_proposed"] += 1
        
        # 存入知识图谱
        if self.kg:
            self.kg.store_experience(
                title=f"失败分析: {name}",
                content=f"错误: {error}\n改进: {improvement}",
                category="error",
                tags=["failure", "analysis", name],
                importance=0.8
            )
        
        logger.info(f"  🔧 失败分析: {name}")
        return record
    
    def deep_reflection(self) -> str:
        """深度反思 - 周期性执行"""
        if not self._task_history:
            return "暂无任务历史，无法进行深度反思"
        
        self._stats["total_reflections"] += 1
        
        # 统计分析
        total = len(self._task_history)
        success = sum(1 for t in self._task_history if t.get("status") == "success")
        failed = total - success
        success_rate = success / max(1, total) * 100
        
        # 失败模式分析
        error_patterns = {}
        for t in self._task_history:
            if t.get("status") == "failed":
                err = t.get("error", "unknown")[:50]
                error_patterns[err] = error_patterns.get(err, 0) + 1
        
        reflection = f"=== 深度反思报告 ===\n"
        reflection += f"分析范围: {total} 个任务\n"
        reflection += f"成功率: {success_rate:.1f}% (成功{success}, 失败{failed})\n"
        reflection += f"节省LLM调用: {self._stats['llm_calls_saved']} 次\n"
        
        if error_patterns:
            reflection += f"\n常见错误模式:\n"
            for err, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]:
                reflection += f"  - [{count}次] {err}\n"
        
        # LLM深度分析
        if self.llm and total >= 3:
            try:
                recent = self._task_history[-10:]
                recent_summary = "\n".join(
                    f"- {t.get('name', '?')}: {t.get('status', '?')}"
                    for t in recent
                )
                
                prompt = f"""任务执行统计:
{total}个任务, 成功率{success_rate:.1f}%

最近任务:
{recent_summary}

用3句话回答: 1)整体表现如何？2)主要问题是什么？3)改进建议？"""
                
                llm_analysis = self.llm.simple_chat(prompt, "你是系统优化专家。简洁回答。")
                reflection += f"\n=== 分析建议 ===\n{llm_analysis}\n"
                
                if self.kg:
                    self.kg.store_optimization(
                        original_method="当前策略",
                        optimized_method=llm_analysis[:200],
                        reason="深度反思",
                        improvement=f"成功率: {success_rate:.1f}%"
                    )
            except Exception as e:
                reflection += f"\nLLM分析失败: {e}\n"
        
        logger.info(f"  📊 深度反思: 成功率 {success_rate:.1f}%")
        return reflection
    
    def get_optimization_history(self, limit: int = 20) -> List[Dict]:
        return [
            {
                "id": r.id, "target": r.target,
                "original": r.original, "improved": r.improved,
                "reason": r.reason, "improvement": r.improvement,
                "timestamp": r.timestamp
            }
            for r in self._optimization_history[-limit:]
        ]
    
    def get_stats(self) -> Dict:
        return dict(self._stats)
