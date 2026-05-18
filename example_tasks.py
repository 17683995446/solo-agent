# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 示例任务集
混合模式：内置多种示例任务，展示 Agent 的能力
"""

from typing import List, Dict


def get_example_tasks() -> List[Dict]:
    """
    获取示例任务列表
    这些任务展示了 Agent 的各种能力
    """
    return [
        # ---- 基础能力展示 ----
        {
            "name": "系统自检",
            "description": "检查 Agent 各模块运行状态",
            "task_type": "health_check",
            "params": {},
            "priority": 10
        },
        {
            "name": "数学计算",
            "description": "计算斐波那契数列前10项之和",
            "task_type": "compute",
            "params": {"expression": "sum([1,1,2,3,5,8,13,21,34,55])"},
            "priority": 5
        },
        {
            "name": "回声测试",
            "description": "测试基本通信链路",
            "task_type": "echo",
            "params": {"message": "Agent 运行正常!"},
            "priority": 3
        },
        
        # ---- LLM 智能任务 ----
        {
            "name": "自我介绍生成",
            "description": "让 LLM 生成 Agent 的自我介绍",
            "task_type": "llm_task",
            "params": {
                "prompt": "请用简洁有力的语言，为这个极简全能持续运行 Agent 写一段自我介绍（100字以内）。突出三个特点：永不停歇、自我优化、知识积累。",
                "system_prompt": "你是一个 AI Agent 的代言人。"
            },
            "priority": 8
        },
        {
            "name": "技术趋势分析",
            "description": "分析当前 AI Agent 技术发展趋势",
            "task_type": "research",
            "params": {
                "topic": "2025年 AI Agent 自主运行和自我优化的技术趋势",
                "depth": "medium"
            },
            "priority": 6
        },
        
        # ---- 知识管理 ----
        {
            "name": "存储核心知识",
            "description": "将 Agent 的核心设计原则存入知识图谱",
            "task_type": "knowledge_query",
            "params": {
                "query": "Agent 设计原则"
            },
            "priority": 7
        },
        
        # ---- 自我改进 ----
        {
            "name": "自我改进分析",
            "description": "分析当前执行策略，提出改进方案",
            "task_type": "self_improve",
            "params": {},
            "priority": 4
        },
        
        # ---- 文件操作 ----
        {
            "name": "写入状态报告",
            "description": "将当前状态写入文件",
            "task_type": "file_write",
            "params": {
                "path": "workspace/status_report.txt",
                "content": "Agent 状态报告\n生成时间: {auto}\n状态: 运行中\n注意: 此文件由 Agent 自动生成"
            },
            "priority": 2
        },
    ]


def get_continuous_tasks() -> List[Dict]:
    """
    获取持续运行时的周期性任务
    这些任务会被周期性地重新加入队列
    """
    return [
        {
            "name": "周期健康检查",
            "description": "定期检查系统健康状态",
            "task_type": "health_check",
            "params": {},
            "priority": 5
        },
        {
            "name": "知识库整理",
            "description": "回顾和整理知识图谱",
            "task_type": "knowledge_query",
            "params": {"query": "所有经验", "limit": 10},
            "priority": 3
        },
        {
            "name": "执行策略优化",
            "description": "基于历史数据优化执行策略",
            "task_type": "self_improve",
            "params": {},
            "priority": 4
        },
    ]
