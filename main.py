# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - 主入口
第一性原理：一切从启动开始，一切永不停止
"""

import os
import sys
import time
import logging
import threading
import argparse

# 确保可以导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager, setup_logging
from knowledge import KnowledgeGraph
from llm import LLMClient
from executor import TaskExecutor
from optimizer import Optimizer
from scheduler import KeepAliveScheduler
from webapi import WebAPI
from example_tasks import get_example_tasks, get_continuous_tasks

logger = logging.getLogger("agent.main")


class SoloAgent:
    """
    极简全能持续运行 Agent
    第一性原理：
    - 启动即永不停歇
    - 执行、优化、存知识三线并行
    - 出错自动修复，经验自动积累
    """
    
    def __init__(self, config_path: str = None):
        # 1. 加载配置
        self.config_mgr = ConfigManager(config_path)
        self.config = self.config_mgr.config
        
        # 2. 初始化日志
        setup_logging(self.config.log_dir)
        logger.info("初始化 SoloAgent...")
        
        # 3. 初始化知识图谱
        self.kg = KnowledgeGraph(self.config.knowledge.db_path)
        
        # 4. 初始化 LLM 客户端
        self.llm = LLMClient(
            api_url=self.config.llm.api_url,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=self.config.llm.timeout
        )
        
        # 5. 初始化执行引擎
        self.executor = TaskExecutor(
            knowledge_graph=self.kg,
            llm_client=self.llm,
            config=self.config
        )
        
        # 6. 初始化优化模块
        self.optimizer = Optimizer(
            knowledge_graph=self.kg,
            llm_client=self.llm
        )
        
        # 7. 初始化调度器
        self.scheduler = KeepAliveScheduler(
            config=self.config,
            knowledge_graph=self.kg,
            executor=self.executor,
            optimizer=self.optimizer
        )
        
        # 8. 初始化 Web API
        self.webapi = WebAPI(
            scheduler=self.scheduler,
            knowledge_graph=self.kg,
            executor=self.executor,
            optimizer=self.optimizer
        )
        
        # 9. 存储初始知识
        self._seed_knowledge()
        
        logger.info("SoloAgent 初始化完成!")
    
    def _seed_knowledge(self):
        """注入初始知识 - Agent 的第一份经验"""
        initial_knowledge = [
            {
                "title": "Agent 核心设计原则",
                "content": "1. 永不停歇：开机即运转，无空闲时刻\n2. 三线并行：执行、优化、存知识同时进行\n3. 第一性原理：从根本出发解决问题\n4. 自我修复：出错自动恢复\n5. 经验积累：越运行越聪明",
                "category": "method",
                "tags": ["core", "principle", "design"],
                "importance": 1.0
            },
            {
                "title": "任务执行流程",
                "content": "1. 从知识图谱查询相关经验\n2. 选择合适的处理器执行任务\n3. 记录执行结果\n4. 反思并优化方法\n5. 将新经验存入知识图谱",
                "category": "method",
                "tags": ["workflow", "execution"],
                "importance": 0.9
            },
            {
                "title": "错误处理策略",
                "content": "1. 单次失败：立即重试(最多3次，指数退避)\n2. 持续失败：分析根因，存入知识图谱\n3. 模块崩溃：自动恢复机制重启\n4. 未知错误：记录并跳过，不中断主循环",
                "category": "method",
                "tags": ["error", "recovery", "strategy"],
                "importance": 0.9
            }
        ]
        
        for k in initial_knowledge:
            try:
                self.kg.store_experience(**k)
            except Exception as e:
                logger.debug(f"初始知识注入跳过(可能已存在): {e}")
    
    def run(self, with_api: bool = True, load_examples: bool = True):
        """启动 Agent"""
        # 加载示例任务
        if load_examples:
            examples = get_example_tasks()
            self.scheduler.add_tasks(examples)
            logger.info(f"已加载 {len(examples)} 个示例任务")
        
        if with_api:
            # API 在独立线程运行
            api_thread = threading.Thread(
                target=self.webapi.run,
                kwargs={
                    "host": self.config.webapi.host,
                    "port": self.config.webapi.port,
                    "debug": self.config.webapi.debug
                },
                daemon=True
            )
            api_thread.start()
            logger.info(
                f"Web API 已在后台启动: "
                f"http://{self.config.webapi.host}:{self.config.webapi.port}"
            )
        
        # 启动主循环（阻塞）
        self.scheduler.start()
        
        # 主循环结束后清理资源
        self.kg.close()
    
    def run_cli(self):
        """命令行交互模式"""
        print("\n" + "=" * 60)
        print("  极简全能持续运行 Agent - 命令行模式")
        print("  输入 'help' 查看命令，'quit' 退出")
        print("=" * 60 + "\n")
        
        # 在后台线程启动调度器
        scheduler_thread = threading.Thread(target=self.scheduler.start, daemon=True)
        scheduler_thread.start()
        
        # 同时启动 API
        api_thread = threading.Thread(
            target=self.webapi.run,
            kwargs={
                "host": self.config.webapi.host,
                "port": self.config.webapi.port,
            },
            daemon=True
        )
        api_thread.start()
        
        time.sleep(1)  # 等待启动
        
        # 命令行交互
        while True:
            try:
                cmd = input("Agent> ").strip()
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if action == 'quit' or action == 'exit':
                    self.scheduler.stop()
                    break
                elif action == 'help':
                    self._print_help()
                elif action == 'status':
                    self._print_status()
                elif action == 'add':
                    self._cli_add_task(arg)
                elif action == 'queue':
                    self._print_queue()
                elif action == 'recent':
                    self._print_recent()
                elif action == 'knowledge':
                    self._cli_knowledge(arg)
                elif action == 'optimize':
                    result = self.optimizer.deep_reflection()
                    print(result)
                elif action == 'pause':
                    self.scheduler.pause()
                    print("已暂停")
                elif action == 'resume':
                    self.scheduler.resume()
                    print("已恢复")
                elif action == 'examples':
                    tasks = get_example_tasks()
                    self.scheduler.add_tasks(tasks)
                    print(f"已添加 {len(tasks)} 个示例任务")
                elif action == 'continuous':
                    tasks = get_continuous_tasks()
                    self.scheduler.add_tasks(tasks)
                    print(f"已添加 {len(tasks)} 个周期性任务")
                else:
                    # 当作自然语言任务处理
                    self.scheduler.add_task(
                        name=cmd[:50],
                        description=cmd,
                        task_type="llm_task",
                        params={"prompt": cmd},
                        priority=5
                    )
                    print(f"已添加任务: {cmd[:50]}")
                    
            except KeyboardInterrupt:
                self.scheduler.stop()
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def _print_help(self):
        help_text = """
可用命令:
  status          - 查看当前状态
  add <任务描述>   - 添加 LLM 任务
  add:type <类型> <描述> - 添加指定类型任务
  queue           - 查看待执行任务
  recent          - 查看最近完成的任务
  knowledge <关键词> - 搜索知识图谱
  optimize        - 触发深度反思
  examples        - 加载示例任务
  continuous      - 加载周期性任务
  pause / resume  - 暂停/恢复
  quit            - 退出

任务类型: echo, compute, llm_task, file_write, file_read,
          research, summarize, health_check, knowledge_query, self_improve

直接输入任意文字也会被当作 LLM 任务执行。
"""
        print(help_text)
    
    def _print_status(self):
        status = self.scheduler.get_status()
        print(f"\n{'='*40}")
        print(f"  Agent: {status['name']} v{status['version']}")
        print(f"  状态: {'运行中' if status['running'] else '已停止'}")
        print(f"  循环: {status['cycle_count']} 次")
        print(f"  队列: {status['queue_size']} 个任务")
        print(f"  已完成: {status['completed_tasks']} 个任务")
        print(f"  运行时间: {status['uptime_seconds']:.0f}s")
        print(f"  成功/失败: {status['stats']['tasks_completed']}/{status['stats']['tasks_failed']}")
        print(f"  自动恢复: {status['stats']['auto_recoveries']} 次")
        if self.kg:
            kg_stats = self.kg.get_stats()
            print(f"  知识节点: {kg_stats['total_nodes']}")
            print(f"  知识边: {kg_stats['total_edges']}")
        print(f"{'='*40}\n")
    
    def _cli_add_task(self, arg: str):
        if not arg:
            print("用法: add <任务描述>")
            return
        
        # 支持 add:type 格式
        if arg.startswith("type:"):
            parts = arg[5:].split(maxsplit=1)
            task_type = parts[0] if parts else "general"
            desc = parts[1] if len(parts) > 1 else ""
            self.scheduler.add_task(
                name=desc[:50] or task_type,
                description=desc,
                task_type=task_type
            )
            print(f"已添加 {task_type} 任务: {desc[:50]}")
        else:
            self.scheduler.add_task(
                name=arg[:50],
                description=arg,
                task_type="llm_task",
                params={"prompt": arg}
            )
            print(f"已添加任务: {arg[:50]}")
    
    def _print_queue(self):
        tasks = self.scheduler.get_pending_tasks()
        if not tasks:
            print("队列为空")
            return
        print(f"\n待执行任务 ({len(tasks)}):")
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. [{t['task_type']}] {t['name']} (优先级: {t['priority']})")
    
    def _print_recent(self):
        tasks = self.scheduler.get_recent_tasks(10)
        if not tasks:
            print("暂无已完成任务")
            return
        print(f"\n最近完成的任务:")
        for i, t in enumerate(tasks, 1):
            status = "✓" if t['status'] == 'success' else "✗"
            print(f"  {status} {i}. {t['name']} -> {t['status']}")
    
    def _cli_knowledge(self, query: str):
        if not query:
            print("用法: knowledge <关键词>")
            return
        results = self.kg.search_nodes(query, limit=10)
        if not results:
            print(f"未找到与 '{query}' 相关的知识")
            return
        print(f"\n找到 {len(results)} 条相关知识:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.get('category', '?')}] {r.get('title', '无标题')}")
            content = r.get('content', '')
            if content:
                print(f"     {content[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="极简全能持续运行 Agent")
    parser.add_argument('--config', '-c', default='config.json', help='配置文件路径')
    parser.add_argument('--mode', '-m', choices=['cli', 'daemon', 'api'], 
                       default='cli', help='运行模式 (默认: cli)')
    parser.add_argument('--no-examples', action='store_true', help='不加载示例任务')
    parser.add_argument('--port', '-p', type=int, default=8899, help='API 端口')
    parser.add_argument('--api-key', default='', help='LLM API Key')
    
    args = parser.parse_args()
    
    # 创建 Agent
    agent = SoloAgent(config_path=args.config)
    
    # 命令行覆盖配置
    if args.api_key:
        agent.config.llm.api_key = args.api_key
    if args.port:
        agent.config.webapi.port = args.port
    
    # 根据模式启动
    if args.mode == 'cli':
        agent.run_cli()
    elif args.mode == 'daemon':
        agent.run(with_api=True, load_examples=not args.no_examples)
    elif args.mode == 'api':
        agent.run(with_api=True, load_examples=not args.no_examples)


if __name__ == '__main__':
    main()
