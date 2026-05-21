# -*- coding: utf-8 -*-
"""
极简全能持续运行 Agent - Web API 模块
第一性原理：API = 状态的窗口 + 控制的遥控器
"""

import logging
from flask import Flask, jsonify, request

logger = logging.getLogger("agent.webapi")


class WebAPI:
    """
    Web API - Agent 的对外接口
    提供状态查询、任务管理、知识操作等 RESTful 接口
    """
    
    def __init__(self, scheduler, knowledge_graph=None, executor=None, optimizer=None):
        self.scheduler = scheduler
        self.kg = knowledge_graph
        self.executor = executor
        self.optimizer = optimizer
        self.app = Flask(__name__)
        self._register_routes()
        logger.info("Web API 路由注册完成")
    
    def _register_routes(self):
        """注册所有 API 路由"""
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """获取 Agent 状态"""
            return jsonify({
                "success": True,
                "data": self.scheduler.get_status()
            })
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            status = self.scheduler.get_status()
            return jsonify({
                "success": True,
                "data": {
                    "status": "running" if status["running"] else "stopped",
                    "uptime": status["uptime_seconds"],
                    "cycle_count": status["cycle_count"],
                    "queue_size": status["queue_size"]
                }
            })
        
        # ---- 任务管理 ----
        
        @self.app.route('/api/tasks', methods=['GET'])
        def get_tasks():
            """获取任务列表"""
            task_type = request.args.get('type')  # pending, completed, recent
            try:
                limit = max(1, min(int(request.args.get('limit', 20)), 100))
            except (ValueError, TypeError):
                limit = 20

            if task_type == 'pending':
                tasks = self.scheduler.get_pending_tasks()
            elif task_type == 'completed':
                tasks = self.scheduler.get_recent_tasks(limit)
            else:
                tasks = self.scheduler.get_recent_tasks(limit)
            
            return jsonify({"success": True, "data": tasks, "count": len(tasks)})
        
        @self.app.route('/api/tasks', methods=['POST'])
        def add_task():
            """添加任务"""
            data = request.get_json(silent=True)
            if not data or not isinstance(data, dict):
                return jsonify({"success": False, "error": "请求体不能为空"}), 400
            
            task = self.scheduler.add_task(
                name=data.get('name', '未命名任务'),
                description=data.get('description', ''),
                task_type=data.get('task_type', 'general'),
                params=data.get('params', {}),
                priority=data.get('priority', 0)
            )
            
            return jsonify({
                "success": True,
                "data": task.to_dict(),
                "message": f"任务 '{task.name}' 已添加到队列"
            })
        
        @self.app.route('/api/tasks/batch', methods=['POST'])
        def add_tasks():
            """批量添加任务"""
            data = request.get_json(silent=True)
            if not data or 'tasks' not in data:
                return jsonify({"success": False, "error": "需要 tasks 数组"}), 400
            if not isinstance(data['tasks'], list) or len(data['tasks']) > 100:
                return jsonify({"success": False, "error": "tasks 必须为列表且不超过100条"}), 400
            
            count = self.scheduler.add_tasks(data['tasks'])
            return jsonify({
                "success": True,
                "data": {"added": count},
                "message": f"已添加 {count} 个任务"
            })
        
        # ---- Agent 控制 ----
        
        @self.app.route('/api/control/pause', methods=['POST'])
        def pause():
            self.scheduler.pause()
            return jsonify({"success": True, "message": "Agent 已暂停"})
        
        @self.app.route('/api/control/resume', methods=['POST'])
        def resume():
            self.scheduler.resume()
            return jsonify({"success": True, "message": "Agent 已恢复"})
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def stop():
            logger.warning("收到停止请求")
            self.scheduler.stop()
            return jsonify({"success": True, "message": "Agent 已停止"})
        
        # ---- 知识图谱 ----
        
        @self.app.route('/api/knowledge/search', methods=['GET'])
        def search_knowledge():
            if not self.kg:
                return jsonify({"success": False, "error": "知识图谱未启用"}), 503
            
            query = request.args.get('q', '')
            if not query or not query.strip():
                return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400
            category = request.args.get('category')
            try:
                limit = max(1, min(int(request.args.get('limit', 20)), 100))
            except (ValueError, TypeError):
                limit = 20
            
            results = self.kg.search_nodes(query, category=category, limit=limit)
            return jsonify({
                "success": True,
                "data": results,
                "count": len(results)
            })
        
        @self.app.route('/api/knowledge/stats', methods=['GET'])
        def knowledge_stats():
            if not self.kg:
                return jsonify({"success": False, "error": "知识图谱未启用"}), 503
            
            return jsonify({
                "success": True,
                "data": self.kg.get_stats()
            })
        
        @self.app.route('/api/knowledge/node/<node_id>', methods=['GET'])
        def get_node(node_id):
            if not self.kg:
                return jsonify({"success": False, "error": "知识图谱未启用"}), 503
            
            node = self.kg.get_node(node_id)
            if node:
                return jsonify({"success": True, "data": node})
            return jsonify({"success": False, "error": "节点不存在"}), 404
        
        @self.app.route('/api/knowledge/node', methods=['POST'])
        def add_knowledge():
            if not self.kg:
                return jsonify({"success": False, "error": "知识图谱未启用"}), 503
            
            data = request.get_json(silent=True)
            if not data or not isinstance(data, dict):
                return jsonify({"success": False, "error": "请求体不能为空"}), 400
            if not data.get('title'):
                return jsonify({"success": False, "error": "title 不能为空"}), 400
            node_id = self.kg.store_experience(
                title=data.get('title', ''),
                content=data.get('content', ''),
                category=data.get('category', 'general'),
                tags=data.get('tags', []),
                importance=data.get('importance', 0.5)
            )
            return jsonify({
                "success": True,
                "data": {"id": node_id},
                "message": "知识已添加"
            })
        
        @self.app.route('/api/knowledge/context', methods=['GET'])
        def get_context():
            if not self.kg:
                return jsonify({"success": False, "error": "知识图谱未启用"}), 503
            
            query = request.args.get('q', '')
            if not query or not query.strip():
                return jsonify({"success": False, "error": "搜索关键词不能为空"}), 400
            context = self.kg.get_relevant_context(query)
            return jsonify({"success": True, "data": {"context": context}})
        
        # ---- 优化 ----
        
        @self.app.route('/api/optimize/history', methods=['GET'])
        def optimization_history():
            if not self.optimizer:
                return jsonify({"success": False, "error": "优化器未启用"}), 503
            
            try:
                limit = max(1, min(int(request.args.get('limit', 20)), 100))
            except (ValueError, TypeError):
                limit = 20
            return jsonify({
                "success": True,
                "data": self.optimizer.get_optimization_history(limit)
            })
        
        @self.app.route('/api/optimize/reflect', methods=['POST'])
        def trigger_reflection():
            if not self.optimizer:
                return jsonify({"success": False, "error": "优化器未启用"}), 503
            
            result = self.optimizer.deep_reflection()
            return jsonify({
                "success": True,
                "data": {"reflection": result}
            })
        
        # ---- 统计 ----
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            status = self.scheduler.get_status()
            stats = {
                "agent": {
                    "cycles": status["cycle_count"],
                    "uptime": status["uptime_seconds"],
                    "queue_size": status["queue_size"],
                    "completed": status["completed_tasks"]
                },
                "executor": status.get("executor_stats", {}),
                "optimizer": status.get("optimizer_stats", {}),
            }
            
            if self.kg:
                stats["knowledge"] = self.kg.get_stats()
            
            return jsonify({"success": True, "data": stats})
        
        # ---- 根路径 ----
        
        @self.app.route('/', methods=['GET'])
        def index():
            return jsonify({
                "name": "极简全能持续运行 Agent",
                "version": "1.0.0",
                "status": "running" if self.scheduler.get_status()["running"] else "stopped",
                "endpoints": {
                    "状态": "/api/status",
                    "健康检查": "/api/health",
                    "任务列表": "/api/tasks",
                    "添加任务": "POST /api/tasks",
                    "知识搜索": "/api/knowledge/search?q=关键词",
                    "知识统计": "/api/knowledge/stats",
                    "优化历史": "/api/optimize/history",
                    "总统计": "/api/stats",
                    "暂停": "POST /api/control/pause",
                    "恢复": "POST /api/control/resume",
                    "停止": "POST /api/control/stop"
                }
            })

        @self.app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"API 异常: {type(e).__name__}: {e}", exc_info=True)
            return jsonify({"success": False, "error": "内部服务器错误"}), 500

    def run(self, host: str = "0.0.0.0", port: int = 8899, debug: bool = False):
        """启动 Web 服务"""
        logger.info(f"🌐 Web API 启动: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
