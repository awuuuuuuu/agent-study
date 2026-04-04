import os
import time
from typing import TypedDict, List, Dict, Any, Optional, Literal
from enum import Enum
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field as PydanticField
from llama_index.core.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 配置 LlamaIndex
Settings.llm = LlamaIndexOpenAI(
    api_key=api_key,
    api_base=base_url,
    model="gpt-4o"
)
Settings.embed_model = OpenAIEmbedding(
    api_key=api_key,
    api_base=base_url,
    model="text-embedding-ada-002"
)


# ======================
# 0. LLM 结构化输出 Schema（Pydantic）
# ======================

class TaskPlanItem(BaseModel):
    """单个任务节点的结构化描述，供 LLM structured_predict 使用"""
    task_id: str                                              = PydanticField(description="任务唯一ID，如 task_1")
    tool_name: Literal["stock_price", "news", "sentiment"]   = PydanticField(description="调用的工具名称")
    tool_type: Literal["data_retrieval", "analysis", "generation"] = PydanticField(description="工具类型")
    params: Dict[str, Any]                                   = PydanticField(description="工具调用参数")
    dependencies: List[str]                                  = PydanticField(description="依赖的 task_id 列表，无依赖则为空列表")
    priority: int                                            = PydanticField(description="优先级，1最高")
    max_retries: int                                         = PydanticField(description="最大重试次数")
    timeout: float                                           = PydanticField(description="超时秒数")
    is_critical: bool                                        = PydanticField(description="是否关键任务，失败则触发早退")

class TaskPlan(BaseModel):
    """完整任务规划，LLM 必须严格按此 Schema 返回"""
    tasks: List[TaskPlanItem] = PydanticField(description="有序任务列表，依赖靠后")

# ======================
# 1. 核心数据结构定义
# ======================
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ToolType(Enum):
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    GENERATION = "generation"

@dataclass
class PerformanceMetrics:
    execution_time: float
    cost_estimate: float
    memory_usage: float
    success_rate: float

@dataclass
class ToolExecutionResult:
    tool_name: str
    status: TaskStatus
    result: Any
    performance: PerformanceMetrics
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    optimization_suggestions: List[str] = field(default_factory=list)

@dataclass
class TaskNode:
    task_id: str
    tool_name: str
    tool_type: ToolType
    params: Dict[str, Any]
    dependencies: List[str]
    priority: int
    max_retries: int
    timeout: float
    is_critical: bool
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[ToolExecutionResult] = None
    retry_count: int = 0

class SystemState(TypedDict):
    user_question: str
    task_dag: List[TaskNode]
    execution_queue: List[str]
    completed_tasks: Dict[str, ToolExecutionResult]
    failed_tasks: Dict[str, ToolExecutionResult]
    system_load: float
    available_tools: List[str]
    performance_history: List[PerformanceMetrics]
    stock_data: str
    news_data: str
    sentiment_analysis: str
    final_recommendation: str
    should_continue: bool
    early_exit_triggered: bool
    current_phase: str
    error_count: int
    monitoring_metrics: Dict[str, Any]

# ======================
# 2. Tool Agent 实现层
# ======================

class BaseToolAgent:
    def __init__(self, name: str, tool_type: ToolType):
        self.name = name
        self.tool_type = tool_type
        self.execution_history: List[ToolExecutionResult] = []
    
    def execute(self, params: Dict[str, Any], timeout: float = 30.0) -> ToolExecutionResult:
        start_time = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_core, params)
                try:
                    result = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    raise TimeoutError(f"任务超时（{timeout}s）")
            
            execution_time = time.time() - start_time
            performance = PerformanceMetrics(
                execution_time=execution_time,
                cost_estimate=self._estimate_cost(params),
                memory_usage=self._get_memory_usage(),
                success_rate=self._calculate_success_rate()
            )

            exec_result = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.COMPLETED,
                result=result,
                performance=performance,
                optimization_suggestions=self._generate_optimization_suggestions(performance)
            )
        except Exception as e:
            execution_time = time.time() - start_time
            exec_result = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(execution_time, 0, 0, 0),
                error_code="TIMEOUT_ERROR" if isinstance(e, TimeoutError) else "EXECUTION_ERROR",
                error_message=str(e)
            )

        self.execution_history.append(exec_result)
        return exec_result

    def _execute_core(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError
    
    def _estimate_cost(self, params: Dict[str, Any]) -> float:
        return 0.01

    def _get_memory_usage(self) -> float:
        return 10.0
    
    def _calculate_success_rate(self) -> float:
        if not self.execution_history:
            return 1.0
        successful = sum(1 for r in self.execution_history if r.status == TaskStatus.COMPLETED)
        return successful / len(self.execution_history)

    def _generate_optimization_suggestions(self, performance: PerformanceMetrics) -> List[str]:
        suggestions = []
        if performance.execution_time > 5.0:
            suggestions.append("考虑增加缓存机制以减少执行时间")
        if performance.memory_usage > 100.0:
            suggestions.append("优化内存使用，考虑分批处理")
        return suggestions
    
class StockPriceAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("stock_price_agent", ToolType.DATA_RETRIEVAL)
    
    def _execute_core(self, params: Dict[str, Any]) -> str:
        ticker = params.get("ticker", "UNKNOWN")
        time.sleep(0.5)
        return f"{ticker} 过去7天平均价格为 245.6 美元，上涨 5.3%。市值约7800亿美元。"
    
class NewsAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("news_agent", ToolType.DATA_RETRIEVAL)
    
    def _execute_core(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")

        try:
            search = TavilySearch(max_results=3)
            results = search.invoke(f"{query} latest news")

            combined = " ".join(r.get("content", "") for r in results)
            return combined[:2000]
        except Exception as e:
            print(f"  [WARN] Tavily 搜索失败: {e}，使用降级数据")
            return "Tesla最新财报显示Q3营收234亿美元，同比增长7.8%。但面临中国市场竞争加剧，股价波动较大。分析师对其自动驾驶技术进展持谨慎乐观态度。"

class SentimentAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("sentiment_agent", ToolType.ANALYSIS)

    def _execute_core(self, params: Dict[str, Any]) -> str:
        text = params.get("text", "")
        if not text.strip():
            return "中性 - 无有效文本进行分析"

        prompt = f"""
        请分析以下文本的情感倾向，并给出详细评分：
        文本：{text}

        请按以下格式输出：
        - 整体情感：正面/负面/中性
        - 情感强度：1-10分
        - 关键词：列出影响情感的关键词
        - 风险提示：如有负面情感，请说明主要风险点
        """
        response = Settings.llm.complete(prompt)
        return response.text

# ======================
# 3. Top Agent 核心调度层
# ======================

class MonitoringDashboard:
    """实时监控看板"""
    def __init__(self):
        self.task_metrics: Dict[str, ToolExecutionResult] = {}
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "total_cost": 0.0
        }

    def update_task_status(self, task_id: str, result: ToolExecutionResult):
        self.task_metrics[task_id] = result
        self.system_metrics["total_tasks"] += 1
        if result.status == TaskStatus.COMPLETED:
            self.system_metrics["completed_tasks"] += 1
        elif result.status == TaskStatus.FAILED:
            self.system_metrics["failed_tasks"] += 1
        self.system_metrics["total_cost"] += result.performance.cost_estimate
        total_time = sum(r.performance.execution_time for r in self.task_metrics.values())
        self.system_metrics["average_execution_time"] = total_time / len(self.task_metrics)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        return {
            "system_metrics": self.system_metrics,
            "task_details": {tid: asdict(r) for tid, r in self.task_metrics.items()},
            "success_rate": self.system_metrics["completed_tasks"] / max(1, self.system_metrics["total_tasks"]),
        }

    def restore_from_state(self, metrics: Dict[str, Any]):
        if metrics:
            self.system_metrics = metrics.get("system_metrics", self.system_metrics)

    def to_state(self) -> Dict[str, Any]:
        return self.get_dashboard_summary()
    
class TopAgent:
    def __init__(self):
        self.tool_agents = {
            "stock_price": StockPriceAgent(),
            "news": NewsAgent(),
            "sentiment": SentimentAgent()
        }
        self.monitoring_dashboard = MonitoringDashboard()
    
    _PLAN_PROMPT = PromptTemplate(
        "你是投资分析系统的 Top Agent，请根据用户问题制定任务执行计划。\n"
        "用户问题：{user_question}\n\n"
        "可用工具：\n"
        "- stock_price：获取股票价格数据（关键任务，高优先级）\n"
        "- news：搜索相关新闻（可降级，中优先级）\n"
        "- sentiment：对新闻做情感分析（依赖 news 的输出，低优先级）\n\n"
        "注意：sentiment 的 params.text 请填写 {{news_data}} 作为占位符，"
        "表示运行时注入 news 任务的结果。"
    )

    def plan_tasks(self, user_question: str) -> List[TaskNode]:
        """智能任务规划：用 structured_predict 约束 LLM 输出格式，无需手动解析 JSON"""
        try:
            plan: TaskPlan = Settings.llm.stream_structured_predict(
                TaskPlan,
                self._PLAN_PROMPT,
                user_question=user_question
            )
            return [
                TaskNode(
                    task_id=item.task_id,
                    tool_name=item.tool_name,
                    tool_type=ToolType(item.tool_type),
                    params=item.params,
                    dependencies=item.dependencies,
                    priority=item.priority,
                    max_retries=item.max_retries,
                    timeout=item.timeout,
                    is_critical=item.is_critical
                )
                for item in plan.tasks
            ]
        except Exception as e:
            print(f"[WARN] structured_predict 规划失败（{e}），使用默认计划")
            return self._get_default_plan()
        
    def _get_default_plan(self) -> List[TaskNode]:
        return [
            TaskNode("task_1", "stock_price", ToolType.DATA_RETRIEVAL,
                     {"ticker": "TSLA"}, [], 1, 3, 10.0, True),
            TaskNode("task_2", "news", ToolType.DATA_RETRIEVAL,
                     {"query": "Tesla TSLA"}, [], 2, 2, 15.0, False),
            TaskNode("task_3", "sentiment", ToolType.ANALYSIS,
                     {"text": "{{news_data}}"}, ["task_2"], 3, 2, 8.0, False)
        ]
    
    def dynamic_dispatch(self, state: SystemState) -> Optional[TaskNode]:
        """动态任务分发：按依赖满足情况 + 优先级选择下一个任务"""
        ready_tasks = []
        for task in state["task_dag"]:
            if task.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                deps_satisfied = all(
                    dep_id in state["completed_tasks"]
                    for dep_id in task.dependencies
                )
                if deps_satisfied:
                    ready_tasks.append(task)
        if not ready_tasks:
            return None
        
        ready_tasks.sort(key=lambda t: (t.priority, -t.timeout))

        if state["system_load"] > 0.8:
            critical_tasks = [t for t in ready_tasks if t.is_critical]
            if critical_tasks:
                return critical_tasks[0]

        return ready_tasks[0]
    
    def execute_task(self, task: TaskNode, state: SystemState) -> ToolExecutionResult:
        """执行单个任务（含参数依赖注入 + 超时控制）"""
        processed_params = self._process_task_params(task.params, state)

        agent = self.tool_agents.get(task.tool_name)
        if not agent:
            return ToolExecutionResult(
                tool_name=task.tool_name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(0, 0, 0, 0),
                error_code="AGENT_NOT_FOUND",
                error_message=f"Tool agent '{task.tool_name}' not found"
            )
        task.status = TaskStatus.RUNNING

        result = agent.execute(processed_params, timeout=task.timeout)
        self.monitoring_dashboard.update_task_status(task.task_id, result)
        return result
    
    def _process_task_params(self, params: Dict[str, Any], state: SystemState) -> Dict[str, Any]:
        """依赖注入：将 {{news_data}} 等占位符替换为 State 中的实际数据"""
        dep_map = {
            "news_data": state.get("news_data", ""),
            "stock_data": state.get("stock_data", ""),
            "sentiment_analysis": state.get("sentiment_analysis", ""),
        }

        processed = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                dep_key = value[2:-2]
                processed[key] = dep_map.get(dep_key, "")
            else:
                processed[key] = value
        print(f"  处理任务参数: {processed}")
        return processed
    
    def should_retry_task(self, task: TaskNode, result: ToolExecutionResult) -> bool:
        """重试判断基于 TaskNode.retry_count"""
        if result.status != TaskStatus.FAILED:
            return False
        if task.retry_count >= task.max_retries:
            return False
        
        if not task.is_critical and task.retry_count > 1:
            return False
        return True
    
    def check_early_exit(self, state: SystemState) -> bool:
        """Early Exit：关键任务失败或错误数过多时提前终止"""
        critical_failed = any(
            task.is_critical and task.status == TaskStatus.FAILED
            for task in state["task_dag"]
        )
        return critical_failed or state["error_count"] > 3

    def generate_final_recommendation(self, state: SystemState) -> str:
        """基于已收集数据生成最终投资建议"""
        completed_count = len(state["completed_tasks"])
        data_integrity = "高" if completed_count >= 2 else "中" if completed_count >= 1 else "低"

        prompt = f"""
        基于以下分析结果，请给出专业的投资建议：

        股价数据：{state.get('stock_data', '数据获取失败')}
        新闻分析：{state.get('news_data', '数据获取失败')[:200]}...
        情感分析：{state.get('sentiment_analysis', '分析失败')}

        系统执行状态：
        - 完成任务数：{completed_count}
        - 失败任务数：{len(state['failed_tasks'])}
        - 数据完整性：{data_integrity}

        请提供：
        1. 明确的投资建议（买入/观望/卖出）
        2. 风险评估（高/中/低）
        3. 建议理由（3-5点）
        4. 数据可信度评估
        """
        response = Settings.llm.complete(prompt)
        return response.text
    
# ======================
# 4. 工作流编排层
# ======================

def initialize_system(state: SystemState):
    """系统初始化：规划任务 DAG"""
    top_agent = TopAgent()
    task_dag = top_agent.plan_tasks(state["user_question"])
    return {
        "task_dag": task_dag,
        "execution_queue": [task.task_id for task in task_dag],
        "completed_tasks": {},
        "failed_tasks": {},
        "system_load": 0.3,
        "available_tools": ["stock_price", "news", "sentiment"],
        "performance_history": [],
        "should_continue": True,
        "early_exit_triggered": False,
        "current_phase": "planning_complete",
        "error_count": 0,
        "monitoring_metrics": {}
    }

def execute_next_task(state: SystemState):
    """执行下一个任务节点"""
    top_agent = TopAgent()
    top_agent.monitoring_dashboard.restore_from_state(state.get("monitoring_metrics", {}))

    if top_agent.check_early_exit(state):
        return {
            "early_exit_triggered": True,
            "should_continue": False,
            "current_phase": "early_exit"
        }

    next_task = top_agent.dynamic_dispatch(state)
    if not next_task:
        return {
            "should_continue": False,
            "current_phase": "all_tasks_completed"
        }

    result = top_agent.execute_task(next_task, state)
    next_task.status = result.status
    next_task.result = result

    updates: Dict[str, Any] = {"error_count": state["error_count"]}

    if result.status == TaskStatus.COMPLETED:
        updates["completed_tasks"] = {**state["completed_tasks"], next_task.task_id: result}
        if next_task.tool_name == "stock_price":
            updates["stock_data"] = result.result
        elif next_task.tool_name == "news":
            updates["news_data"] = result.result
        elif next_task.tool_name == "sentiment":
            updates["sentiment_analysis"] = result.result

    elif result.status == TaskStatus.FAILED:
        # [优化1] 重试计数累加到 TaskNode 上
        if top_agent.should_retry_task(next_task, result):
            next_task.status = TaskStatus.RETRYING
            next_task.retry_count += 1
            updates["error_count"] = state["error_count"] + 1
        else:
            updates["failed_tasks"] = {**state["failed_tasks"], next_task.task_id: result}
            updates["error_count"] = state["error_count"] + 1
            if not next_task.is_critical:
                next_task.status = TaskStatus.SKIPPED

    pending_tasks = [t for t in state["task_dag"]
                     if t.status in (TaskStatus.PENDING, TaskStatus.RETRYING)]
    updates["should_continue"] = len(pending_tasks) > 0 and not updates.get("early_exit_triggered", False)
    # [优化2] 将监控数据序列化回 State
    updates["monitoring_metrics"] = top_agent.monitoring_dashboard.to_state()

    return updates

def generate_final_report(state: SystemState):
    """生成最终投资报告"""
    top_agent = TopAgent()
    
    top_agent.monitoring_dashboard.restore_from_state(state.get("monitoring_metrics", {}))
    recommendation = top_agent.generate_final_recommendation(state)
    dashboard_summary = top_agent.monitoring_dashboard.get_dashboard_summary()
    return {
        "final_recommendation": recommendation,
        "current_phase": "completed",
        "monitoring_metrics": dashboard_summary
    }

def should_continue_execution(state: SystemState) -> str:
    if state.get("early_exit_triggered", False):
        return "generate_report"
    elif state.get("should_continue", True):
        return "execute_task"
    else:
        return "generate_report"
    
# ======================
# 5. 工作流构建
# ======================
workflow = StateGraph(SystemState)
workflow.add_node("initialize", initialize_system)
workflow.add_node("execute_task", execute_next_task)
workflow.add_node("generate_report", generate_final_report)
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "execute_task")
workflow.add_conditional_edges(
    "execute_task",
    should_continue_execution,
    {
        "execute_task": "execute_task",
        "generate_report": "generate_report"
    }
)
workflow.add_edge("generate_report", END)
app = workflow.compile()

# ======================
# 6. 主程序执行
# ======================

if __name__ == "__main__":
    initial_state = {
        "user_question": "特斯拉（TSLA）最近表现如何？值得买入吗？",
        "task_dag": [],
        "execution_queue": [],
        "completed_tasks": {},
        "failed_tasks": {},
        "system_load": 0.0,
        "available_tools": [],
        "performance_history": [],
        "stock_data": "",
        "news_data": "",
        "sentiment_analysis": "",
        "final_recommendation": "",
        "should_continue": True,
        "early_exit_triggered": False,
        "current_phase": "initialized",
        "error_count": 0,
        "monitoring_metrics": {}
    }

    try:
        print("启动增强版多智能体投资分析系统...")
        print("=" * 60)

        result = app.invoke(initial_state)

        print("\n=== 投资分析报告 ===")
        print(f"问题: {result['user_question']}")
        print(f"执行阶段: {result['current_phase']}")

        print(f"\n数据收集结果:")
        print(f"股价数据: {result.get('stock_data', '未获取')}")
        news = result.get('news_data', '')
        print(f"新闻数据: {news[:100]}..." if news else "新闻数据: 未获取")
        print(f"情感分析: {result.get('sentiment_analysis', '未完成')}")

        print(f"\n最终建议:")
        print(result.get('final_recommendation', '无法生成建议'))

        print(f"\n系统执行统计:")
        dashboard = result.get('monitoring_metrics', {})
        if dashboard:
            metrics = dashboard.get('system_metrics', {})
            print(f"- 总任务数: {metrics.get('total_tasks', 0)}")
            print(f"- 完成任务数: {metrics.get('completed_tasks', 0)}")
            print(f"- 失败任务数: {metrics.get('failed_tasks', 0)}")
            print(f"- 成功率: {dashboard.get('success_rate', 0):.2%}")
            print(f"- 平均执行时间: {metrics.get('average_execution_time', 0):.2f}秒")
            print(f"- 总成本估算: ${metrics.get('total_cost', 0):.4f}")
        else:
            print(f"- 完成任务: {len(result.get('completed_tasks', {}))}")
            print(f"- 失败任务: {len(result.get('failed_tasks', {}))}")

        print(f"\n错误统计: {result.get('error_count', 0)} 个错误")
        if result.get('early_exit_triggered'):
            print("系统触发了早期退出机制")

    except Exception as e:
        print(f"系统执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
