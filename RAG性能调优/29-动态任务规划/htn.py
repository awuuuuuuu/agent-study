from typing import Dict, Any

# ================ 1. HTN 任务分解规则字典 (静态 SOP) ================
# 这里定义了业务的骨架。大模型和引擎都必须遵守这里的拆解逻辑。

DECOMPOSITION_RULES = {
    "审查合同": [
        {"name": "分析结构", "type": "原子任务"},
        {"name": "风险评估", "type": "复合任务"},  # 遇到复合任务，引擎会自动去查字典继续拆
        # 【特性】: 在字典配置中明确标记高危任务，要求执行前强制人工确认
        {"name": "生成报告", "type": "原子任务", "requires_human_approval": True} 
    ],
    "风险评估": [
        {"name": "检查排他性", "type": "原子任务"},
        {"name": "检查数据条款", "type": "原子任务"},
        {"name": "检查终止条件", "type": "原子任务"}
    ]
}

# ================ 2. 带有 HITL (Human-in-the-Loop) 的 HTN 引擎 ================

class HTNEngineHITL:
    """支持人工干预的分层任务网络执行引擎"""
    
    def __init__(self):
        self.depth = 0          # 记录当前的树深度，用于控制台缩进打印
        self.risk_count = 0     # 全局风险计数器

    def _ask_human(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        【人工干预核心接口】
        在生产环境中，这里应该是触发一个 Webhook，发企业微信通知，或者挂起数据库状态等待前端审批。
        这里使用 input() 阻塞主线程来模拟等待人工。
        """
        indent = "  " * self.depth
        print(f"\n{indent}⚠️ 【系统挂起：等待人工干预】")
        print(f"{indent}ℹ️ 提示: {prompt}")
        print(f"{indent}ℹ️ 当前上下文状态: {context}")
        
        # 阻塞等待人类输入指令
        decision = input(f"{indent}👉 请输入指令 (y:同意/继续 | n:拒绝/终止 | 其他文字:补充修改意见): ").strip()
        print(f"{indent}✅ 人工已介入，系统恢复执行流...\n")
        return decision

    def execute(self, task_name: str, context: Dict[str, Any] = None, subtask_config: Dict = None) -> Dict[str, Any]:
        """核心执行路由：判断任务类型并分发"""
        context = context or {}
        indent = "  " * self.depth
        subtask_config = subtask_config or {}

        # -------------------------------------------------------------
        # 干预埋点 1：基于静态规则的拦截 (执行高危任务前的强制审批)
        # -------------------------------------------------------------
        if subtask_config.get("requires_human_approval"):
            decision = self._ask_human(f"即将执行高危任务 [{task_name}]，是否允许？", context)
            
            if decision.lower() == 'n':
                print(f"{indent}❌ 人类拒绝了任务 [{task_name}]，当前分支终止。")
                return {"状态": "被人工终止"}
            elif decision.lower() != 'y':
                # 人类输入了具体的意见文字，将其“注入”到系统的上下文中
                context["人工附加要求"] = decision
                print(f"{indent}📝 已将人类干预意见写入上下文: {decision}")

        print(f"{indent}▶️ 执行任务: {task_name}")

        # 路由分发：复合任务去拆解，原子任务直接干活
        if task_name in DECOMPOSITION_RULES:
            return self._decompose_and_execute(task_name, context)
        else:
            return self._execute_primitive(task_name, context)
        
    def _decompose_and_execute(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """分解复合任务并递归执行其子任务"""
        indent = "  " * self.depth
        subtasks = DECOMPOSITION_RULES[task_name]
        results = {}
        
        self.depth += 1 # 树深度 +1

        for subtask in subtasks:
            # 递归调用自身，注意把 subtask_config 传进去以读取人工审批标志
            result = self.execute(subtask["name"], context, subtask_config=subtask)
            
            # 如果人工刚刚终止了该子任务，则向上抛出中断信号
            if result.get("状态") == "被人工终止":
                self.depth -= 1
                return {"状态": "被人工终止", "阶段": subtask["name"]}

            results[subtask["name"]] = result

            # 【动态感知】：捕获底层原子任务抛出的风险
            if result.get("发现风险"):
                self.risk_count += 1
                context["高风险"] = True
                context["风险计数"] = self.risk_count

                # -------------------------------------------------------------
                # 干预埋点 2：基于动态状态的异常抛出 (累计风险过高，呼叫人类)
                # -------------------------------------------------------------
                if self.risk_count >= 2 and not context.get("已确认风险处理"):
                    decision = self._ask_human(
                        f"系统在审查中累计发现 {self.risk_count} 个风险点！是否强行插入【深度风险分析】任务？", 
                        context
                    )
                    
                    if decision.lower() == 'y':
                        print(f"{indent}  [动态插入] 人类授权，开始执行深度风险分析...")
                        deep_analysis_result = self._execute_primitive("深度风险分析", context)
                        results["深度风险分析"] = deep_analysis_result
                    else:
                        print(f"{indent}  [人工指令] 忽略该风险警告，继续标准流程。")
                    
                    # 标记已处理，防止在这个复合任务的后续循环中被反复弹窗打断
                    context["已确认风险处理"] = True 

        self.depth -= 1 # 树深度回退
        return {"状态": "完成", "子任务结果": results}
    
    def _execute_primitive(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行最底层的原子任务 (实际业务逻辑写在这里)"""
        indent = "  " * self.depth
        
        if task_name == "检查排他性":
            if context.get("合同类型") == "SaaS":
                print(f"{indent}  🔍 触发 SaaS 专用排他性检查 -> ⚠️ 发现风险 (第5.2条)")
                return {"状态": "完成", "发现风险": True, "条款": "第5.2条"}
            print(f"{indent}  🔍 标准排他性检查 -> ✅ 无异常")
            return {"状态": "完成", "发现风险": False}
            
        elif task_name == "检查数据条款":
            if context.get("涉及个人数据"):
                print(f"{indent}  🔍 触发 GDPR 严格数据审查 -> ⚠️ 发现风险 (第8.3条)")
                return {"状态": "完成", "发现风险": True, "条款": "第8.3条"}
            print(f"{indent}  🔍 标准数据审查 -> ✅ 无异常")
            return {"状态": "完成", "发现风险": False}
            
        elif task_name == "深度风险分析":
            print(f"{indent}  🔬 [执行] 正在调用法务大模型进行深度扫描...")
            return {"状态": "完成", "分析结果": "法务介入，风险已标记并在库中备案。"}
            
        elif task_name == "生成报告":
            # 读取由于人工干预而产生的“动态注入意见”
            human_note = f" \n{indent}  [附加说明] 已应用人类干预意见: {context['人工附加要求']}" if context.get("人工附加要求") else ""
            
            if context.get("高风险"):
                print(f"{indent}  📄 [生成] 高风险详细评估报告 (Red Flag Report){human_note}")
            else:
                print(f"{indent}  📄 [生成] 常规放行审查报告 (Standard Report){human_note}")
            return {"状态": "完成", "报告生成": True}
        
        elif task_name == "分析结构":
            print(f"{indent}  🔍 分析合同总体结构 -> ✅ 完整")
            return {"状态": "完成", "发现风险": False}
            
        # 默认兜底
        print(f"{indent}  ⚙️ 执行基础任务: {task_name}")
        return {"状态": "完成", "发现风险": False}


# ================ 3. 运行测试演示 ================

if __name__ == "__main__":
    engine = HTNEngineHITL()
    
    print("=" * 70)
    print("🤖 企业级 Agent 架构演示: HTN 树状拆解 + 人工干预(HITL)")
    print("=" * 70)
    print("当前模拟任务：SaaS 合同审查 (将触发多处报警与拦截)")
    print("-" * 70)
    
    # 初始化全局系统上下文
    initial_context = {
        "合同类型": "SaaS",
        "涉及个人数据": True
    }
    
    # 启动顶层任务
    final_result = engine.execute("审查合同", initial_context)
    
    print("\n" + "=" * 70)
    print("🏁 任务流执行结束")
    print("最终输出状态:")
    import json
    # 剔除过于冗长的结果以保持终端整洁，仅打印核心状态
    status = final_result.get("状态")
    print(f"执行结果: {status}")