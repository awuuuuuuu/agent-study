import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    # model="gpt-4o",
    api_key=api_key,
    base_url=base_url
)

def query(user_prompt):
    """
    发送用户提示到 OpenAI API 并返回响应内容
    
    参数:
        user_prompt (str): 用户输入的提示内容
        
    返回:
        str: AI 的响应内容
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {str(e)}"
    
class ContextTemplateFactory:
    """上下文模板工厂类 - 负责创建和管理不同类型的上下文模板"""

    @staticmethod
    def create_customer_feedback_template():
        """创建客户反馈分析模板"""
        return {
            "name": "客户反馈分析",
            "template": """请从客户反馈中提取主要问题，参考以下分类标准：
            1. 物流问题：配送延迟、包装损坏、配送地址错误等
            2. 产品问题：质量缺陷、功能不符、规格错误等  
            3. 服务问题：客服响应慢、态度不好、解决方案不满意等
            4. 价格问题：价格变动、优惠券问题、退款问题等

            示例分析：
            原文："快递三天了还没到，客服说在路上，态度还挺好的"
            分析结果：物流延迟（主要问题）+ 客服响应快且态度好（正面反馈）

            现在请分析以下客户反馈：
            {user_input}""",
            "keywords": ["快递", "客服", "配送", "延迟", "态度", "服务", "物流", "反馈", "评价", "投诉", "建议"]
        }
    
    @staticmethod
    def create_technical_doc_template():
        """创建技术文档总结模板"""
        return {
            "name": "技术文档总结",
            "template": """请按照以下结构总结技术文档：
            1. 核心概念：提取3-5个关键技术概念
            2. 主要功能：列出核心功能特性
            3. 使用场景：说明适用的业务场景
            4. 注意事项：标注重要的限制或注意点

            参考格式：
            - 核心概念：[概念1], [概念2], [概念3]
            - 主要功能：[功能描述]
            - 使用场景：[场景描述]  
            - 注意事项：[注意点]

            请总结以下内容：
            {user_input}""",
            "keywords": ["技术", "文档", "API", "框架", "算法", "接口", "认证", "请求", "错误处理"]
        }
    
    @staticmethod
    def create_default_template():
        """创建默认通用模板"""
        return {
            "name": "通用总结",
            "template": """请对以下内容进行总结：
            {user_input}""",
            "keywords": []
        }
    
    @classmethod
    def get_all_templates(cls):
        """获取所有可用的模板"""
        return {
            "客户反馈分析": cls.create_customer_feedback_template(),
            "技术文档总结": cls.create_technical_doc_template(),
            "通用总结": cls.create_default_template()
        }
    
class DynamicContextInjector:
    """
    动态上下文注入器 - 使用模板方法模式定义处理流程
    
    处理流程：
    1. 初始化模板库
    2. 检测任务类型  
    3. 选择合适模板
    4. 注入上下文信息
    5. 执行AI查询
    """

    def __init__(self):
        """初始化：加载所有上下文模板"""
        self.context_templates = self._initialize_templates()
    
    def _initialize_templates(self):
        """步骤1：初始化模板库（模板方法的具体步骤）"""
        print("正在初始化上下文模板库...")
        templates = ContextTemplateFactory.get_all_templates()
        print(f"成功加载 {len(templates)} 个模板类型")
        return templates
    
    def _detect_task_type(self, user_input):
        """步骤2：检测任务类型（模板方法的具体步骤）"""
        print(f"正在分析输入内容：{user_input}...")

        user_input_lower = user_input.lower()

        for template_name, template_config in self.context_templates.items():
            if template_name == "通用总结":
                continue

            for keyword in template_config["keywords"]:
                if keyword in user_input_lower:
                    print(f"检测到关键词 '{keyword}'，匹配模板：{template_name}")
                    return template_name
        print("未找到特定关键词，使用通用总结模板")
        return "通用总结"
    
    def _select_template(self, task_type):
        """步骤3：选择合适的模板（模板方法的具体步骤）"""
        print(f"选择模板类型：{task_type}")

        selected_template = self.context_templates.get(task_type)
        if not selected_template:
            print("模板不存在，回退到通用模板")
            selected_template = self.context_templates["通用总结"]
        
        return selected_template
    
    def _inject_context(self, user_input, template_config):
        """步骤4：注入上下文信息（模板方法的具体步骤）"""
        print("正在注入上下文信息...")

        template_content = template_config["template"]
        enhanced_prompt = template_content.format(user_input=user_input)

        print(f"增强后提示词长度：{len(enhanced_prompt)} 字符")
        return enhanced_prompt

    def _execute_query(self, enhanced_prompt):
        """步骤5：执行AI查询（模板方法的具体步骤）"""
        print("正在调用AI模型...")
        print("-" * 50)
        
        result = query(enhanced_prompt)
        return result
    
    def process_with_context(self, user_input, task_type=None):
        """
        模板方法：定义完整的处理流程
        这是主要的算法骨架，按固定顺序执行各个步骤
        """
        print("开始动态上下文注入处理流程")
        print("=" * 60)
        
        try:
            if task_type is None:
                detected_type = self._detect_task_type(user_input)
            else:
                detected_type = task_type
                print(f"使用指定任务类型：{detected_type}")

            template_config = self._select_template(detected_type)

            enhanced_prompt = self._inject_context(user_input, template_config)

            result = self._execute_query(enhanced_prompt)

            print("处理流程完成")
            return result
            
        except Exception as e:
            print(f"处理过程中出现错误：{str(e)}")
            return f"处理失败：{str(e)}"
        
    def query_with_context(self, user_input, task_type=None):
        """向后兼容的方法名"""
        return self.process_with_context(user_input, task_type)
    
def demo_comparison():
    """演示基础Prompt vs 上下文融合Prompt的效果对比"""
    injector = DynamicContextInjector()

    test_cases = [
        "快递三天了还没到，客服说在路上，态度还挺好的，但是我很着急用这个东西",
        "这个API文档介绍了RESTful接口的使用方法，包括认证、请求格式和错误处理"
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n=== 测试用例 {i} ===")
        print(f"原始输入: {test_input}")
        
        # 基础Prompt
        print("\n【未融合Prompt结果】:")
        basic_result = query(f"总结这段话: {test_input}")
        print(basic_result)
        
        # 上下文融合Prompt  
        print("\n【上下文融合Prompt结果】:")
        context_result = injector.query_with_context(test_input)
        print(context_result)
        
        print("\n" + "="*80)

if __name__ == "__main__":    
    print("\n=== 动态上下文注入演示 ===")
    # 动态上下文注入演示
    demo_comparison()
