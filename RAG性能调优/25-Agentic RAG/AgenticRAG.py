import os
from dotenv import load_dotenv

# 引入 LlamaIndex 最新核心组件
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 引入 OpenAI 专属集成包 (v0.10+ 标准)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.agent.openai import OpenAIAgent

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
custom_api_base = os.getenv("OPENAI_BASE_URL")

# =====================================================================
# 1. 全局配置 (使用最新 Settings 标准)
# =====================================================================
print("⚙️ 正在初始化 LlamaIndex 全局配置...")
Settings.llm = OpenAI(
    api_key=api_key, 
    api_base=custom_api_base,
    model="gpt-4o",
    temperature=0.1
)
Settings.embed_model = OpenAIEmbedding(
    api_key=api_key, 
    api_base=custom_api_base,
    model_name="text-embedding-3-small"
)

# =====================================================================
# 2. 文档读取与工具生成
# =====================================================================
print("📚 开始处理文档并生成专属检索工具...")

file_list = ["./data/crag.txt", "./data/selfrag.txt", "./data/kgrag.txt"]
name_list = ["c-rag", "self-rag", "kg-rag"]

tools = []

for file, name in zip(file_list, name_list):
    # 强制修正工具名称（OpenAI 规范：只允许字母、数字和下划线）
    safe_name = name.replace("-", "_")
    print(f"  -> 正在处理文档: {file} (工具前缀: {safe_name})")
    
    try:
        # 读取并切分
        docs = SimpleDirectoryReader(input_files=[file]).load_data()
        nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(docs)

        # 构建向量索引 (用于精准细节检索)
        vector_index = VectorStoreIndex(nodes)
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

        # 构建摘要索引 (用于宏观理解)
        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

        # 封装为标准 Tool (使用 ToolMetadata 规范元数据)
        vector_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"{safe_name}_vector_tool",
                description=f"用于查询 {name} 相关的具体技术细节、方法、公式或实验参数。"
            )
        )

        summary_tool = QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"{safe_name}_summary_tool",
                description=f"用于获取 {name} 的整体概述，包括核心思想、主要贡献和结论。"
            )
        )

        tools.extend([vector_tool, summary_tool])
        print(f"     ✓ {safe_name} 处理完成")
    except Exception as e:
        print(f"     ✗ 处理 {file} 时出错: {e}")

print(f"🎉 共成功创建 {len(tools)} 个检索工具！\n")

# =====================================================================
# 3. 初始化 OpenAI 原生智能体
# =====================================================================
try:
    print("🤖 正在唤醒 OpenAIAgent...")
    agent = OpenAIAgent.from_tools(
        tools=tools,
        llm=Settings.llm,
        verbose=True,  # 开启日志，观察 Function Calling 过程
        system_prompt=(
            "你是一个顶尖的 RAG 领域专家。你的主要职责是回答关于各类 RAG 技术的问题。\n"
            "请智能且自动地调用提供的工具获取背景信息。\n"
            "严格要求：必须基于工具返回的信息作答，严禁利用自身训练数据虚构信息！"
        )
    )
    print("✓ Agent 初始化成功！系统已就绪。\n")
except Exception as e:
    print("✗ Agent 初始化失败！详细报错如下：\n")
    import traceback
    traceback.print_exc()
    exit(1)


# =====================================================================
# 4. 运行主函数测试
# =====================================================================
def main():
    print("=== Agentic RAG 系统启动 ===")
    print("可用工具列表：")
    for tool in tools:
        print(f" - {tool.metadata.name}: {tool.metadata.description}")
    print("=" * 60 + "\n")
    
    # 示例查询
    queries = [
        "请对比 c-rag 和 self-rag 的核心技术路线，并说明它们在减少幻觉方面的差异。",
        "从 kg_rag 文档中提取核心产品说明，并整理成面向客户的简洁介绍。",
        "请总结这三种 RAG 方法各自的最优适用场景。"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"🗣️ 用户查询 {i}: {query}")
        print("-" * 60)
        try:
            response = agent.chat(query)
            print(f"\n💡 最终回答:\n{response}")
        except Exception as e:
            print(f"\n❌ 查询失败: {e}")
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()