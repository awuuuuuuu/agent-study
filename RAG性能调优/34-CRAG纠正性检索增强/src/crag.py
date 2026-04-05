import os
import asyncio
import logging
import sys
import nest_asyncio

# 【调试神器 1】解决 LlamaIndex 在 async 环境下的事件循环死锁问题！
nest_asyncio.apply()

# 【调试神器 2】开启底层网络请求日志，看看是不是卡在发往 OpenAI 的请求上了
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO,  # 开启 INFO 级别日志
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    Document, PromptTemplate, SummaryIndex
)

# 【修改 1】引入官方的 OpenAI LLM 和 Embedding 组件，删除了 DashScope
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.workflow import (
    Workflow, Event, step, Context, StartEvent, StopEvent
)
from llama_index.core.schema import NodeWithScore
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever

from dotenv import load_dotenv

load_dotenv()

# 【修改 2】极其干净的 OpenAI LLM 配置
Settings.llm = OpenAI(
    model="gpt-4", 
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# 【修改 3】配置 OpenAI 的 Embedding 模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

tavily_api_key = os.getenv("TAVILY_API_KEY")

# ============ CRAG工作流事件定义 ============
class PrepEvent(Event):
    """准备检索事件"""
    pass

class RetrieveEvent(Event):
    """检索事件 - CRAG第1步：从本地知识库检索"""
    retrieved_nodes: list[NodeWithScore]

class RelevanceEvalEvent(Event):
    """相关性评估事件 - CRAG第2步：评估检索质量"""
    relevant_results: list[str]

class TextExtractEvent(Event):
    """文本提取事件 - CRAG第3步：提取相关文本"""
    relevant_text: str

class QueryEvent(Event):
    """查询事件 - CRAG第4步：融合本地和外部知识"""
    relevant_text: str
    search_text: str

# ============ CRAG核心提示模板 ============
# CRAG关键组件1：相关性评估提示
RELEVANCY_PROMPT = PromptTemplate(
    template="""作为评估员，请评估检索到的文档与用户问题的相关性。

检索到的文档:
{context_str}

用户问题:
{query_str}

评估标准:
- 文档是否包含与问题相关的关键词或主题
- 评估不需要过于严格，主要目标是过滤明显不相关的内容

请回答 'yes'（相关）或 'no'（不相关）:"""
)

# CRAG关键组件2：查询转换提示（用于外部搜索优化）
TRANSFORM_QUERY_PROMPT = PromptTemplate(
    template="""请优化以下查询以提高搜索效果:

原始查询: {query_str}

请提供优化后的查询（仅返回查询内容）:"""
)

# ============ CRAG核心工作流 ============
class CorrectiveRAGWorkflow(Workflow):
    """
    CRAG (Corrective Retrieval-Augmented Generation) 工作流
    核心流程：检索 → 评估 → 纠正 → 融合
    """
    
    @step
    async def prepare_for_retrieval(self, ctx: Context, ev: StartEvent) -> PrepEvent | None:
        """CRAG步骤0：准备检索环境"""
        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None
        
        tavily_api_key: str | None = ev.get("tavily_api_key")
        index = ev.get("index")

        # 设置CRAG所需的核心组件
        await ctx.store.set("llm", Settings.llm)
        await ctx.store.set("index", index)

        # 设置外部搜索工具（CRAG纠正能力的关键）
        if tavily_api_key:
            await ctx.store.set("tavily_tool", TavilyToolSpec(api_key=tavily_api_key))
        
        await ctx.store.set("query_str", query_str)
        await ctx.store.set("retriever_kwargs", retriever_kwargs)
        
        return PrepEvent()
    
    @step
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent | None:
        """CRAG步骤1：从本地知识库检索相关文档"""
        query_str = await ctx.store.get("query_str")
        retriever_kwargs = await ctx.store.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.store.get("index", default=None)
        if not index:
            raise ValueError("CRAG需要预构建的向量索引")
        
        retriever: BaseRetriever = index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)

        await ctx.store.set("retrieved_nodes", result)
        
        return RetrieveEvent(retrieved_nodes=result)
    
    @step
    async def eval_relenance(self, ctx: Context, ev: RetrieveEvent) -> RelevanceEvalEvent:
        """CRAG步骤2：评估检索文档的相关性"""
        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.store.get("query_str")
        llm = await ctx.store.get("llm")

        relevancy_results = []
        for node in retrieved_nodes:
            resp = await llm.acomplete(
                RELEVANCY_PROMPT.format(
                    context_str=node.text, query_str=query_str
                )
            )
            relevancy_results.append(resp.text.lower().strip())

        await ctx.store.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)
    
    @step
    async def extract_relevant_texts(self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """CRAG步骤3：提取相关文本 - 过滤掉不相关的内容"""
        retrieved_nodes = await ctx.store.get("retrieved_nodes")
        relevancy_results = ev.relevant_results

        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if "yes" in result  # 稍微放宽一点正则容错，防止 LLM 输出 "Yes."
        ]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)
    
    @step
    async def transform_query(self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """CRAG步骤4：查询转换和外部搜索 - CRAG核心创新！"""
        relevant_text = ev.relevant_text
        relevancy_results = await ctx.store.get("relevancy_results")
        query_str = await ctx.store.get("query_str")
        search_text = ""

        # CRAG关键判断：如果有不相关的文档，触发外部搜索纠正
        # 同样放宽容错，只要包含 no 就触发重写
        if any("no" in res for res in relevancy_results):  
            llm = await ctx.store.get("llm")
            resp = await llm.acomplete(
                TRANSFORM_QUERY_PROMPT.format(query_str=query_str)
            )
            transformed_query_str = resp.text.strip()
            
            # 使用Tavily进行外部搜索
            tavily_tool = await ctx.store.get("tavily_tool")
            if tavily_tool:
                try:
                    search_results = tavily_tool.search(transformed_query_str, max_results=3)
                    search_text = "\n".join([result.text for result in search_results])
                except Exception as e:
                    print(f"外部搜索失败: {e}")
                    search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)
    
    @step
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """CRAG步骤5：生成最终结果 - 融合本地和外部知识"""
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.store.get("query_str")

        # 合并相关文本和搜索文本
        combined_text = relevant_text
        if search_text:
            combined_text += "\n\n外部搜索结果:\n" + search_text

        if not combined_text.strip():
            return StopEvent(result="抱歉，没有找到相关信息来回答您的问题。")

        # 使用合并的文本创建索引并查询
        documents = [Document(text=combined_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        result = query_engine.query(query_str)
        
        return StopEvent(result=result)
    
async def run_crag_demo():
    """
    CRAG演示函数 - 展示CRAG的核心能力
    """
    print("启动CRAG演示系统...")
    
    # 【修改 4】检查点改为检查 OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 加载文档
        documents = SimpleDirectoryReader("./data").load_data()
        print(f"成功加载 {len(documents)} 个文档")
        
        # 创建工作流
        workflow = CorrectiveRAGWorkflow()
        
        # 构建索引
        index = VectorStoreIndex.from_documents(documents)
        print("索引构建完成")
        
        # 测试查询1: 关于Llama2的问题（应该能在本地文档中找到答案）
        print("\n" + "="*50)
        print("测试查询1: Llama2是如何预训练的？")
        print("="*50)
        
        response1 = await workflow.run(
            query_str="Llama2是如何预训练的？",
            index=index,
            tavily_api_key=tavily_api_key,
        )
        print("回答1:")
        print(str(response1))
        
        # 测试查询2: 关于最新ChatGPT功能的问题（可能需要外部搜索）
        print("\n" + "="*50)
        print("测试查询2: 最新ChatGPT记忆功能是什么？")
        print("="*50)
        
        response2 = await workflow.run(
            query_str="最新ChatGPT记忆功能是什么？",
            index=index,
            tavily_api_key=tavily_api_key,
        )
        print("回答2:")
        print(str(response2))
        
        print("\nCRAG演示完成！")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

# 主函数
async def main():
    await run_crag_demo()

if __name__ == "__main__":
    asyncio.run(main())