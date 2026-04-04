"""基于 LangGraph 与 LangChain 的长文本处理智能体"""
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# =============================================================================
# 状态定义
# 项目使用了 LangGraph 框架来构建工作流。
# LangGraph 是一个基于状态的工作流框架，它要求定义一个状态类来：
# 在不同的节点（函数）之间传递数据
# 跟踪整个工作流的执行状态
# 确保数据的类型安全和一致性
# 
# 状态定义明确了整个长文本生成流程中需要传递的数据：
# original_text: 原始输入文本
# chunks: 切分后的文本块
# summaries: 每个文本块的摘要
# planning_tree: 生成的文章结构树
# final_output: 最终生成的文章
# vectorstore: 向量数据库存储
# =============================================================================
class GenerationState(TypedDict):
    file_path: str
    original_text: str
    chunks: List[str]
    summaries: List[str]
    planning_tree: Dict
    final_output: str
    vector_store: FAISS

def load_node(state: GenerationState) -> GenerationState:
    """文档读取阶段"""
    print("=" * 60)
    print("📄 [加载阶段] 读取源文档")
    print("=" * 60)

    file_path = state["file_path"]
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        state["original_text"] = docs[0].page_content
        print(f"✅ 成功读取文档: {file_path}")
        print(f"📊 文档总长度: {len(state['original_text'])} 字符")
    except Exception as e:
        print(f"❌ 读取文档失败: {e}")
        state["original_text"] = "文档加载失败。"
        
    print("✅ 加载阶段完成\n")
    return state

def split_node(state: GenerationState) -> GenerationState:
    """分块阶段：使用 RecursiveCharacterTextSplitter"""
    print("=" * 60)
    print("🔄 [分块阶段] 开始智能文本切分")
    print("=" * 60)

    chunks = text_splitter.split_text(state["original_text"])
    state["chunks"] = chunks

    print(f"📊 切分统计:")
    print(f"   切分块数: {len(chunks)} 块")

    for i, chunk in enumerate(chunks, 1):
        chars = len(chunk)
        preview = chunk[:100].replace('\n', ' ') + "..." if len(chunk) > 100 else chunk
        print(f"   块 {i}: ({chars} 字符) | {preview}")
    
    print("✅ 分块阶段完成\n")
    return state

def summarize_and_memorize_node(state: GenerationState) -> GenerationState:
    print("=" * 60)
    print("🧠 [记忆阶段] 构建上下文记忆")
    print("=" * 60)
    
    summaries = []
    print("📝 正在生成摘要...")

    for i, chunk in enumerate(state["chunks"], 1):
        print(f"   处理块 {i}/{len(state['chunks'])}...", end=" ", flush=True)
        summary = generate_summary(chunk)
        summaries.append(summary)
        print("✅")
        print(f"      内容: {summary[:60]}...")

    state["summaries"] = summaries
    print(f"\n🔍 构建向量数据库...")
    state["vector_store"] = FAISS.from_texts(summaries, embedding=embeddings)
    print("✅ 向量数据库构建完成")
    print("✅ 记忆阶段完成\n")
    return state

def generate_summary(chunk: str) -> str:
    """生成精简摘要"""
    chunk_length = len(chunk)
    target_length = int(chunk_length * 0.3)

    prompt = f"""请对以下内容进行高度精简的摘要。要求：
1. 摘要长度尽量不超过{target_length}字符
2. 只保留最核心的观点和关键信息
3. 使用简洁的语言，避免冗余表达

待摘要内容：
{chunk}"""
    
    response = llm.invoke(prompt)
    summary = response.content

    if len(summary) > target_length:
        compress_prompt = f"请将以下摘要进一步压缩到{target_length}字符以内，只保留最关键的信息：\n{summary}"
        response = llm.invoke(compress_prompt)
        summary = response.content
        
    return summary
    
def planning_node(state: GenerationState) -> GenerationState:
    print("=" * 60)
    print("📋 [规划阶段] 构建精简文章结构")
    print("=" * 60)

    planning_tree = build_planning_tree(state["summaries"])
    state["planning_tree"] = planning_tree

    print("✅ 精简结构树生成完成")
    print(f"\n📖 大纲结构: {planning_tree.get('title', '未定义')}")

    for i, section in enumerate(planning_tree.get('sections', []), 1):
        print(f"   {i}. {section.get('title', '未知章节')}")
        
    print("✅ 规划阶段完成\n")
    return state

def build_planning_tree(summaries: List[str]) -> Dict:
    combined = "\n\n".join(f"Block {i+1}: {s}" for i, s in enumerate(summaries))
    prompt = f"""
    请根据以下文本块摘要，生成一份精简的综合报告结构大纲。
    
    要求：
    - 总共只生成3-4个主要章节，每章不超过1个合并段落
    - 将相关小节内容合并为综合性段落
    - 输出为严格JSON格式，不要包含任何其他文字
    
    摘要汇总：
    {combined}
    
    请只输出JSON，格式如下：
    {{
      "title": "报告主标题",
      "sections": [
        {{"title": "发展现状与技术基础"}},
        {{"title": "应用领域与实践案例"}},
        {{"title": "挑战问题与未来趋势"}}
      ]
    }}
    """

    response = llm.invoke(prompt)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
        
    try:
        parsed_json = json.loads(content.strip())
    except json.JSONDecodeError:
        parsed_json = {
            "title": "文档分析报告",
            "sections": [
                {"title": "核心技术与发展现状"},
                {"title": "应用实践与行业影响"},
                {"title": "挑战机遇与未来展望"}
            ]
        }
    return parsed_json

def generate_node(state: GenerationState) -> GenerationState:
    print("=" * 60)
    print("✍️ [生成阶段] 最终报告生成")
    print("=" * 60)

    tree = state["planning_tree"]
    content_parts = []

    if "title" in tree:
        content_parts.append(f"# {tree['title']}\n")

    sections = tree.get("sections", [])

    for i, section in enumerate(sections, 1):
        sec_title = section["title"]
        print(f"🔄 生成章节 {i}/{len(sections)}: {sec_title}")
        content_parts.append(f"## {sec_title}")

        context = retrieve_relevant_memory(sec_title, state["vector_store"])
        content = generate_section_content(sec_title, context)
        content_parts.append(content)

        if state["vectorstore"] is not None:
            state["vectorstore"].add_texts([content])

    state["final_output"] = "\n\n".join(content_parts)
    print("✅ 生成阶段完成\n")
    
    return state

def retrieve_relevant_memory(query: str, vector_store: FAISS, top_k: int = 3) -> str:
    if vector_store is None:
        return "向量存储不可用"
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n".join(d.page_content for d in docs)

def generate_section_content(title: str, context: str) -> str:
    prompt = f"""
    你是专业撰稿人。请根据参考上下文，撰写以下章节的综合性内容。
    
    # 上下文参考：
    {context}
    
    # 目标章节：
    {title}
    
    要求：
    1. 将相关内容合并为一个完整的综合段落
    2. 涵盖该主题的核心要点和关键信息
    3. 语言精炼，逻辑清晰，段落长度适中（200-400字）
    """
    response = llm.invoke(prompt)
    return response.content

def create_generation_workflow() -> StateGraph:
    workflow = StateGraph(GenerationState)

    workflow.add_node("load", load_node)
    workflow.add_node("split", split_node)
    workflow.add_node("summarize_and_memorize", summarize_and_memorize_node)
    workflow.add_node("plan", planning_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "load")
    workflow.add_edge("load", "split")
    workflow.add_edge("split", "summarize_and_memorize")
    workflow.add_edge("summarize_and_memorize", "plan")
    workflow.add_edge("plan", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

if __name__ == "__main__":

    sample_file = "data.txt"

    print("🚀 启动基于文档的长文本生成智能体")
    app = create_generation_workflow()
    
    # 初始状态只提供文件路径
    initial_state = {"file_path": sample_file}
    
    result = app.invoke(initial_state)

    print("=" * 60)
    print("🎉 执行完成！最终报告如下：")
    print("=" * 60)
    print(result["final_output"])

