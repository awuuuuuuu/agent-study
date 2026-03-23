from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding

import os
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True,
    temperature=0.1
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    api_key=DASHSCOPE_API_KEY,
    embed_batch_size=10
)

def demonstrate_sliding_window_splitter(documents, chunk_size, chunk_overlap):
    """
    演示 LlamaIndex 中保持句子完整性的滑动窗口切片。
    
    Args:
        documents (list[Document]): 待切分的文档列表。
        chunk_size (int): 每个切块的目标 Token 数量。
        chunk_overlap (int): 相邻切块之间重叠的 Token 数量。
    """
    print(f"\n{'='*50}")
    print(f"正在演示【滑动窗口切片】...")
    print(f"切块大小 (chunk_size): {chunk_size}")
    print(f"重叠大小 (chunk_overlap): {chunk_overlap}")
    print(f"{'='*50}\n")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print("\n--- 切分后生成的原始切块：---")
    print(f"文档被切分为 {len(nodes)} 个切块。")
    for i, node in enumerate(nodes, 1):
        content = node.get_content().strip()
        print(f"\n【切块 {i}】 (长度: {len(content)} 字符):")
        print("-" * 50)
        print(f"内容:\n\"{content}\"")
        print("-" * 50)

        print("\n--- 关键点：观察相邻切块的重叠部分 ---")
    if len(nodes) > 1:
        # 为了更好地展示重叠，我们只截取重叠部分的内容
        # 由于是句子级别的切分，重叠部分是完整的句子
        overlap_content_end_of_chunk1 = nodes[0].get_content()[-chunk_overlap:].strip()
        overlap_content_start_of_chunk2 = nodes[1].get_content()[:chunk_overlap].strip()
        print(f"切块 1 的末尾 ({chunk_overlap} 字符): \"...{overlap_content_end_of_chunk1}\"")
        print(f"切块 2 的开头 ({chunk_overlap} 字符): \"{overlap_content_start_of_chunk2}...\"")
        print(f"\n你可以看到，切块 1 的末尾与切块 2 的开头存在重叠，这就是 chunk_overlap 的作用。")
    else:
        print("文档太短，未能生成多个切块。请使用更长的文档以观察效果。")

    print(f"\n滑动窗口切片测试完成。")
    print(f"{'='*50}\n")

documents = [
    Document(
        text="""
        LlamaIndex 是一个用于构建 LLM 应用程序的数据框架。
        它提供了一套工具，可以帮助开发者将私有数据与大型语言模型（LLMs）连接起来，
        实现包括问答、检索增强生成（RAG）等功能。
        LlamaIndex 支持多种数据源，包括 PDF、数据库、API 等。
        其核心概念包括文档加载器、节点解析器、索引和查询引擎。

        文档加载器负责将各种格式和来源的数据摄取到 LlamaIndex 中。
        节点解析器随后将这些加载的文档分解成更小、更易于管理的单元，称为节点。
        这些节点通常是句子或段落，具体取决于解析策略。
        索引是构建在这些节点之上的数据结构，旨在实现高效存储和检索，
        通常涉及向量嵌入以进行语义搜索。
        最后，查询引擎促进了与索引数据的交互，允许用户提出问题
        并利用 LLM 和检索到的信息合成答案。

        --- 以下是与 LlamaIndex 主题不太直接相关的内容 ---

        此外，Python 作为一门通用编程语言，其简洁性和丰富的库生态使其在 AI 领域广受欢迎。
        例如，NumPy 和 Pandas 是数据处理的基础，它们提供了强大的工具用于数值操作和结构化数据。
        Scikit-learn 则提供了全面的机器学习算法套件，适用于分类、回归和聚类等任务。
        这些工具共同构成了数据科学家和 AI 从业者的强大工具箱，
        使他们能够高效地开发和部署复杂的 AI 模型。

        --- 以下是另一个相关但概念上独立的部分 ---

        句子窗口切片是一种高级的切片策略，它在每个切片中包含一个目标句子，
        并在其前后添加一定数量的“窗口”句子作为上下文。
        这种方法旨在检索时为 LLM 提供丰富的局部上下文，从而提高生成答案的连贯性。
        语义切片则尝试根据文本的语义内容来划分段落，
        而不是仅仅依靠固定的字符数或句子数量。
        它利用嵌入模型计算句子或短语之间的语义相似度，
        识别出主题或含义发生自然转变的断点。
        这两种高级方法都能有效提升 RAG 应用的召回和生成质量。
        选择正确的切片策略通常取决于数据的具体特征和预期的查询类型。
        """
    )
]

demonstrate_sliding_window_splitter(documents, chunk_size = 150, chunk_overlap = 50)