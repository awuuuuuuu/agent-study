import re
import textwrap
from typing import List, Callable
from pydantic import Field
from llama_index.core import Settings, Document
from llama_index.core.node_parser import NodeParser, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.utils import get_tokenizer

import os
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

class HybridNodeParser(NodeParser):
    primary_parser: NodeParser
    secondary_parser: NodeParser
    max_chunk_size: int = 1024
    tokenizer: Callable = Field(default_factory=get_tokenizer, exclude=True)

    def _parse_nodes(self, documents: List[Document], **kwargs) -> List[Document]:
        print("--- 开始执行【混合切分】... ---")

        primary_nodes = self.primary_parser.get_nodes_from_documents(documents)
        print(f"\n{'='*25} 第一步（语义切分）结果 {'='*25}")
        print(f"初步切分出 {len(primary_nodes)} 个语义段落。")
        for i, p_node in enumerate(primary_nodes, 1):
            print(f"\n【原始语义段落 {i}】 (大小: {len(self.tokenizer(p_node.get_content()))} tokens)")
            print("-" * 60)
            print(textwrap.indent(p_node.get_content().strip(), '  '))
            print("-" * 60)

        final_nodes = []
        print(f"\n{'='*25} 第二步（检查与二次切分）过程 {'='*25}")
        for i, node in enumerate(primary_nodes, 1):
            node_size = len(self.tokenizer(node.get_content()))
            print(f"\n>>> 正在检查【原始语义段落 {i}】 (大小: {node_size} tokens)...")

            if node_size <= self.max_chunk_size:
                print(f"  └── 结果: 大小合适 (<= {self.max_chunk_size} tokens)，直接采纳。")
                final_nodes.append(node)
            else:
                print(f"  └── 结果: 段落过大 (> {self.max_chunk_size} tokens)，将使用滑动窗口进行二次切分。")
                print("\n      【即将被切分的原始内容】")
                print("      " + "-"*50)
                print(textwrap.indent(node.get_content().strip(), '      | '))
                print("      " + "-"*50)

                sub_nodes = self.secondary_parser.get_nodes_from_documents([Document(text=node.get_content())])
                print(f"\n      【二次切分结果】: 被切分成了 {len(sub_nodes)} 个重叠的子切块。")
                for j, s_node in enumerate(sub_nodes, 1):
                    print(f"\n        【子切块 {i}.{j}】 (大小: {len(self.tokenizer(s_node.get_content()))} tokens)")
                    print("        " + "-"*40)
                    print(textwrap.indent(s_node.get_content().strip(), '        | '))
                    print("        " + "-"*40)

                final_nodes.extend(sub_nodes)

        print("\n--- 【混合切分】完成！---")
        return final_nodes

    @classmethod
    def from_defaults(cls, **kwargs):
        raise NotImplementedError("请直接实例化此类，不要使用 from_defaults")

Settings.llm = OpenAILike(
    model = "qwen-plus",
    api_base = DASHSCOPE_BASE_URL,
    api_key = DASHSCOPE_API_KEY,
    is_chat_model = True,
    temperature = 0.1
)

Settings.embed_model = DashScopeEmbedding(
    model_name = "text-embedding-v4",
    api_key = DASHSCOPE_API_KEY,
    embed_batch_size = 10
)

def chinese_sentence_tokenizer(text: str) -> list[str]:
    sentences = re.findall(r'[^。！？.!?]+[。！？.!?]', text)
    return sentences

# --- 准备长文档 ---
long_document = Document(
    text="""
    LlamaIndex 是一个用于构建、评估和部署高级 RAG（检索增强生成）应用程序的数据框架。它的核心使命是帮助开发者将私有或领域特定的数据与大型语言模型（LLMs）无缝连接。通过提供数据连接器、索引结构、查询引擎和评估工具，LlamaIndex 极大地简化了从原始数据到生产级 RAG 应用的全过程。它支持上百种数据源，包括 PDF 文件、数据库、Notion 页面以及各种 API，确保了数据接入的灵活性。

    在技术架构上，LlamaIndex 的设计哲学是模块化和可扩展性。其核心组件包括文档加载器（Document Loaders）、节点解析器（Node Parsers）、索引（Indices）、检索器（Retrievers）和查询引擎（Query Engines）。文档加载器负责从不同来源摄取数据。节点解析器（如我们正在讨论的这些）将加载的文档分解成更小、更易于管理的“节点”（Nodes），这是构建索引的基础单元。索引结构，特别是向量索引，利用嵌入模型将节点转换为高维向量，从而实现高效的语义搜索和检索。检索器则根据用户查询从索引中找出最相关的节点。最后，查询引擎协调检索器和 LLM，生成最终的、富有上下文的答案。这套架构使得开发者可以像搭乐高一样，自由组合和定制自己的 RAG 管道。

    评估是 LlamaIndex 另一个强大的功能。它提供了一整套工具来量化 RAG 应用的性能，包括答案的忠实度（Faithfulness）、相关性（Relevancy）以及上下文检索的准确率和召回率。通过这些量化指标，开发者可以系统地调试和优化他们的应用，而不仅仅是凭感觉。
    """
)

semantic_parser = SemanticSplitterNodeParser(
    buffer_size = 1,
    breakpoint_percentile_threshold = 95,
    sentence_splitter = chinese_sentence_tokenizer,
    embed_model = Settings.embed_model
)

window_spiltter = SentenceSplitter(
    chunk_size = 256,
    chunk_overlap = 50
)

hybrid_parser = HybridNodeParser(
    primary_parser = semantic_parser,
    secondary_parser = window_spiltter,
    max_chunk_size = 300,
    tokenizer = get_tokenizer()
)

final_nodes = hybrid_parser.get_nodes_from_documents([long_document])

# --- 打印最终结果  ---
print(f"\n{'='*25} 最终生成的切块列表 {'='*25}")
print(f"切块总数: {len(final_nodes)}")
for i, node in enumerate(final_nodes, 1):
    content = node.get_content().strip()
    print(f"\n【最终切块 {i}】:")
    print("-" * 50)
    print(textwrap.indent(content, '  '))
    print("-" * 50)