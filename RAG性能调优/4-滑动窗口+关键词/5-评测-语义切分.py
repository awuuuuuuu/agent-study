import re
import numpy as np
import matplotlib.pyplot as plt
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.utils import get_tokenizer

import os
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

Settings.llm = OpenAILike(
    model = "qwen-plus",
    api_base = DASHSCOPE_BASE_URL,
    api_key = DASHSCOPE_API_KEY,
    is_chat_model = True,
    temperature = 0.1
)
Settings.embed_model = DashScopeEmbedding(
    model = "text-embedding-v4",
    api_key = DASHSCOPE_API_KEY,
    embed_batch_size = 10
)

llama_tokenizer = get_tokenizer()

def chinese_sentence_tokenizer(text: str) -> list[str]:
    sentences = re.findall(r'[^。！？…\n]+[。！？…\n]?', text)
    return [s.strip() for s in sentences if s.strip()]

long_text = (
    "大语言模型（LLM）是人工智能领域的一项革命性技术。其发展可以追溯到早期的神经网络和自然语言处理研究。最初的模型，如循环神经网络（RNN），在处理长序列文本时遇到了梯度消失的问题。随后，长短期记忆网络（LSTM）在一定程度上缓解了这个问题，但仍然存在局限性。\n\n"
    "真正的突破来自于2017年提出的Transformer架构。Transformer模型完全基于自注意力机制（Self-Attention），能够并行处理文本序列，极大地提高了训练效率和模型性能。这为构建更大、更深的模型（如BERT和GPT系列）奠定了基础。BERT使用双向编码器，在理解上下文方面表现出色，而GPT则采用自回归解码器，在文本生成方面尤为强大。\n\n"
    "如今，LLM的应用已经渗透到各行各业。它们不仅能进行智能问答、内容创作和代码生成，还在医疗、金融和教育等领域展现出巨大潜力。例如，在医疗领域，LLM可以帮助医生分析病历、提供诊断建议。在教育领域，它可以为学生提供个性化的辅导和学习资源。\n\n"
    "然而，LLM的发展也伴随着挑战。模型的偏见、幻觉问题以及高昂的计算成本都是亟待解决的难题。未来的研究方向可能包括提高模型的效率、增强其可解释性，并建立更负责任的人工智能治理框架。确保技术的普惠和安全是所有研究者共同的责任。"
)
document = Document(text = long_text)

def plot_similarity_and_chunks(splitter: SemanticSplitterNodeParser, title: str):
    sentences = chinese_sentence_tokenizer(document.get_content())

    if len(sentences) < 2:
        print(f"错误：只找到了 {len(sentences)} 个句子，无法计算句子间的相似度。")
        return

    print(f"正在为 {len(sentences)} 个句子生成嵌入向量...")
    embeddings = Settings.embed_model.get_text_embedding_batch(sentences, show_progress=True)

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    similarities = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]
    breakpoint_threshold_value = np.percentile(similarities, splitter.breakpoint_percentile_threshold)
    print(f"计算出的相似度阈值为: {breakpoint_threshold_value:.4f}")

    # --- 可视化 ---
    plt.figure(figsize=(12, 6))
    plt.plot(similarities, marker='o', linestyle='-', label='相邻句子相似度')
    plt.axhline(y=breakpoint_threshold_value, color='r', linestyle='--', label=f'切分阈值 ({splitter.breakpoint_percentile_threshold}百分位)')
    plt.title(title, fontsize=16)
    plt.xlabel("句子连接处索引")
    plt.ylabel("余弦相似度")
    plt.legend()
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("无法设置中文字体 'SimHei'，图表中的中文可能显示为乱码。")
    plt.grid(True)
    plt.show()

    nodes = splitter.get_nodes_from_documents([document])
    print("\n--- 切分结果 ---")
    for i, node in enumerate(nodes):
        print(f"====== 节点 {i+1} (长度: {len(llama_tokenizer(node.get_content()))} tokens) ======")
        print(node.get_content().strip())
        print("-" * 20)

print("="*20 + " 实验一：使用保守阈值 (95) " + "="*20)
conservative_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
    sentence_splitter=chinese_sentence_tokenizer
)
plot_similarity_and_chunks(conservative_splitter, "相似度与切分点 (阈值=95%)")


# 实验二：使用激进阈值 (5)
print("\n" + "="*20 + " 实验二：使用激进阈值 (5) " + "="*20)
aggressive_splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=5,
    embed_model=Settings.embed_model,
    sentence_splitter=chinese_sentence_tokenizer 
)
plot_similarity_and_chunks(aggressive_splitter, "相似度与切分点 (阈值=5%)")