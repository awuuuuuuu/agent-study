import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

@dataclass
class SourceInfo:
    """数据源信息"""
    title: str
    content: str
    url: str

class SimpleRAGSystem:
    """基于 LlamaIndex 的极简 RAG 系统"""
    def __init__(self):
        """初始化系统"""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        self.llm = OpenAI(model="gpt-4o", api_key=api_key, api_base=base_url)
        self.embed_model = OpenAIEmbedding(api_key=api_key, api_base=base_url)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def get_knowledge_sources(self) -> List[SourceInfo]:
        """获取预定义知识源"""
        return [
            SourceInfo(
                title="人工智能基础知识",
                content="""
                人工智能（AI）是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。
                
                主要领域包括：
                1. 机器学习：让计算机从数据中学习
                2. 深度学习：使用多层神经网络
                3. 自然语言处理：理解和生成人类语言
                4. 计算机视觉：分析图像和视频
                
                关于AI是否会在未来十年内超越人类智能，专家意见存在分歧。
                一些专家认为通用人工智能（AGI）还需要数十年时间，
                另一些专家则认为可能在更短时间内实现突破。
                目前AI在特定任务上表现优异，但在通用智能方面仍有差距。
                """,
                url="https://zh.wikipedia.org/wiki/人工智能"
            ),
            SourceInfo(
                title="机器学习与深度学习",
                content="""
                机器学习是人工智能的一个重要分支，专注于开发能够从数据中学习的算法。
                
                层次关系：
                - 人工智能是最广泛的概念
                - 机器学习是人工智能的一个分支
                - 深度学习是机器学习的一个子领域
                
                深度学习确实是机器学习的一个子领域，使用多层神经网络来学习复杂模式。
                
                主要类型：
                1. 监督学习：使用标记数据训练
                2. 无监督学习：从未标记数据发现模式
                3. 强化学习：通过试错学习策略
                
                深度学习的核心技术包括神经网络、反向传播、CNN、RNN、Transformer等。
                """,
                url="https://zh.wikipedia.org/wiki/机器学习"
            ),
            SourceInfo(
                title="Python在数据科学中的地位",
                content="""
                Python是一种高级编程语言，在数据科学领域极其流行。
                
                Python确实是数据科学中最流行的语言之一，证据包括：
                
                1. 调查数据：
                   - Stack Overflow调查显示Python在数据科学家中使用率最高
                   - Kaggle平台调查显示超过80%的数据科学家使用Python
                   - GitHub上数据科学项目中Python项目数量最多
                
                2. 技术优势：
                   - 语法简洁易读，学习曲线平缓
                   - 丰富的数据科学库（pandas, numpy, scikit-learn）
                   - 强大的机器学习框架（TensorFlow, PyTorch）
                   - 优秀的可视化工具（matplotlib, seaborn）
                   - 活跃的社区支持
                
                主要竞争对手包括R语言、SQL、Java等，但Python在综合能力上领先。
                """,
                url="https://zh.wikipedia.org/wiki/Python"
            )
        ]
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("🔧 正在构建知识库...")
        
        sources = self.get_knowledge_sources()
        documents = []

        for source in sources:
            doc = Document(
                text=source.content,
                metadata={
                    "title": source.title,
                    "url": source.url
                }
            )
            documents.append(doc)

        self.index = VectorStoreIndex.from_documents(documents)

        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
        print(f"✅ 知识库构建完成，包含 {len(documents)} 个文档")

    def ask_question(self, question: str):
        """询问问题并获取答案"""
        print(f"❓ 问题：{question}")
        print("🤔 正在思考...")

        response = self.query_engine.query(question)

        print(f"🤖 AI回答：")
        print(response.response)
        print("\n📚 证据来源：")
        for i, node in enumerate(response.source_nodes, 1):
            title = node.node.metadata.get('title', '未知来源')
            url = node.node.metadata.get('url', '无链接')
            # 相似度分数是基于余弦相似度（cosine similarity）计算，计算范围是 -1 到 1，但在实际应用中转换为 0 到 1
            score = getattr(node, 'score', 0.0)
            
            print(f"   {i}. {title}")
            print(f"      🔗 {url}")
            print(f"      📊 相关性: {score:.3f}")
            print(f"      📄 内容片段：{node.node.text[:200]}...")
            print()

        return response
    
def main():
    """主函数：演示基于LlamaIndex的极简RAG系统"""
    print("=== 基于 LlamaIndex 的极简 RAG 系统演示 ===\n")

    print("🚀 正在初始化RAG系统...")
    rag_system = SimpleRAGSystem()

    rag_system.build_knowledge_base()

    questions = [
        "人工智能将在未来十年内超越人类智能吗？",
        "深度学习是机器学习的一个子领域吗？", 
        "Python是数据科学中最流行的语言吗？"
    ]

    print("\n=== 开始问答演示 ===\n")
    
    # 逐一提问
    for i, question in enumerate(questions, 1):
        print(f"{'='*60}")
        print(f"第 {i} 个问题")
        print(f"{'='*60}")
        
        rag_system.ask_question(question)
        print(f"{'='*60}\n")

        print("✨ RAG系统演示完成！")

if __name__ == "__main__":
    main()