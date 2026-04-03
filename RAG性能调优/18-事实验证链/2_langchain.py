import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

@dataclass
class SourceInfo:
    """数据源信息"""
    url: str
    title: str
    content: str
    timestamp: datetime

class SimpleRAGSystem:
    """基于 LangChain 的简单 RAG 系统"""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        self.embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
        self.llm = OpenAI(
            model="gpt-4o",
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vector_store = None
        self.qa_chain = None

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是一个事实验证专家。请根据提供的上下文信息回答问题。

上下文信息：
{context}

问题：{input}

请按照以下格式回答：
验证结果：[真实/虚假/不确定]
置信度：[0-100%]
推理过程：[详细说明推理过程]
证据来源：[引用具体的证据片段]
"""
        )

    def get_knowledge_sources(self) -> List[SourceInfo]:
        """获取知识源（使用预定义内容，避免网络爬取的复杂性）"""
        sources = [
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/人工智能",
                title="人工智能 - 维基百科",
                content="""
                人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
                致力于创造能够执行通常需要人类智能的任务的机器和软件。
                
                人工智能的主要领域包括：
                1. 机器学习：让计算机从数据中学习模式和规律
                2. 深度学习：使用多层神经网络进行复杂模式识别
                3. 自然语言处理：让计算机理解和生成人类语言
                4. 计算机视觉：让计算机理解和分析图像和视频
                5. 专家系统：基于规则和知识的决策系统
                
                关于人工智能是否会在未来十年内超越人类智能，学术界存在很大争议。
                一些专家认为通用人工智能（AGI）的实现还需要数十年时间，
                而另一些专家则认为可能在更短时间内实现突破。
                目前的AI系统在特定任务上表现优异，但在通用智能方面仍有很大差距。
                """,
                timestamp=datetime.now()
            ),
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/机器学习",
                title="机器学习 - 维基百科", 
                content="""
                机器学习是人工智能的一个重要分支，专注于开发能够从数据中学习和改进的算法。
                
                机器学习与人工智能的关系：
                - 人工智能是更广泛的概念，包括所有模拟人类智能的技术
                - 机器学习是实现人工智能的主要方法之一
                - 深度学习是机器学习的一个子领域
                
                层次关系：人工智能 > 机器学习 > 深度学习
                
                机器学习的主要类型：
                1. 监督学习：使用标记数据进行训练
                2. 无监督学习：从未标记数据中发现模式
                3. 强化学习：通过试错学习最优策略
                
                常用算法包括线性回归、决策树、随机森林、支持向量机、神经网络等。
                """,
                timestamp=datetime.now()
            ),
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/深度学习",
                title="深度学习 - 维基百科",
                content="""
                深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示。
                
                深度学习确实是机器学习的一个子领域。这种层次关系是明确的：
                - 人工智能是最广泛的概念
                - 机器学习是人工智能的一个分支
                - 深度学习是机器学习的一个专门领域
                
                深度学习的核心概念：
                1. 神经网络：模拟人脑神经元的计算模型
                2. 反向传播：训练神经网络的核心算法
                3. 卷积神经网络（CNN）：主要用于图像处理
                4. 循环神经网络（RNN）：用于处理序列数据
                5. 变换器（Transformer）：用于自然语言处理
                
                深度学习在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。
                """,
                timestamp=datetime.now()
            ),
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/Python",
                title="Python - 维基百科",
                content="""
                Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。
                
                Python在数据科学领域的地位：
                Python确实是数据科学中最流行的编程语言之一。根据多项调查和统计：
                
                1. Stack Overflow 开发者调查显示，Python在数据科学家中使用率最高
                2. Kaggle 数据科学平台的调查显示，超过80%的数据科学家使用Python
                3. GitHub上数据科学相关项目中，Python项目数量最多
                
                Python在数据科学中的优势：
                - 语法简洁易读，学习曲线平缓
                - 丰富的数据科学库生态系统（pandas, numpy, scikit-learn等）
                - 强大的机器学习框架（TensorFlow, PyTorch等）
                - 优秀的数据可视化工具（matplotlib, seaborn等）
                - 活跃的社区支持
                
                主要竞争对手包括R语言、SQL、Java等，但Python在综合能力上领先。
                """,
                timestamp=datetime.now()
            )
        ]
        return sources
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("🔧 正在构建知识库...")

        sources = self.get_knowledge_sources()

        documents = []
        for source in sources:
            chunks = self.text_splitter.split_text(source.content)


            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source.url,
                        "title": source.title,
                        "timestamp": source.timestamp.isoformat()
                    }
                )
                documents.append(doc)

        self.vector_store = FAISS.from_documents(documents, self.embeddings)

        combine_docs_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt_template)
        self.qa_chain = create_retrieval_chain(self.vector_store.as_retriever(), combine_docs_chain)

        print(f"✅ 知识库构建完成，包含 {len(documents)} 个文档片段")

    def ask_question(self, question: str):
        """询问问题并获取答案"""
        if not self.qa_chain:
            raise ValueError("知识库未初始化，请先调用 build_knowledge_base()")

        print(f"❓ 问题：{question}")
        print("🤔 正在思考...")

        result = self.qa_chain.invoke({"input": question})
        answer = result["answer"]
        source_docs = result["context"]

        print(f"🤖 AI回答：")
        print(answer)
        
        # 显示证据来源
        if source_docs:
            print("\n📚 证据来源：")
            for i, doc in enumerate(source_docs, 1):
                print(f"   {i}. {doc.metadata.get('title', '未知来源')}")
                print(f"      🔗 {doc.metadata.get('source', '无链接')}")
                print(f"      📄 内容片段：{doc.page_content[:200]}...")
                print()
        
        return answer, source_docs
    
def main():
    """主函数：演示基于LangChain的RAG系统"""
    print("=== 基于 LangChain 的简单 RAG 系统演示 ===\n")

    print("🚀 正在初始化RAG系统...")
    rag_system = SimpleRAGSystem()

    rag_system.build_knowledge_base()

    questions = [
        "人工智能将在未来十年内超越人类智能吗？",
        "深度学习是机器学习的一个子领域吗？", 
        "Python是数据科学中最流行的语言吗？"
    ]

    print("\n=== 开始问答演示 ===\n")

    for i, question in enumerate(questions, 1):
        print(f"{'='*60}")
        print(f"第 {i} 个问题")
        print(f"{'='*60}")
        
        try:
            answer, sources = rag_system.ask_question(question)
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ 处理问题时出错: {e}")
            print(f"{'='*60}\n")

    print("✨ RAG系统演示完成！")

if __name__ == "__main__":
    # 运行演示程序
    main()