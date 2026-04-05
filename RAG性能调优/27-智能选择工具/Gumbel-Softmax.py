import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

"""
Gumbel-Softmax 工具选择器演示
--------------------------------------
这个示例展示了 Gumbel-Softmax 技术在工具选择中的核心应用。
Gumbel-Softmax 允许在前向传播和反向传播过程中进行可微分的离散选择。
"""

# =======================
# 1. 工具定义 - 三个相似但不同的搜索工具
# =======================
def academic_search(query):
    """学术搜索工具 - 用于搜索学术论文、研究资料"""
    return f"[学术搜索] 找到关于「{query}」的学术论文"

def news_search(query):
    """新闻搜索工具 - 用于搜索最新新闻、时事信息"""
    return f"[新闻搜索] 找到关于「{query}」的最新新闻"

def wiki_search(query):
    """百科搜索工具 - 用于搜索基础知识、定义解释"""
    return f"[百科搜索] 找到关于「{query}」的百科解释"

TOOLS = {
    "academic": academic_search,
    "news": news_search,
    "wiki": wiki_search
}

# =======================
# 2. Gumbel-Softmax 工具选择器 - 核心组件
# =======================
class ToolSelector(nn.Module):
    def __init__(self, num_tools=3, input_dim=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_tools)
        )
        self.tool_names = list(TOOLS.keys())

    def forward(self, x, tau=0.5, hard=False):
        logits = self.model(x)

        # Gumbel-Softmax 采样 - 核心部分
        # 1. 添加 Gumbel 噪声: logits + g, 其中 g ~ Gumbel(0,1)
        # 2. 应用 softmax: softmax((logits + g)/tau)
        # 3. tau 是温度参数: 高温使分布更均匀(探索)，低温使分布更尖锐(利用)
        y_soft = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        return y_soft, logits
    
    def predict(self, x):
        with torch.no_grad():
            _, logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item(), logits[0]
        
# =======================
# 3. 基于FAISS的文本编码器
# =======================
class FAISSTextEncoder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化FAISS文本编码器
        
        Args:
            model_name: 预训练的Sentence Transformer模型名称
        """
        #  1. 加载预训练嵌入模型
        print("加载文本嵌入模型...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384

        # 2. 初始化 FAISS 索引
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # 3. 构建工具类别的代表性查询库
        self._build_tool_knowledge_base()
        print("FAISS文本编码器初始化完成")

    def _build_tool_knowledge_base(self):
        """构建工具类别的代表性查询知识库"""
        # 为每个工具类别准备代表性查询
        tool_queries = {
            "academic": [
                "深度学习研究论文", "机器学习算法分析", "人工智能理论研究",
                "数据挖掘技术综述", "神经网络模型实验", "学术文献调研",
                "计算机视觉最新进展", "自然语言处理研究方法"
            ],
            "news": [
                "最新科技新闻", "今日AI公司动态", "科技企业融资消息",
                "人工智能行业报道", "技术突破新闻", "市场动态更新",
                "本周发布的新产品", "昨天宣布的并购事件"
            ],
            "wiki": [
                "人工智能基本概念", "机器学习定义解释", "深度学习原理介绍",
                "算法基础知识", "技术术语解释", "发展历史概述",
                "什么是神经网络", "强化学习的基本原理"
            ]
        }

        all_queries = []
        self.query_to_tool = {}

        for tool, queries in tool_queries.items():
            for query in queries:
                all_queries.append(query)
                self.query_to_tool[query] = tool
        
        print(f"为{len(all_queries)}个参考查询生成嵌入向量...")
        
        embeddings = self.model.encode(all_queries)
        self.index.add(embeddings.astype('float32'))
        self.reference_queries = all_queries

    def encode_query(self, query):
        """
        将查询文本转换为特征向量
        
        Args:
            query: 查询文本
            
        Returns:
            torch.Tensor: 特征向量
        """
        # 1. 生成查询的嵌入向量
        query_embedding = self.model.encode([query])

        # 2. 在 FAISS 中检索最相似的参考查询
        k = 6  # 检索前6个最相似的
        similarities, indices = self.index.search(
            query_embedding.astype('float32'), k
        )

        # 3. 构建特征向量
        feature_vector = np.zeros(64)

        # 4. 基于相似度分配特征权重
        tool_counts = {"academic": 0, "news": 0, "wiki": 0}

        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= len(self.reference_queries):
                continue

            ref_query = self.reference_queries[idx]
            tool = self.query_to_tool[ref_query]

            tool_idx = ["academic", "news", "wiki"].index(tool)
            base_idx = tool_idx * 20

            # 使用相似度作为权重
            weight = max(0, sim)  # 确保非负
            feature_idx = base_idx + tool_counts[tool]
            if feature_idx < base_idx + 15:  # 确保不超出每个工具的特征空间
                feature_vector[feature_idx] = weight
                tool_counts[tool] += 1

        # 5. 添加特殊特征（保留原有逻辑的一部分）
        if any(w in query for w in ["今日", "最新", "刚刚"]):
            feature_vector[1 * 20 + 15] = 1.0  # news特殊特征
            
        if any(w in query for w in ["研究", "论文"]):
            feature_vector[0 * 20 + 15] = 1.0  # academic特殊特征
            
        if any(w in query for w in ["什么是", "定义", "概念"]):
            feature_vector[2 * 20 + 15] = 1.0  # wiki特殊特征
        
        return torch.FloatTensor(feature_vector).unsqueeze(0)

# 全局编码器实例
text_encoder = None

def encode_query(query):
    """
    将查询文本转换为特征向量的包装函数
    
    Args:
        query: 查询文本
        
    Returns:
        torch.Tensor: 特征向量
    """
    global text_encoder
    
    # 延迟初始化编码器
    if text_encoder is None:
        text_encoder = FAISSTextEncoder()
    
    return text_encoder.encode_query(query)


# =======================
# 4. 训练函数
# =======================
def train_model(train_data, epochs=5):
    # 初始化模型
    selector = ToolSelector(num_tools=3)
    optimizer = optim.Adam(selector.parameters(), lr=0.01)

    # 训练前测试
    print("\n 训练前预测:")
    test_query = "人工智能的基本概念"
    idx, _ = selector.predict(encode_query(test_query))
    print(f"查询: '{test_query}'")
    print(f"预测工具: {selector.tool_names[idx]}")

    # 训练循环
    print("\n 开始训练...")
    for epoch in range(epochs):
        total_loss = 0

        tau = max(0.5, 1.0 - epoch * 0.1)

        for step, (query, target_tool) in enumerate(train_data):
            # 1. 编码查询
            x = encode_query(query)

            # 2. 前向传播 - 使用 Gumbel-Softmax
            y_soft, logits = selector(x, tau=tau)

            # 3. 计算损失
            tool_idx = selector.tool_names.index(target_tool)
            target = torch.zeros_like(logits)
            target[0, tool_idx] = 1.0
            loss = nn.functional.cross_entropy(logits, target)

            # 4. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 打印进度
            if step % 5 == 0 or step == len(train_data) - 1:
                probs = torch.softmax(logits, dim=-1).detach().numpy()[0]
                print(f"[Epoch {epoch}, Step {step}] Query='{query}' | "
                      f"学术:{probs[0]:.3f} 新闻:{probs[1]:.3f} 百科:{probs[2]:.3f} | "
                      f"Loss: {loss.item():.4f} | τ={tau:.2f}")
        
        print(f"Epoch {epoch} 平均损失: {total_loss / len(train_data):.4f}")

    
     # 训练后测试
    print("\n 训练后预测:")
    test_queries = [
        "人工智能的基本概念和发展历史",  # 应该偏向百科
        "最新的人工智能研究进展",       # 应该偏向学术
        "今日人工智能公司股价动态"      # 应该偏向新闻
    ]

    for query in test_queries:
        idx, logits = selector.predict(encode_query(query))
        probs = torch.softmax(logits, dim=-1).detach().numpy()
        print(f"查询: '{query}'")
        print(f"预测工具: {selector.tool_names[idx]}")
        print(f"概率分布: 学术={probs[0]:.3f}, 新闻={probs[1]:.3f}, 百科={probs[2]:.3f}")
        print()
    
    return selector


# =======================
# 5. 主程序
# =======================
if __name__ == "__main__":
    print("Gumbel-Softmax 工具选择器演示 - FAISS版")
    
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 预先初始化文本编码器
    print("初始化FAISS文本编码器...")
    text_encoder = FAISSTextEncoder()
    
    # 训练数据 - 包含明确和模糊边界的查询
    train_data = [
        # 明确的学术查询
        ("深度学习最新研究方法", "academic"),
        ("知识图谱构建技术综述", "academic"),
        ("强化学习算法比较研究", "academic"),
        
        # 明确的新闻查询
        ("今日人工智能公司股价", "news"),
        ("最新科技企业并购消息", "news"),
        ("本周AI创业公司融资动态", "news"),
        
        # 明确的百科查询
        ("什么是知识图谱", "wiki"),
        ("人工智能的基本概念", "wiki"),
        ("机器学习算法分类介绍", "wiki"),
        
        # 模糊边界查询
        ("人工智能研究现状分析", "academic"),
        ("人工智能领域最新突破", "news"),
        ("人工智能的发展历史", "wiki"),
    ]
    
    # 训练模型
    selector = train_model(train_data, epochs=5)
    
    # 实际应用演示
    print("\n 实际应用演示:")
    test_queries = [
        "人工智能在医疗领域的应用研究",
        "今日发布的自动驾驶技术突破",
        "知识图谱的基本概念和构建方法"
    ]
    
    for query in test_queries:
        # 1. 使用 Gumbel-Softmax 选择工具
        idx, logits = selector.predict(encode_query(query))
        selected_tool = selector.tool_names[idx]
        
        # 2. 调用选定的工具
        result = TOOLS[selected_tool](query)
        
        # 3. 显示结果和概率分布
        probs = torch.softmax(logits, dim=-1).detach().numpy()
        print(f"查询: '{query}'")
        print(f"工具选择: {selected_tool}")
        print(f"概率分布: 学术={probs[0]:.3f}, 新闻={probs[1]:.3f}, 百科={probs[2]:.3f}")
        print(f"执行结果: {result}")
        print("-" * 50)
