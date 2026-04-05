"""
记忆向量核心模块
"""
import numpy as np
from datetime import datetime

class MemoryVector:
    """记忆向量封装类"""
    def __init__(self, content: str, importance: float = None):
        self.content = content
        self.importance = importance if importance is not None else self._calculate_importance(content)
        self.timestamp = datetime.now()
        self.access_count = 0
        self.embedding = self._generate_embedding()
    
    def _generate_embedding(self) -> np.array:
        """生成嵌入向量"""
        np.random.seed(hash(self.content) % 2 **32)
        vector = np.random.randn(128).astype('float32')

        # 关键词增强
        keywords = ['排他性', '条款', '违约', '重要', '合同']
        for i, keyword in enumerate(keywords):
            if keyword in self.content:
                vector[i * 20 : (i + 1) * 20] += 0.5
        
        return vector / np.linalg.norm(vector)
    
    def update_access(self):
        """更新访问统计"""
        self.access_count += 1
    
    def _calculate_importance(self, content: str) -> float:
        """计算内容重要性"""
        score = 0.3
        
        # 关键词检测
        keywords = ['排他性', '违约金', '重要', '终止', '义务']
        for keyword in keywords:
            if keyword in content:
                score += 0.15
        
        # 长度加分
        if len(content) > 30:
            score += 0.1
        
        return min(1.0, score)