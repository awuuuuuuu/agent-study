"""
FAISS记忆检索器 - 简化版本
"""
import numpy as np
from typing import List
from .memory_vector import MemoryVector

class MemoryRetriever:
    """FAISS记忆检索器"""
    
    def retrieve(self, query: str, storage, top_k: int = 5) -> List[MemoryVector]:
        """检索相关记忆"""
        query_vector = self._encode_query(query)
        return storage.search(query_vector, top_k)
    
    def _encode_query(self, query: str) -> np.ndarray:
        """编码查询"""
        np.random.seed(hash(query) % 2**32)
        vector = np.random.randn(128).astype('float32')
        
        keywords = ['排他性', '条款', '违约', '重要', '合同']
        for i, keyword in enumerate(keywords):
            if keyword in query:
                vector[i*20:(i+1)*20] += 0.5
        
        return vector / np.linalg.norm(vector)