"""
Memory Transformer 主模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List
from core.memory_vector import MemoryVector
from core.memory_storage import MemoryStorage
from core.memory_retriever import MemoryRetriever

class MemoryTransformer:
    """FAISS优化的Memory Transformer主模型"""

    def __init__(self, short_limit: int = 15, long_limit: int = 100):
        self.storage = MemoryStorage(short_limit, long_limit)
        self.retriever = MemoryRetriever()

    def process(self, content: str) -> Dict:
        """处理输入内容"""
        
        memory = MemoryVector(content)
        self.storage.add_memory(memory)

        return {
            'memory_status': self.storage.get_status(),
            'content': content
        }
    
    def query(self, query: str, top_k: int = 5) -> List[MemoryVector]:
        """查询相关记忆"""
        return self.retriever.retrieve(query, self.storage, top_k)
        


    