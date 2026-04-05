"""
FAISS分层记忆存储系统
"""
import faiss
import numpy as np
from typing import List
from .memory_vector import MemoryVector

class MemoryStorage:
    """FAISS分层记忆存储"""
    def __init__(self, short_limit: int = 15, long_limit: int = 100):
        self.short_memories: List[MemoryVector] = []
        self.long_memories: List[MemoryVector] = []
        self.short_limit = short_limit
        self.long_limit = long_limit

        self.short_index = faiss.IndexFlatIP(128)
        self.long_index = faiss.IndexFlatIP(128)

    def add_memory(self, memory: MemoryVector):
        """添加记忆"""
        if memory.importance >= 0.6:
            self._add_long_term(memory)
        else:
            self._add_short_term(memory)
    
    def _add_short_term(self, memory: MemoryVector):
        """添加短期记忆"""
        self.short_memories.append(memory)
        self.short_index.add(memory.embedding.reshape(1, -1))
        
        if len(self.short_memories) > self.short_limit:
            # 转移重要记忆到长期
            important = [m for m in self.short_memories if m.importance > 0.5]
            for mem in important[:len(important)//2]:
                self._add_long_term(mem)
            
            # 重建短期索引
            self.short_memories = self.short_memories[-self.short_limit//2:]
            self._rebuild_short_index()

    def _add_long_term(self, memory: MemoryVector):
        """添加长期记忆"""
        if memory not in self.long_memories:
            self.long_memories.append(memory)
            self.long_index.add(memory.embedding.reshape(1, -1))

    def _rebuild_short_index(self):
        """重建短期索引"""
        self.short_index = faiss.IndexFlatIP(128)
        if self.short_memories:
            embeddings = np.vstack([m.embedding for m in self.short_memories])
            self.short_index.add(embeddings)
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[MemoryVector]:
        """FAISS搜索"""
        results = []
        
        # 搜索短期记忆
        if len(self.short_memories) > 0:
            scores, indices = self.short_index.search(query_vector.reshape(1, -1), min(top_k, len(self.short_memories)))
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self.short_memories[idx], score))
        
        # 搜索长期记忆
        if len(self.long_memories) > 0:
            scores, indices = self.long_index.search(query_vector.reshape(1, -1), min(top_k, len(self.long_memories)))
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self.long_memories[idx], score))

        # 排序返回
        results.sort(key=lambda x: x[1], reverse=True)
        memories = [mem for mem, _ in results[:top_k]]
        
        for mem in memories:
            mem.update_access()
        
        return memories
    
    def get_all_memories(self) -> List[MemoryVector]:
        """获取所有记忆"""
        return self.short_memories + self.long_memories

    def get_status(self) -> dict:
        """获取状态"""
        return {
            'short_term': len(self.short_memories),
            'long_term': len(self.long_memories),
            'total': len(self.short_memories) + len(self.long_memories)
        }