"""
Memory Transformer 综合演示
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import MemoryTransformer

class TraditionalContext:
    """传统上下文窗口模拟"""
    def __init__(self, max_size=8):
        self.context = []
        self.max_size = max_size
    
    def add(self, content):
        self.context.append(content)
        if len(self.context) > self.max_size:
            self.context.pop(0)
    
    def search(self, query):
        results = []
        for item in self.context:
            if any(word in item for word in query.split()):
                results.append(item)
        return results
    
def run_multi_turn_demo():
    """多轮对话演示"""
    print("多轮对话记忆保持演示")
    print("=" * 50)
    
    model = MemoryTransformer()
    
    conversations = [
        ("用户", "我需要审核代理合同，特别关注排他性条款"),
        ("助手", "好的，我会重点关注排他性条款。请提供合同内容"),
        ("用户", "合同提到违反排他性条款要支付50%违约金"),
        ("助手", "50%违约金确实较高，需要评估风险"),
        ("用户", "还有终止后2年内仍需履行排他性义务"),
        ("助手", "2年义务期较长，建议协商缩短"),
        ("用户", "如果违反排他性条款会面临什么后果？"),
        ("助手", "主要是50%违约金和可能的法律诉讼"),
        ("用户", "有办法降低违约金比例吗？"),
        ("助手", "可以协商或增加免责条款"),
        ("用户", "总结一下排他性条款的主要风险点"),
    ]
    
    print("模拟10轮对话，观察记忆系统变化...")

     # 记录每轮对话后的实际状态
    for i, (role, content) in enumerate(conversations[:-1]):
        result = model.process(f"{role}: {content}")
        
        # 获取真实的记忆状态
        actual_status = model.storage.get_status()
        total_memories = actual_status['short_term'] + actual_status['long_term']
        
        # 每3轮显示一次状态，确保数据准确
        if (i + 1) % 3 == 0 or i == 0:
            print(f"   第{i+1}轮对话后 → 总记忆: {total_memories}条 (短期: {actual_status['short_term']}, 长期: {actual_status['long_term']})")
    
    # 显示最终状态
    final_status = model.storage.get_status()
    final_total = final_status['short_term'] + final_status['long_term']
    print(f"   最终状态 → 总记忆: {final_total}条 (短期: {final_status['short_term']}, 长期: {final_status['long_term']})")
    
    final_query = conversations[-1][1]
    print(f"\n智能检索测试: {final_query}")
    
    retrieved = model.query(final_query, top_k=3)
    print(f"从 {final_total} 条记忆中检索到 {len(retrieved)} 条最相关内容:")
    
    for i, memory in enumerate(retrieved):
        content_preview = memory.content.replace("用户: ", "").replace("助手: ", "")[:40]
        print(f"   {i+1}. {content_preview}...")
        print(f"      [重要性: {memory.importance:.2f}, 访问次数: {memory.access_count}]")
    
    return model

def run_comparison_demo():
    """性能对比演示"""
    print("\n传统方案 vs Memory Transformer 对比")
    print("=" * 50)
    
    traditional = TraditionalContext(max_size=6)
    memory_transformer = MemoryTransformer()
    
    test_sequence = [
        "重要：排他性条款违约金为合同总额50%",
        "一般对话：今天天气很好",
        "重要：合同终止后排他性义务仍需履行2年",
        "一般对话：你好，有什么可以帮助的？",
        "一般对话：谢谢你的帮助",
        "重要：代理区域享有独家销售权",
        "一般对话：再见，明天见",
        "一般对话：祝你工作顺利",
        "一般对话：今天会议很成功",
        "一般对话：下周继续讨论",
    ]
    
    print("输入10条信息（3条重要，7条普通）...")
    important_items = []
    for i, item in enumerate(test_sequence):
        traditional.add(item)
        memory_transformer.process(item)
        if "重要" in item:
            important_items.append(item)
            print(f"   第{i+1}项: [关键信息] {item[3:50]}...")
        else:
            print(f"   第{i+1}项: [普通对话] {item[5:35]}...")
    
    # 显示存储状态
    print(f"\n存储状态:")
    print(f"   传统方案: 保留最近 {len(traditional.context)} 条记录")
    
    mt_status = memory_transformer.storage.get_status()
    print(f"   Memory Transformer: 总计 {mt_status['short_term'] + mt_status['long_term']} 条记录")
    print(f"     - 短期记忆: {mt_status['short_term']} 条")
    print(f"     - 长期记忆: {mt_status['long_term']} 条")
    
    query = "排他性条款违约金"
    print(f"\n测试查询: '{query}'")
    
    # 传统方案检索
    traditional_results = traditional.search("排他性 违约金")
    traditional_important = [r for r in traditional_results if "重要" in r]
    
    # Memory Transformer检索
    mt_results = memory_transformer.query(query, top_k=5)
    mt_important = [r for r in mt_results if "重要" in r.content]
    
    total_important = len(important_items)
    
    print(f"\n检索结果对比:")
    print(f"   重要信息总数: {total_important} 条")
    print(f"   传统方案:")
    print(f"     - 检索到: {len(traditional_results)} 条相关记录")
    print(f"     - 其中重要信息: {len(traditional_important)} 条 ({len(traditional_important)/total_important*100:.0f}%)")
    
    print(f"   Memory Transformer:")
    print(f"     - 检索到: {len(mt_results)} 条相关记录")
    print(f"     - 其中重要信息: {len(mt_important)} 条 ({len(mt_important)/total_important*100:.0f}%)")
    
    # 计算性能差异
    traditional_recall = len(traditional_important) / total_important
    mt_recall = len(mt_important) / total_important
    
    if mt_recall > traditional_recall:
        improvement = (mt_recall - traditional_recall) * 100
        print(f"\n性能优势: Memory Transformer 重要信息召回率高出 {improvement:.0f}%")
    elif mt_recall == traditional_recall:
        print(f"\n两种方案在此测试中表现相当")
    else:
        decline = (traditional_recall - mt_recall) * 100
        print(f"\n注意: Memory Transformer 在此测试中表现略低 {decline:.0f}%")
    
    # 显示具体检索到的重要信息
    if mt_important:
        print(f"\nMemory Transformer 检索到的重要信息:")
        for i, memory in enumerate(mt_important):
            content = memory.content.replace("重要：", "")[:40]
            print(f"   {i+1}. {content}... (重要性: {memory.importance:.2f})")
    
    return {
        'traditional_recall': traditional_recall,
        'mt_recall': mt_recall,
        'improvement': mt_recall - traditional_recall
    }

def main():
    """主演示程序"""
    print("Memory Transformer 综合演示")
    print("=" * 60)
    
    # 运行多轮对话演示
    model = run_multi_turn_demo()
    
    # 运行性能对比演示
    results = run_comparison_demo()

    print("\n" + "=" * 50)
    print("演示总结:")
    print(f"   传统方案重要信息召回率: {results['traditional_recall']*100:.0f}%")
    print(f"   Memory Transformer召回率: {results['mt_recall']*100:.0f}%")
    
    if results['improvement'] > 0:
        print(f"   Memory Transformer 在此测试中表现更优")
    elif results['improvement'] == 0:
        print(f"   两种方案在此测试中表现相当")
    else:
        print(f"   传统方案在此测试中表现更优")
    
    print("\n注意: 实际性能取决于具体使用场景和数据特征")


if __name__ == "__main__":
    main()