#!/usr/bin/env python3
"""智能网关测试"""
import requests

def test_gateway():
    """测试智能模型选择"""
    base_url = "http://localhost:8000"
    
    # 健康检查
    try:
        if requests.get(f"{base_url}/health", timeout=5).status_code != 200:
            print("❌ 网关服务异常")
            return
        print("✅ 网关服务正常")
    except:
        print("❌ 无法连接到网关服务")
        return
    
    # 测试用例 - 修复问题：添加预期结果说明
    test_cases = [
        {"question": "什么是AI？", "expected": "快速模型"},
        {"question": "请解释机器学习的基本原理", "expected": "平衡模型"}, 
        {"question": "设计一个完整的分布式推荐系统架构", "expected": "高级模型"}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['question']}")
        print(f"预期: {case['expected']}")
        
        try:
            print("正在处理请求...")
            response = requests.post(f"{base_url}/v1/chat/completions", 
                json={"model": "auto", "messages": [{"role": "user", "content": case["question"]}]}, 
                timeout=30)  # 增加超时时间到30秒
            
            if response.status_code == 200:
                result = response.json()
                print(f"实际选择: {result['model']}")
                # 显示响应内容长度
                if 'choices' in result and result['choices']:
                    content_len = len(result['choices'][0]['message']['content'])
                    print(f"响应长度: {content_len} 字符")
            else:
                print(f"请求失败: {response.status_code}")
                print(f"响应内容: {response.text}")
        except requests.exceptions.Timeout:
            print("请求超时 - 服务器响应时间过长")
        except requests.exceptions.ConnectionError:
            print("连接错误 - 无法连接到服务器")
        except Exception as e:
            print(f"其他异常: {e}")
    
    # 显示统计信息
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"\n系统统计:")
            print(f"可用引擎: {stats['engine_types']}")
            print(f"API状态: {'可用' if stats['api_available'] else '模拟'}")
    except requests.exceptions.Timeout:
        print("统计信息获取超时")
    except Exception as e:
        print(f"无法获取统计信息: {e}")
    
    print("\n测试完成")

if __name__ == "__main__":
    test_gateway()
