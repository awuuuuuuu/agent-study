from swift import load_dataset

dataset = load_dataset(['swift/self-cognition'], model_name=['阿无', 'awu'], model_author=['魔搭', 'ModelScope'])[0]
print(dataset)
print(dataset[0])
"""
Dataset({
    features: ['messages'],
    num_rows: 108
})
{'messages': [{'role': 'user', 'content': '你是？'}, {'role': 'assistant', 'content': '我是小黄，由魔搭训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？'}]}
"""

# 支持重采样：（超过108后进行重采样）
dataset = load_dataset(['swift/self-cognition#500'], model_name=['阿无', 'awu'], model_author=['魔搭', 'ModelScope'])[0]
print(dataset)
"""
Dataset({
    features: ['messages'],
    num_rows: 500
})
"""