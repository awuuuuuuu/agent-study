import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

######
# 核心优化策略：
# 1. 干预时机不要过早：修改 Attention 层属于模型内部干预，Attention 的主要作用是让模型关注输入序列的不同部分，
#   直接增强特定 token 的 attention score 并不直接等同于“让模型多生成这个词”，反而可能破坏模型对上下文的理解，导致生成内容不连贯。
# 2. 干预目标确保精确：更直接、有效的方法是直接干预最终生成词汇的概率分布（logits），
#   在它进入 softmax 之前修改，可以精确地提升或降低特定词汇的生成概率。
# 3. 关键词 Tokenization 问题：代码中使用 tokenizer.convert_tokens_to_ids 获取词汇 ID，
#   但像 “大模型”、“自动驾驶” 等词通常会被 tokenizer 拆分为多个 token（如 "大"、"模型"），只会关注第一个 token，效果有限。
# 4. 状态管理复杂：通过 model.current_input_ids 传递输入信息给 hook 函数的方式不够健壮，
#   且在生成多个 token 的过程中，input_ids 会变化，逻辑会变得很复杂。

# ================== 1. 加载模型与 Tokenizer ==================
model_name = "Qwen/Qwen3-0.6B"

print("正在加载模型和Tokenizer...")
# 使用 pipeline 加载模型
chat_pipeline = pipeline(
    task=Tasks.chat,
    model=model_name,
    device_map="auto"
)

# 获取模型和 tokenizer
model = chat_pipeline.model
tokenizer = chat_pipeline.tokenizer
print("模型加载完成！")


# ================== 2. 定义关键词和获取 Token IDs ==================
# 定义关键词列表
keywords = ["大模型", "人工智能", "医疗", "自动驾驶", "智能客服"]
focus_token_ids = []

for keyword in keywords:
    token_ids = tokenizer.encode(keyword, add_special_tokens=False)
    if token_ids:
        focus_token_ids.extend(token_ids)

# 去重，并转换为 tensor
focus_token_ids = torch.tensor(list(set(focus_token_ids)), device=model.device)
print(f"需要关注的关键词 Token ID: {focus_token_ids}")

# ================== 3. 定义 Hook 函数（作用于 Logits） ==================
def modify_logits_hook(module, input, output):
    """
    这个 hook 函数在 lm_head 计算出 logits 后被调用。
    它会直接提升我们关注的关键词的 logits 值。
    """
    logits = output[0] if isinstance(output, tuple) else output
    last_token_logits = logits[:, -1, :]

    #在使用 focus_token_ids 之前，确保它和 logits 在同一个设备上。避免 RuntimeError 的关键。
    device_aware_focus_ids = focus_token_ids.to(last_token_logits.device)

    bias = 3.0
    # 使用aware_focus_ids 进行索引
    last_token_logits[:, device_aware_focus_ids] += bias

    return output

# ================== 4. 注册与移除 Hook ==================
hook_handle = None

def add_hook():
    global hook_handle
    # 将 hook 注册到模型的 lm_head (输出层)，这是最直接影响生成结果的地方
    if hasattr(model, "lm_head") and hook_handle is None:
        print("Hook 已注册到 lm_head")
        hook_handle = model.lm_head.register_forward_hook(modify_logits_hook)
    else:
        print("未能找到 lm_head 或 Hook 已存在。")

def remove_hook():
    global hook_handle
    if hook_handle:
        hook_handle.remove()
        hook_handle = None
        print("Hook 已移除")

## ================== 5. 定义生成函数 ==================
def generate_text(prompt):
    """
    使用 pipeline 生成文本。
    """
    messages = [{"role":"user", "content": prompt}]

    response = chat_pipeline(
        messages,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        no_repeat_ngram_size=2
    )

    print(response)

    content = response['message']['content']

    if '</think>' in content:
        content = content.split('</think>', 1)[-1].strip()
    return content

# ================== 6. 定义关键词统计函数 ==================
def count_keywords(text, keywords):
    count = 0
    present_keywords = []
    for word in keywords:
        if word in text:
            count += text.count(word)
            present_keywords.append(word)
    return count, present_keywords


# ================== 7. 测试对比 ==================
test_prompt = """
请用自然流畅的语言，深入探讨一下人工智能和大模型的未来发展趋势，并结合医疗、自动驾驶、智能客服等具体行业，分析它们的潜在应用和挑战。

"""

print("\n" + "="*20 + " 1. 不使用 Hook 生成 " + "="*20)

remove_hook()
output_no_hook = generate_text(test_prompt)
print(f"\n【生成结果】:\n{output_no_hook}")
count1, present1 = count_keywords(output_no_hook, keywords)
print(f"\n【关键词统计】: 数量: {count1}, 出现的词: {present1}")

print("\n" + "="*20 + " 2. 使用 Hook 生成（强调关键词） " + "="*20)
add_hook()
output_with_hook = generate_text(test_prompt)
print(f"\n【生成结果】:\n{output_with_hook}")
count2, present2 = count_keywords(output_with_hook, keywords)
print(f"\n【关键词统计】: 数量: {count2}, 出现的词: {present2}")


remove_hook()