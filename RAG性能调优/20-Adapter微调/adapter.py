import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# --- 配置：使用超小模型适配M1 CPU ---
# 选择bert-tiny的原因：
# 1. 参数量仅4.4M，适合CPU训练
# 2. 保持BERT架构完整性，便于理解Adapter插入位置
# 3. 在M1芯片上训练速度快，适合演示和学习
MODEL_NAME = "prajjwal1/bert-tiny"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
NUM_LABELS = 2  # 二分类：正面/负面，降低复杂度专注于Adapter机制

print("=== Adapter微调 vs LoRA微调的核心区别 ===")
print("1. Adapter：在模型层之间插入小型神经网络模块")
print("2. LoRA：通过低秩矩阵分解修改原有权重矩阵")
print("3. Adapter添加新的网络层，LoRA修改现有层的权重")

# --- 手动实现Adapter层（突出Adapter特点）---
class AdapterLayer(nn.Module):
    """
    Adapter层的核心实现 - 这是Adapter微调的精髓所在：
    
    设计理念：通过"瓶颈"结构实现参数高效微调
    - 下投影层：将输入维度(如128)压缩到更小维度(如16)，形成信息瓶颈
      目的：强制模型学习最重要的特征表示，减少参数量
    - 激活函数：ReLU增加非线性变换能力，让Adapter能学习复杂模式
    - 上投影层：将压缩后的特征恢复到原始维度，保持与原模型的兼容性
    - 残差连接：保持原始信息流不被破坏，关键！
    
    为什么使用残差连接？
    1. 保证即使Adapter输出为0，原始信息仍能完整传递
    2. 让Adapter学习"增量"信息，而非替代原有表示
    3. 训练稳定性更好，避免梯度消失问题
    """
    def __init__(self, input_dim, adapter_dim=16):
        super().__init__()
        # 下投影：大维度->小维度，创建信息瓶颈，强制学习重要特征
        self.down_project = nn.Linear(input_dim, adapter_dim)
        # 激活函数：增加非线性，让简单的线性变换具备复杂的表达能力
        self.activation = nn.ReLU()
        # 上投影：小维度->大维度，恢复到原始空间，保持模型兼容性
        self.up_project = nn.Linear(adapter_dim, input_dim)
        # Dropout：防止过拟合，特别重要因为Adapter参数相对较少
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Adapter的前向传播：下投影 -> 激活 -> 上投影 + 残差连接
        # 这个过程可以理解为：压缩信息 -> 非线性变换 -> 恢复维度 -> 与原信息融合
        adapter_output = self.up_project(self.activation(self.down_project(x)))
        # 残差连接是核心：x + adapter_output 确保原始信息不丢失
        # 这样Adapter只需要学习"需要调整的部分"，而不是重新学习所有内容
        return x + self.dropout(adapter_output)

# 展示Adapter微调的完整架构
class ModelWithAdapter(nn.Module):
    """
    展示Adapter微调的完整架构
    
    核心设计思想：
    1. 保持预训练模型完全不变（冻结参数）
    2. 在关键位置插入可训练的Adapter模块
    3. 让Adapter学习任务特定的知识，而基础模型提供通用语言理解能力
    
    这种设计的优势：
    - 参数效率：只训练<1%的参数就能获得良好效果
    - 模块化：不同任务可以使用不同的Adapter，基础模型共享
    - 稳定性：不会破坏预训练模型已学到的知识
    """
    def __init__(self, base_model, adapter_dim=16):
        super().__init__()
        self.base_model = base_model
        
        # 冻结基础模型的所有参数（Adapter微调的关键特点）
        # 为什么要冻结？
        # 1. 保护预训练知识：避免在小数据集上过拟合而破坏通用能力
        # 2. 参数效率：只需训练少量Adapter参数，大幅降低计算成本
        # 3. 避免灾难性遗忘：确保模型不会忘记预训练时学到的语言知识
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 在BERT的每个Transformer层后添加Adapter
        # 为什么每层都加？因为不同层学习不同抽象级别的特征：
        # - 底层：词汇、语法特征
        # - 中层：句法、语义关系  
        # - 高层：任务相关的抽象概念
        # 每层都加Adapter可以让模型在各个抽象级别都进行任务适配
        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_size, adapter_dim) 
            for _ in range(base_model.config.num_hidden_layers)
        ])
        
        print(f"✓ 基础模型参数已冻结")
        print(f"✓ 添加了 {len(self.adapters)} 个Adapter层")
        print(f"✓ 每个Adapter层参数量: {adapter_dim * hidden_size * 2 + adapter_dim + hidden_size}")
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # 获取基础模型的隐藏状态
        # 为什么要output_hidden_states=True？
        # 因为我们需要获取每一层的输出，然后在每层后面插入对应的Adapter
        # 这样可以让Adapter在不同的抽象层次上都发挥作用
        bert_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True  # 关键：获取所有层的隐藏状态
        }
        if token_type_ids is not None:
            bert_inputs['token_type_ids'] = token_type_ids
            
        outputs = self.base_model.bert(**bert_inputs)
        hidden_states = outputs.hidden_states  # 包含embedding层+所有transformer层的输出
        
        # 在每一层的输出上应用对应的Adapter
        # 为什么从hidden_states[1:]开始？
        # 因为hidden_states[0]是embedding层输出，我们从第一个transformer层开始加Adapter
        adapted_hidden_states = []
        for i, (hidden_state, adapter) in enumerate(zip(hidden_states[1:], self.adapters)):
            # 每个Adapter处理对应层的输出，学习该层级的任务特定特征
            adapted_state = adapter(hidden_state)
            adapted_hidden_states.append(adapted_state)
        
        # 使用最后一层的Adapter输出进行分类
        # 为什么使用最后一层？
        # 1. 最后一层包含了最高级的语义抽象，最适合做分类决策
        # 2. 经过所有Adapter处理后，包含了从底层到高层的任务适配信息
        # 3. 遵循BERT的设计：[CLS]标记在最后一层的表示用于分类任务
        pooled_output = adapted_hidden_states[-1][:, 0, :]  # 取[CLS]标记的表示
        
        # 使用原始分类器进行最终预测
        # 注意：分类器的参数也被冻结了，这确保了只有Adapter在学习任务特定知识
        logits = self.base_model.classifier(pooled_output)
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

# --- 演示数据 ---
# 正负样本对比鲜明，便于观察学习效果
data = {
    'text': ["很好用", "太差了", "不错", "很糟糕", "非常棒", "很失望"],
    'label': [1, 0, 1, 0, 1, 0]  # 1:正面, 0:负面
}
dataset = Dataset.from_pandas(pd.DataFrame(data)).rename_column("label", "labels")

def tokenize_function(examples):
    # max_length=32？
    # 1. 中文短文本通常不超过32个token，避免不必要的padding
    # 2. 减少计算量，在CPU上运行更快
    # 3. 对于情感分析任务，32个token足够捕获关键信息
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(tokenize_function, batched=True).remove_columns(["text"])

# --- 创建带Adapter的模型 ---
print(f"\n=== 创建Adapter模型 ===")
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
# adapter_dim=8？
# 1. 平衡效果与效率：8维足够学习基本的任务特定模式，但参数量很少
# 2. 避免过拟合：在小数据集上，过大的adapter_dim容易过拟合
# 3. 计算友好：小维度在CPU上计算更快
adapter_model = ModelWithAdapter(base_model, adapter_dim=8)

# 统计参数 - 通过对比总参数量和可训练参数量，可以直观看到Adapter的参数效率
total_params = sum(p.numel() for p in adapter_model.parameters())
trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")
print("💡 关键观察：只需训练不到1%的参数，这就是Adapter微调的核心优势！")

# --- 快速训练演示 ---
# 训练参数的选择都有特定原因：
training_args = TrainingArguments(
    output_dir="./adapter_output",
    num_train_epochs=2,         # 训练轮次：
                        # 1. Adapter参数少，收敛快，避免过拟合
                        # 2. 演示目的，快速看到效果
                        # 3. 在小数据集上，过多轮次容易过拟合
    per_device_train_batch_size=2,  # 小批次适合小数据集和CPU环境
    learning_rate=1e-3,  # 为什么用1e-3而不是常见的1e-5？
                        # 因为只训练Adapter参数，可以用更大的学习率加速收敛
    logging_steps=1,     # 每步都记录，方便观察训练过程
    save_steps=100,      # 不频繁保存，减少I/O开销
    report_to="none",    # 不使用wandb等工具，简化演示环境
    remove_unused_columns=False,  # 保持数据完整性
)

trainer = Trainer(
    model=adapter_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print(f"\n=== 开始Adapter微调 ===")
print("💡 核心理念：只有插入的Adapter模块参数会被训练，基础模型参数保持冻结")
print("这样既保护了预训练知识，又让模型适应新任务！")
trainer.train()

# --- 演示推理 ---
print(f"\n=== Adapter推理演示 ===")
adapter_model.eval()  # 切换到评估模式，关闭dropout等训练特有的操作

# 确保模型在CPU上运行（适配M1环境）
# 为什么要显式指定CPU？
# 1. M1芯片的MPS可能不稳定，CPU更可靠
# 2. 小模型在CPU上推理速度已经很快
# 3. 避免设备不匹配的错误
adapter_model = adapter_model.cpu()
test_texts = ["非常棒", "很失望", "还可以"]

print("💡 观察要点：看看训练后的Adapter如何影响模型的预测结果")
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    # 确保输入也在CPU上，避免设备不匹配
    inputs = {k: v.cpu() for k, v in inputs.items()}
    
    with torch.no_grad():  # 推理时不需要计算梯度，节省内存和计算
        outputs = adapter_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        result = "正面" if prediction == 1 else "负面"
        print(f"文本: '{text}' -> 预测: {result} (置信度: {confidence:.3f})")
 