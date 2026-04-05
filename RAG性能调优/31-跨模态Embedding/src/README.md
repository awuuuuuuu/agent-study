
## 技术架构

```
用户上传图片 → MultiModal Embedding → 图片向量 → 余弦相似度计算 → 相似商品排序
```

## 核心组件

### 1. 向量提取
- `get_multimodal_embedding()` - 提取图片向量
- `get_text_embedding()` - 提取文本向量

### 2. 向量存储
- `setup_product_vectors()` - 初始化商品向量数据库

### 3. 相似度检索
- `search_similar_products()` - 基于余弦相似度的商品检索

## 安装运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置API密钥：
```bash
export DASHSCOPE_API_KEY='your_api_key_here'
```

3. 启动应用：
```bash
streamlit run app.py
```

