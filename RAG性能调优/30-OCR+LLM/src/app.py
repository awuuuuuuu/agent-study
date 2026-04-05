import os
import streamlit as st
from PIL import Image
import io
import base64
from http import HTTPStatus
import dashscope
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv(), override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 【核心修复1】显式设置 DashScope 库的全局 API Key
dashscope.api_key = DASHSCOPE_API_KEY

# 商品数据库
product_database = [
    "浅蓝色的衬衫，棉质面料，适合夏季穿着，尺码从S到XL",
    "红色连衣裙，棉质面料，适合夏季穿着，尺码从S到XL",
    "白色T恤，纯棉材质，圆领短袖，百搭款式",
    "黑色皮鞋，真皮材质，商务正装，耐磨防滑",
    "运动鞋，轻便透气，适合跑步健身，多色可选",
]

def get_multimodal_embedding(image_bytes):
    """获取图片的多模态向量"""
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{image_base64}"

    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{'image': image_data}]
    )

    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        # 【核心修复2】把底层的图片解析报错直接暴露在网页上
        st.error("❌ 图片向量化调用失败！")
        st.error(f"错误码: {resp.code} | 错误信息: {resp.message}")
        return None

def get_text_embedding(text):
    """获取文本的多模态向量"""
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{'text': text}]
    )
    
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        # 【核心修复2】把底层的文本解析报错直接暴露在网页上
        st.error(f"❌ 文本 '{text[:10]}...' 向量化失败！")
        st.error(f"错误码: {resp.code} | 错误信息: {resp.message}")
        return None

def setup_product_vectors():
    """初始化商品向量数据库"""
    product_embeddings = []
    
    # 增加一个进度条，让加载过程更直观
    progress_text = "正在调用千问多模态模型生成商品向量..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(product_database)
    for i, product in enumerate(product_database):
        embedding = get_text_embedding(product)
        if embedding:
            product_embeddings.append({
                'text': product,
                'embedding': embedding
            })
        # 更新进度条
        my_bar.progress((i + 1) / total, text=f"已处理: {i+1}/{total}")
        
    my_bar.empty() # 加载完清空进度条
    return product_embeddings

def search_similar_products(image_bytes, product_embeddings, top_k=3):
    """多模态向量相似度检索"""
    image_embedding = get_multimodal_embedding(image_bytes)
    
    if not image_embedding:
        return [] # 如果图片向量化失败，提前终止
        
    if not product_embeddings:
        st.warning("底层的商品库向量为空，无法进行检索匹配。")
        return []
    
    similarities = []
    image_vec = np.array(image_embedding).reshape(1, -1)
    
    for product_data in product_embeddings:
        product_vec = np.array(product_data['embedding']).reshape(1, -1)
        similarity = cosine_similarity(image_vec, product_vec)[0][0]
        similarities.append({
            'text': product_data['text'],
            'similarity': similarity
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

def main():
    st.set_page_config(page_title="多模态图搜文", layout="wide")
    
    # 侧边栏：开发者调试工具
    with st.sidebar:
        st.header("🛠️ 开发者工具")
        if st.button("🗑️ 清理缓存并重新加载", use_container_width=True):
            if 'product_embeddings' in st.session_state:
                del st.session_state['product_embeddings']
            st.rerun() # 强制刷新页面

    st.title("多模态向量商品检索系统")
    
    if not DASHSCOPE_API_KEY:
        st.error("请设置DASHSCOPE_API_KEY环境变量")
        st.stop() # 直接停止渲染后续组件
    
    # 初始化商品向量数据库
    if 'product_embeddings' not in st.session_state:
        st.session_state.product_embeddings = setup_product_vectors()
        
    # 状态展示
    loaded_count = len(st.session_state.product_embeddings)
    if loaded_count > 0:
        st.success(f"✅ 已成功加载 {loaded_count} 个商品的多模态向量。")
    else:
        st.error("⚠️ 当前加载了 0 个商品向量！请检查上面的红色报错信息。")
    
    # 图片上传区
    st.write("---")
    uploaded_file = st.file_uploader("📤 上传商品图片 (支持JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2]) # 左右分栏让 UI 更好看
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 上传的商品图片", use_container_width=True)
            
            # 转换图片为字节 (包含健壮的格式处理)
            img_byte_arr = io.BytesIO()
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
        
        with col2:
            st.write("### 🔍 检索结果")
            if st.button("开始多模态检索", type="primary", use_container_width=True):
                with st.spinner("正在将图片映射到高维多模态空间并检索..."):
                    results = search_similar_products(
                        img_byte_arr, 
                        st.session_state.product_embeddings
                    )
                    
                    if results:
                        for i, item in enumerate(results):
                            # 用 Markdown 画一个简单的进度条来展示相似度
                            sim_score = item['similarity']
                            st.markdown(f"**Top {i+1}** | 相似度: `{sim_score:.3f}`")
                            st.progress(float(max(0, min(1, sim_score)))) # 确保在 0-1 之间
                            st.info(f"**商品描述:** {item['text']}")
                    elif loaded_count > 0:
                        st.write("未找到相似商品。")
    
    # 显示原始商品数据库
    st.write("---")
    with st.expander("📚 查看原始文本商品数据库"):
        for i, product in enumerate(product_database):
            st.write(f"{i+1}. {product}")

if __name__ == "__main__":
    main()