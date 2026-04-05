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
load_dotenv(find_dotenv(), override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = DASHSCOPE_BASE_URL

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
    return None

def get_text_embedding(text):
    """获取文本的多模态向量"""
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{'text': text}]
    )
    
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    return None

def setup_product_vectors():
    """初始化商品向量数据库"""
    product_embeddings = []
    for product in product_database:
        embedding = get_text_embedding(product)
        if embedding:
            product_embeddings.append({
                'text': product,
                'embedding': embedding
            })
    return product_embeddings

def search_similar_products(image_bytes, product_embeddings, top_k=3):
    """多模态向量相似度检索"""
    image_embedding = get_multimodal_embedding(image_bytes)
    if not image_embedding or not product_embeddings:
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
    st.title("多模态向量商品检索系统")
    
    if not DASHSCOPE_API_KEY:
        st.error("请设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 初始化商品向量数据库
    if 'product_embeddings' not in st.session_state:
        with st.spinner("初始化商品向量数据库..."):
            st.session_state.product_embeddings = setup_product_vectors()
        st.success(f"已加载 {len(st.session_state.product_embeddings)} 个商品向量")
    
    # 图片上传
    uploaded_file = st.file_uploader("上传商品图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的商品图片", width=300)
        
        # 转换图片为字节
        img_byte_arr = io.BytesIO()
        # 处理RGBA格式图片，转换为RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 执行检索
        if st.button("检索相似商品"):
            with st.spinner("正在检索..."):
                results = search_similar_products(
                    img_byte_arr, 
                    st.session_state.product_embeddings
                )
                
                if results:
                    st.write("### 检索结果")
                    for i, item in enumerate(results):
                        st.write(f"{i+1}. [相似度: {item['similarity']:.3f}] {item['text']}")
                else:
                    st.write("未找到相似商品")
    
    # 显示商品数据库
    with st.expander("商品数据库"):
        for i, product in enumerate(product_database):
            st.write(f"{i+1}. {product}")

if __name__ == "__main__":
    main()