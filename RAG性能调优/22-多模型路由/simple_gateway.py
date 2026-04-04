"""LLM 智能网关"""
import asyncio
import json
import time
import uuid
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from typing import List, Optional

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

app = FastAPI(title="LLM 智能网关", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

engines = {
    "fast": {"model": "gpt-3.5-turbo", "tier": "快速"},
    "balanced": {"model": "gpt-4o", "tier": "平衡"},
    "premium": {"model": "gpt-5.2", "tier": "高级"}
}

def select_engine(question: str):
    """智能选择引擎"""
    content = question.lower()

    # 复杂度关键词匹配
    complex_words = ["设计", "架构", "分析", "系统", "算法", "优化"]
    medium_words = ["解释", "原理", "方法", "流程"]
    simple_words = ["什么是", "定义", "翻译"]
    
    if any(word in content for word in complex_words) or len(question) > 50:
        engine_type = "premium"
    elif any(word in content for word in medium_words) or len(question) > 20:
        engine_type = "balanced"  
    else:
        engine_type = "fast"

    engine = engines[engine_type]
    print(f"选择引擎: {engine['model']} ({engine['tier']})")
    return engine

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """处理聊天请求"""
    user_question = request.messages[-1].content
    engine = select_engine(user_question)

    try:
        if request.stream:
            return StreamingResponse(
                stream_response(request, engine),
                media_type="text/plain"
            )
        else:
            return await complete_response(request, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def stream_response(request: ChatRequest, engine):
    """流式响应"""
    if client:
        try:
            response = client.chat.completions.create(
                model=engine['model'],
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    data = {"choices": [{"delta": {"content": chunk.choices[0].delta.content}}]}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        except:
            pass

    for part in ["这是", "来自", engine['tier'], "模型", "的", "响应"]:
        data = {"choices": [{"delta": {"content": part}}]}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.2)
    yield "data: [DONE]\n\n"

async def complete_response(request: ChatRequest, engine):
    """完整响应"""
    print(f"使用{engine['tier']}模型处理请求...")
    
    if client:
        try:
            response = client.chat.completions.create(
                model=engine['model'],
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                timeout=25  # 设置API调用超时
            )
            
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content
                    },
                    "finish_reason": response.choices[0].finish_reason
                }]
            }
        except Exception as e:
            print(f"API调用失败，使用模拟响应: {e}")
    
    # 模拟响应 - 根据复杂度生成不同长度的回答
    user_question = request.messages[-1].content
    if engine['tier'] == "快速":
        content = f"这是一个简单问题的快速回答：{user_question[:20]}..."
    elif engine['tier'] == "平衡":
        content = f"这是一个中等复杂度问题的详细解答。问题：{user_question}。这需要更全面的分析和说明。"
    else:  # 高级
        content = f"这是一个复杂问题的深入分析：{user_question}。需要从多个角度进行详细阐述，包括技术架构、实现方案、性能优化等多个方面的综合考虑。"
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine['model'],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "engines": len(engines)}

@app.get("/stats")
async def stats():
    return {
        "engine_types": list(engines.keys()),
        "api_available": client is not None
    }

if __name__ == "__main__":
    print(" LLM 智能网关启动")
    print(f" 地址: http://localhost:8000")
    print(f" 引擎: {len(engines)}个")
    print(f" API: {'可用' if client else '模拟'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)