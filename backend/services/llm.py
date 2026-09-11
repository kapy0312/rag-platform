import os
import json
from typing import AsyncGenerator, List

import httpx

# ============================================================
# 本地端／雲端切換：改 RAG_ENV 這一個環境變數就好
#   RAG_ENV=local（預設）-> 打本機 Ollama 的 qwen3:14b
#   RAG_ENV=cloud         -> 打 Google Gemini 2.5 Flash-Lite（透過 LiteLLM）
# ============================================================
RAG_ENV = os.getenv("RAG_ENV", "local")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.89.23.28:11434")
OLLAMA_LLM_MODEL = "qwen3:14b"
OLLAMA_REWRITE_MODEL = "qwen3:8b"

CLOUD_LLM_MODEL = "gemini/gemini-2.5-flash-lite"
CLOUD_REWRITE_MODEL = "gemini/gemini-2.5-flash-lite"

if RAG_ENV == "cloud":
    import litellm


def build_prompt(question: str, chunks: List[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[來源 {i+1}｜{c['original_name']} 第 {c['page_number']} 頁]\n{c['text']}"
        )
    context = "\n\n".join(context_parts)
    return f"""你是一位專業技術文件助理，請根據以下參考資料回答問題。
只根據提供的資料回答，若資料不足請說明。回答使用繁體中文。

參考資料：
{context}

問題：{question}

回答："""


async def stream_llm(question: str, chunks: List[dict]) -> AsyncGenerator[str, None]:
    prompt = build_prompt(question, chunks)

    if RAG_ENV == "cloud":
        response = await litellm.acompletion(
            model=CLOUD_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=1024,
        )
        async for part in response:
            token = part.choices[0].delta.content
            if token:
                yield token
        return

    # local：打 Ollama /api/generate
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "think": False,
        "stream": True,
        "options": {"num_predict": 1024},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{OLLAMA_URL}/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


async def rewrite_query(question: str) -> str:
    """把口語問題改寫成適合檢索的詞彙，用小模型跑，只做這一件事"""
    prompt = f"""將以下問題改寫成更適合在文件中檢索的關鍵字組合，保留原意，
只輸出改寫後的句子本身，不要加任何說明或標點以外的文字，
必須使用繁體中文，不可以輸出簡體字：

問題：{question}
改寫："""

    if RAG_ENV == "cloud":
        response = await litellm.acompletion(
            model=CLOUD_REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # local：打 Ollama /api/generate
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_REWRITE_MODEL,
            "prompt": prompt,
            "stream": False,
        })
        resp.raise_for_status()
        return resp.json()["response"].strip()