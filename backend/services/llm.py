import httpx
import json
from typing import AsyncGenerator, List

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:14b"


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
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "think": False,
        "stream": True,
        "options": {
            "num_predict": 1024,
        }
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json=payload
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
