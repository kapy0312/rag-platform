import os
from typing import List

import httpx
from rank_bm25 import BM25Okapi

# ============================================================
# 本地端／雲端切換：改 RAG_ENV 這一個環境變數就好
#   RAG_ENV=local（預設）-> 打本機 Ollama 的 bge-m3
#   RAG_ENV=cloud         -> 打 DeepInfra 代管的 bge-m3（同一顆模型權重）
# ============================================================
RAG_ENV = os.getenv("RAG_ENV", "local")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.89.23.28:11434")
OLLAMA_EMBED_MODEL = "bge-m3"

DEEPINFRA_EMBED_URL = "https://api.deepinfra.com/v1/openai/embeddings"
DEEPINFRA_EMBED_MODEL = "BAAI/bge-m3"

if RAG_ENV == "cloud":
    DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]


async def get_dense_embedding(text: str) -> List[float]:
    embeddings = await _get_embeddings([text])
    return embeddings[0]


async def get_dense_embeddings_batch(texts: List[str]) -> List[List[float]]:
    return await _get_embeddings(texts)


async def _get_embeddings(texts: List[str]) -> List[List[float]]:
    if RAG_ENV == "cloud":
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                DEEPINFRA_EMBED_URL,
                headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                json={"input": texts, "model": DEEPINFRA_EMBED_MODEL, "encoding_format": "float"},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # OpenAI 相容格式每筆帶 index，API 不保證順序跟輸入一致，照 index 排序後再取值
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]

    # local：打 Ollama /api/embed
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def tokenize(text: str) -> List[str]:
    """簡單 character-level bigram tokenizer，對中文友好"""
    text = text.lower()
    tokens = []
    words = text.split()
    for word in words:
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            tokens.extend([word[i:i+2] for i in range(len(word)-1)] or [word])
        else:
            tokens.append(word)
    return tokens


def build_bm25(corpus: List[str]) -> BM25Okapi:
    tokenized = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized)


def bm25_scores(bm25: BM25Okapi, query: str, n: int) -> List[int]:
    """回傳 top-n 的 corpus index（分數降冪）"""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:n]


def reciprocal_rank_fusion(
    dense_ids: List[str],
    sparse_ids: List[str],
    k: int = 60
) -> List[str]:
    """RRF fusion，回傳依分數排序的 id 清單"""
    scores: dict[str, float] = {}
    for rank, id_ in enumerate(dense_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
    for rank, id_ in enumerate(sparse_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)