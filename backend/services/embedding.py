import os
import httpx
from rank_bm25 import BM25Okapi
from typing import List

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.89.23.28:11434")
EMBED_MODEL = "bge-m3"

async def get_dense_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": [text]}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


async def get_dense_embeddings_batch(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts}
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
