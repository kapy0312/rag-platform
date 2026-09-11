import os
import uuid
from typing import List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    FilterSelector, PayloadSchemaType
)

# ============================================================
# 本地端／雲端切換：改 RAG_ENV 這一個環境變數就好
#   RAG_ENV=local（預設）-> 接本機 Docker 跑的 Qdrant（localhost:6333）
#   RAG_ENV=cloud         -> 接 Qdrant Cloud（需要 QDRANT_URL / QDRANT_API_KEY）
# ============================================================
RAG_ENV = os.getenv("RAG_ENV", "local")

COLLECTION = "rag_documents"
VECTOR_DIM = 1024

if RAG_ENV == "cloud":
    QDRANT_URL = os.environ["QDRANT_URL"]
    QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = AsyncQdrantClient(host="localhost", port=6333)


async def ensure_collection():
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if COLLECTION not in names:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )

    # category_id / document_id 這兩個過濾用欄位一律確保有索引——
    # Qdrant Cloud 沒索引會直接報 400，本機版沒索引也能查（只是慢），
    # 兩邊都建索引，行為統一、不用在這裡另外分支
    info = await client.get_collection(COLLECTION)
    existing_fields = set(info.payload_schema.keys()) if info.payload_schema else set()

    if "category_id" not in existing_fields:
        await client.create_payload_index(
            collection_name=COLLECTION,
            field_name="category_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
    if "document_id" not in existing_fields:
        await client.create_payload_index(
            collection_name=COLLECTION,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )


async def upsert_chunks(
    document_id: str,
    category_id: int,
    chunks: List[dict],
    filename: str,
    original_name: str
):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload={
                "document_id": document_id,
                "category_id": category_id,
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "filename": filename,
                "original_name": original_name,
            }
        )
        for chunk in chunks
    ]
    await client.upsert(collection_name=COLLECTION, points=points)


async def search_dense(
    query_vector: List[float],
    top_k: int = 10,
    category_id: Optional[int] = None
):
    query_filter = None
    if category_id is not None:
        query_filter = Filter(
            must=[FieldCondition(key="category_id", match=MatchValue(value=category_id))]
        )
    results = await client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True
    )
    return results


async def get_all_chunks(category_id: Optional[int] = None) -> List[dict]:
    scroll_filter = None
    if category_id is not None:
        scroll_filter = Filter(
            must=[FieldCondition(key="category_id", match=MatchValue(value=category_id))]
        )
    all_points = []
    offset = None
    while True:
        result, next_offset = await client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(result)
        if next_offset is None:
            break
        offset = next_offset
    return [{"id": str(p.id), "payload": p.payload} for p in all_points]


async def delete_by_document_id(document_id: str):
    await client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ))
    )


async def delete_by_category_id(category_id: int):
    await client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="category_id", match=MatchValue(value=category_id))]
        ))
    )