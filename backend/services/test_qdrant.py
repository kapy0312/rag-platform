"""
本機測試 vector_store_cloud.py（Qdrant Cloud 版本）。

用法：
    $env:QDRANT_URL="你的Cluster URL"
    $env:QDRANT_API_KEY="你的API Key"
    python test_qdrant.py
"""
import asyncio
import random
import vector_store_cloud as vector_store

FAKE_VECTOR = [random.random() for _ in range(vector_store.VECTOR_DIM)]


async def main():
    print("=== 1. 確保 collection 存在 ===")
    await vector_store.ensure_collection()
    print("OK")

    print("\n=== 2. 寫入一筆假 chunk ===")
    await vector_store.upsert_chunks(
        document_id="test-doc-001",
        category_id=999,
        chunks=[{
            "chunk_index": 0,
            "page_number": 1,
            "text": "這是一段測試文字，用來驗證 Qdrant Cloud 連線正常。",
            "embedding": FAKE_VECTOR,
        }],
        filename="test-doc-001.pdf",
        original_name="測試文件.pdf",
    )
    print("OK")

    print("\n=== 3. 用同一個向量做 dense search（應該找回剛剛寫入的那筆，分數接近 1.0）===")
    results = await vector_store.search_dense(FAKE_VECTOR, top_k=3, category_id=999)
    for r in results:
        print(f"score={r.score:.4f}  text={r.payload['text']}")

    print("\n=== 4. get_all_chunks 撈出該分類全部 chunk ===")
    chunks = await vector_store.get_all_chunks(category_id=999)
    print(chunks)

    print("\n=== 5. 刪除剛剛寫入的文件 ===")
    await vector_store.delete_by_document_id("test-doc-001")
    print("OK")

    print("\n=== 6. 驗證刪除後查不到 ===")
    chunks_after = await vector_store.get_all_chunks(category_id=999)
    print(f"刪除後（應為空清單）：{chunks_after}")

    if chunks_after == []:
        print("\n>>> 全部通過，Qdrant Cloud 版 vector_store_cloud.py 邏輯正確 <<<")
    else:
        print("\n>>> 有異常，刪除沒有完全生效 <<<")


if __name__ == "__main__":
    asyncio.run(main())