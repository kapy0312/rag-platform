from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# 撈出「致富心態_v2.pdf」這份文件全部的 chunk（1000 筆上限，這本書 173 個 chunk 遠遠夠用）
results, _ = client.scroll(
    collection_name="rag_documents",
    scroll_filter={
        "must": [
            {"key": "original_name", "match": {"value": "致富心態_v2.pdf"}}
        ]
    },
    limit=1000,
    with_payload=True,
    with_vectors=False,
)

print(f"這份文件總共有 {len(results)} 個 chunk")
print()

found = False
for point in results:
    text = point.payload.get("text", "")
    if "所得超過一定門檻的人" in text:
        found = True
        print("=== 找到目標句子所在的 chunk ===")
        print("point id:", point.id)
        print("payload 裡記錄的 page_number:", point.payload.get("page_number"))
        print("chunk_index:", point.payload.get("chunk_index"))
        print("完整內容：")
        print(text)
        print()

if not found:
    print("完全沒有任何一個 chunk 包含這句話——代表這段內容根本沒被存進 Qdrant")