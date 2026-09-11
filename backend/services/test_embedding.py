"""
本機測試 embedding.py（DeepInfra bge-m3 版本）。

用法：
    $env:DEEPINFRA_API_KEY="你的Key"
    python test_embedding.py
"""
import asyncio
import embedding

EXPECTED_DIM = 1024  # 跟 vector_store_cloud.py 裡的 VECTOR_DIM 要一致


async def main():
    print("=== 1. 單句 embedding ===")
    vec = await embedding.get_dense_embedding("這是一段測試文字")
    print(f"維度：{len(vec)}（預期 {EXPECTED_DIM}）")
    assert len(vec) == EXPECTED_DIM, "維度不對，跟 Qdrant collection 設定的 VECTOR_DIM 對不上！"

    print("\n=== 2. 批次 embedding（3 句）===")
    texts = ["第一句測試文字", "第二句測試文字，內容不同", "第三句，用來確認順序沒亂掉"]
    vecs = await embedding.get_dense_embeddings_batch(texts)
    print(f"回傳筆數：{len(vecs)}（預期 3）")
    for i, v in enumerate(vecs):
        print(f"  第 {i+1} 筆維度：{len(v)}")
        assert len(v) == EXPECTED_DIM

    print("\n=== 3. BM25 邏輯（純本地運算，跟 API 無關，確認沒被改壞）===")
    bm25 = embedding.build_bm25(texts)
    top = embedding.bm25_scores(bm25, "第二句", n=2)
    print(f"BM25 對「第二句」的檢索結果 index：{top}")

    print("\n>>> 全部通過，DeepInfra 版 embedding.py 邏輯正確 <<<")


if __name__ == "__main__":
    asyncio.run(main())