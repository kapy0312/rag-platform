from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models import QueryRequest
from services.embedding import get_dense_embedding, build_bm25, bm25_scores, reciprocal_rank_fusion
from services.vector_store import search_dense, get_all_chunks
from services.llm import stream_llm, rewrite_query
import json

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("")
async def query(body: QueryRequest):
    search_text = body.question
    if body.use_rewrite:
        search_text = await rewrite_query(body.question)
        print(f"[rewrite] 原問題：{body.question} → 改寫後：{search_text}")
        
    # 1. Dense embedding（改寫後的句子同時餵給 dense 跟 BM25 兩條路）
    query_vector = await get_dense_embedding(search_text)

    # 2. Dense search（top 10）
    dense_results = await search_dense(query_vector, top_k=10, category_id=body.category_id)
    dense_ids = [str(r.id) for r in dense_results]
    dense_map = {str(r.id): r.payload for r in dense_results}

    # 3. BM25 sparse search
    all_chunks = await get_all_chunks(category_id=body.category_id)
    if not all_chunks:
        raise HTTPException(status_code=400, detail="該類別尚無文件，請先上傳 PDF")

    corpus_texts = [c["payload"]["text"] for c in all_chunks]
    bm25 = build_bm25(corpus_texts)
    top_bm25_indices = bm25_scores(bm25, search_text, n=10)
    sparse_ids = [all_chunks[i]["id"] for i in top_bm25_indices]
    sparse_map = {all_chunks[i]["id"]: all_chunks[i]
                  ["payload"] for i in top_bm25_indices}

    # 4. RRF fusion
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)
    merged_map = {**sparse_map, **dense_map}

    # 5. 取 top_k
    top_ids = fused_ids[:body.top_k]
    top_chunks = [merged_map[id] for id in top_ids if id in merged_map]

    # log dense scores
    print("\n=== Dense Search Scores ===")
    for r in dense_results:
        p = r.payload
        print(f"  [{r.score:.4f}] {p['original_name']} p.{p['page_number']} chunk{p['chunk_index']}")
    print(f"=== Top {body.top_k} after RRF ===")
    for i, c in enumerate(top_chunks):
        print(f"  [{i+1}] {c['original_name']} p.{c['page_number']}")

    # 6. SSE streaming
    sources = [
        {
            "filename": c["filename"],
            "original_name": c["original_name"],
            "page_number": c["page_number"],
            "snippet": c["text"][:120],
        }
        for c in top_chunks
    ]

    async def event_stream():
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
        async for token in stream_llm(body.question, top_chunks):
            yield f"event: token\ndata: {json.dumps(token, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
