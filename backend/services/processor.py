import fitz  # PyMuPDF
import uuid
import asyncio
from typing import List
from services.embedding import get_dense_embeddings_batch
from services.vector_store import upsert_chunks

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_text_by_page(pdf_path: str) -> List[dict]:
    """回傳 [{page_number, text}]"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def sliding_window_chunks(pages: List[dict]) -> List[dict]:
    """
    把所有 page 文字串接後做 sliding window chunking，
    同時保留來源 page_number（取 chunk 起始位置對應的 page）
    """
    # 建立 (cumulative_char_index, page_number) 映射
    full_text = ""
    page_boundaries = []  # (start_char, page_number)
    for p in pages:
        page_boundaries.append((len(full_text), p["page_number"]))
        full_text += p["text"] + "\n"

    def char_to_page(char_idx: int) -> int:
        page_num = 1
        for start, pn in page_boundaries:
            if char_idx >= start:
                page_num = pn
            else:
                break
        return page_num

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(full_text):
        end = min(start + CHUNK_SIZE, len(full_text))
        text = full_text[start:end].strip()
        if text:
            chunks.append({
                "chunk_index": chunk_index,
                "page_number": char_to_page(start),
                "text": text,
            })
            chunk_index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


async def process_document(
    pdf_path: str,
    document_id: str,
    category_id: int,
    filename: str,
    original_name: str,
) -> int:
    """
    完整 pipeline：解析 → chunking → embedding → 寫入 Qdrant
    回傳 chunk 數量
    """
    pages = extract_text_by_page(pdf_path)
    if not pages:
        raise ValueError("PDF 無可解析文字（可能是掃描圖片型 PDF）")

    chunks = sliding_window_chunks(pages)
    if not chunks:
        raise ValueError("chunking 結果為空")

    # 批次 embedding（每批 16 個避免 OOM）
    BATCH = 16
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = await get_dense_embeddings_batch(texts)
        for j, emb in enumerate(embeddings):
            chunks[i + j]["embedding"] = emb

    await upsert_chunks(
        document_id=document_id,
        category_id=category_id,
        chunks=chunks,
        filename=filename,
        original_name=original_name,
    )

    return len(chunks)
