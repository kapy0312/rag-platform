from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from aiosqlite import Connection
from database import get_db
from models import DocumentOut
from services.processor import process_document
from services.vector_store import delete_by_document_id
import aiofiles
import uuid
import os

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def _run_processing(pdf_path, doc_id, category_id, filename, original_name, db_path):
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute("UPDATE documents SET status='processing' WHERE id=?", (doc_id,))
        await db.commit()
    try:
        chunk_count = await process_document(pdf_path, doc_id, category_id, filename, original_name)
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE documents SET status='done', chunk_count=? WHERE id=?",
                (chunk_count, doc_id)
            )
            await db.commit()
    except Exception as e:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE documents SET status='error', error_msg=? WHERE id=?",
                (str(e), doc_id)
            )
            await db.commit()


@router.get("", response_model=list[DocumentOut])
async def list_documents(category_id: int, db: Connection = Depends(get_db)):
    async with db.execute(
        "SELECT * FROM documents WHERE category_id = ? ORDER BY created_at DESC",
        (category_id,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category_id: int = Form(...),
    db: Connection = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受 PDF 檔案")

    async with db.execute("SELECT id FROM categories WHERE id = ?", (category_id,)) as cur:
        if not await cur.fetchone():
            raise HTTPException(status_code=404, detail="類別不存在")

    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.pdf"
    pdf_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(pdf_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    await db.execute(
        "INSERT INTO documents (id, category_id, filename, original_name, status) VALUES (?, ?, ?, ?, 'pending')",
        (doc_id, category_id, filename, file.filename)
    )
    await db.commit()

    from database import DB_PATH
    background_tasks.add_task(
        _run_processing, pdf_path, doc_id, category_id,
        filename, file.filename, str(DB_PATH)
    )

    async with db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)) as cur:
        row = await cur.fetchone()
    return dict(row)


@router.get("/{doc_id}/status")
async def get_status(doc_id: str, db: Connection = Depends(get_db)):
    async with db.execute(
        "SELECT id, status, chunk_count, error_msg FROM documents WHERE id = ?", (
            doc_id,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="文件不存在")
    return dict(row)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, db: Connection = Depends(get_db)):
    async with db.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)) as cur:
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="文件不存在")

    await delete_by_document_id(doc_id)

    pdf_path = os.path.join(UPLOAD_DIR, row["filename"])
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    await db.commit()
