from fastapi import APIRouter, Depends, HTTPException
from aiosqlite import Connection
from database import get_db
from models import CategoryCreate, CategoryOut
from services.vector_store import delete_by_category_id
import aiosqlite

router = APIRouter(prefix="/api/categories", tags=["categories"])


@router.get("", response_model=list[CategoryOut])
async def list_categories(db: Connection = Depends(get_db)):
    async with db.execute("SELECT * FROM categories ORDER BY created_at DESC") as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.post("", response_model=CategoryOut, status_code=201)
async def create_category(body: CategoryCreate, db: Connection = Depends(get_db)):
    try:
        async with db.execute(
            "INSERT INTO categories (name, description, color) VALUES (?, ?, ?)",
            (body.name, body.description, body.color)
        ) as cur:
            cat_id = cur.lastrowid
        await db.commit()
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=409, detail="類別名稱已存在")

    async with db.execute("SELECT * FROM categories WHERE id = ?", (cat_id,)) as cur:
        row = await cur.fetchone()
    return dict(row)


@router.delete("/{cat_id}", status_code=204)
async def delete_category(cat_id: int, db: Connection = Depends(get_db)):
    async with db.execute("SELECT id FROM categories WHERE id = ?", (cat_id,)) as cur:
        if not await cur.fetchone():
            raise HTTPException(status_code=404, detail="類別不存在")

    # 刪除 Qdrant 向量
    try:
        await delete_by_category_id(cat_id)
    except Exception as e:
        print(f"[WARN] Qdrant delete failed: {e}")

    # 刪除 SQLite 文件記錄與類別
    await db.execute("DELETE FROM documents WHERE category_id = ?", (cat_id,))
    await db.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
    await db.commit()
