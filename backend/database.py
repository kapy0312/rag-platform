import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# 本地端／雲端切換：改 RAG_ENV 這一個環境變數就好
#   RAG_ENV=local（預設）-> 本機 SQLite（backend/rag.db）
#   RAG_ENV=cloud         -> AWS DynamoDB（三張表要先用 AWS CLI 建好）
# 兩條路對外暴露完全一樣的函式介面（list_categories / create_category /
# get_category / delete_category / list_documents / get_document /
# create_document / update_document_status / delete_document），
# categories.py / documents.py 不用管現在是哪個環境，直接呼叫就好
# ============================================================
RAG_ENV = os.getenv("RAG_ENV", "local")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------- cloud：DynamoDB ----------------
if RAG_ENV == "cloud":
    import boto3
    from botocore.exceptions import ClientError

    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

    categories_table = _dynamodb.Table("rag_categories")
    categories_by_name_table = _dynamodb.Table("rag_categories_by_name")
    documents_table = _dynamodb.Table("rag_documents_meta")

    # 固定保留給計數器用的 id，永遠不會被真正的分類拿去用
    _COUNTER_ID = 0

    def get_next_category_id() -> int:
        resp = categories_table.update_item(
            Key={"id": _COUNTER_ID},
            UpdateExpression="ADD next_id :incr",
            ExpressionAttributeValues={":incr": 1},
            ReturnValues="UPDATED_NEW",
        )
        return int(resp["Attributes"]["next_id"])

    def list_categories() -> list[dict]:
        resp = categories_table.scan()
        items = [i for i in resp["Items"] if i["id"] != _COUNTER_ID]
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def get_category(cat_id: int) -> dict | None:
        resp = categories_table.get_item(Key={"id": cat_id})
        return resp.get("Item")

    def create_category(name: str, description: str | None, color: str) -> dict:
        try:
            categories_by_name_table.put_item(
                Item={"name": name, "category_id": None},
                ConditionExpression="attribute_not_exists(#n)",
                ExpressionAttributeNames={"#n": "name"},
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ValueError("DUPLICATE_NAME")
            raise

        cat_id = get_next_category_id()
        item = {
            "id": cat_id,
            "name": name,
            "description": description,
            "color": color,
            "created_at": _now_iso(),
        }
        categories_table.put_item(Item=item)
        categories_by_name_table.update_item(
            Key={"name": name},
            UpdateExpression="SET category_id = :cid",
            ExpressionAttributeValues={":cid": cat_id},
        )
        return item

    def delete_category(cat_id: int) -> dict | None:
        cat = get_category(cat_id)
        if cat is None:
            return None
        docs = list_documents(cat_id)
        if docs:
            with documents_table.batch_writer() as batch:
                for d in docs:
                    batch.delete_item(Key={"id": d["id"]})
        categories_by_name_table.delete_item(Key={"name": cat["name"]})
        categories_table.delete_item(Key={"id": cat_id})
        return cat

    def list_documents(category_id: int) -> list[dict]:
        resp = documents_table.query(
            IndexName="category_id-created_at-index",
            KeyConditionExpression="category_id = :cid",
            ExpressionAttributeValues={":cid": category_id},
            ScanIndexForward=False,
        )
        return resp["Items"]

    def get_document(doc_id: str) -> dict | None:
        resp = documents_table.get_item(Key={"id": doc_id})
        return resp.get("Item")

    def create_document(doc_id: str, category_id: int, filename: str, original_name: str) -> dict:
        item = {
            "id": doc_id,
            "category_id": category_id,
            "filename": filename,
            "original_name": original_name,
            "chunk_count": 0,
            "status": "pending",
            "error_msg": None,
            "created_at": _now_iso(),
        }
        documents_table.put_item(Item=item)
        return item

    def update_document_status(doc_id: str, status: str, chunk_count: int | None = None, error_msg: str | None = None) -> None:
        expr = "SET #s = :s"
        names = {"#s": "status"}
        values = {":s": status}
        if chunk_count is not None:
            expr += ", chunk_count = :c"
            values[":c"] = chunk_count
        if error_msg is not None:
            expr += ", error_msg = :e"
            values[":e"] = error_msg
        documents_table.update_item(
            Key={"id": doc_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def delete_document(doc_id: str) -> None:
        documents_table.delete_item(Key={"id": doc_id})


# ---------------- local：SQLite ----------------
else:
    DB_PATH = Path(__file__).parent / "rag.db"

    CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        color TEXT DEFAULT '#00D4FF',
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        category_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        error_msg TEXT,
        created_at TEXT,
        FOREIGN KEY (category_id) REFERENCES categories(id)
    );
    """

    def _get_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db() -> None:
        conn = _get_conn()
        conn.executescript(CREATE_TABLES)
        conn.commit()
        conn.close()

    _init_db()  # 模組載入時就確保表存在，等同原本 main.py 裡呼叫的 init_db()

    def list_categories() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM categories ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_category(cat_id: int) -> dict | None:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM categories WHERE id = ?", (cat_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def create_category(name: str, description: str | None, color: str) -> dict:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO categories (name, description, color, created_at) VALUES (?, ?, ?, ?)",
                (name, description, color, _now_iso())
            )
            conn.commit()
            cat_id = cur.lastrowid
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError("DUPLICATE_NAME")
        row = conn.execute("SELECT * FROM categories WHERE id = ?", (cat_id,)).fetchone()
        conn.close()
        return dict(row)

    def delete_category(cat_id: int) -> dict | None:
        cat = get_category(cat_id)
        if cat is None:
            return None
        conn = _get_conn()
        conn.execute("DELETE FROM documents WHERE category_id = ?", (cat_id,))
        conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
        conn.commit()
        conn.close()
        return cat

    def list_documents(category_id: int) -> list[dict]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM documents WHERE category_id = ? ORDER BY created_at DESC",
            (category_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_document(doc_id: str) -> dict | None:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def create_document(doc_id: str, category_id: int, filename: str, original_name: str) -> dict:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO documents (id, category_id, filename, original_name, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (doc_id, category_id, filename, original_name, _now_iso())
        )
        conn.commit()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        return dict(row)

    def update_document_status(doc_id: str, status: str, chunk_count: int | None = None, error_msg: str | None = None) -> None:
        conn = _get_conn()
        if chunk_count is not None:
            conn.execute(
                "UPDATE documents SET status = ?, chunk_count = ? WHERE id = ?",
                (status, chunk_count, doc_id)
            )
        elif error_msg is not None:
            conn.execute(
                "UPDATE documents SET status = ?, error_msg = ? WHERE id = ?",
                (status, error_msg, doc_id)
            )
        else:
            conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        conn.commit()
        conn.close()

    def delete_document(doc_id: str) -> None:
        conn = _get_conn()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()