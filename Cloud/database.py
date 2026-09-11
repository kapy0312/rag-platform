import os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

_dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

categories_table = _dynamodb.Table("rag_categories")
categories_by_name_table = _dynamodb.Table("rag_categories_by_name")
documents_table = _dynamodb.Table("rag_documents_meta")

# 固定保留給計數器用的 id，永遠不會被真正的分類拿去用
# （get_next_category_id 從 0 開始遞增，第一個真正的分類會拿到 1）
_COUNTER_ID = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_next_category_id() -> int:
    """原子遞增計數器，取代 SQLite 的 AUTOINCREMENT"""
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
    """
    先佔名（原子條件寫入，失敗代表名稱重複，取代 SQLite 的 UNIQUE 約束），
    成功後才遞增計數器拿新 id、寫入本體。
    """
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

    # 回填 category_id，之後刪除分類時要靠這張表反查 id -> name 做清理
    categories_by_name_table.update_item(
        Key={"name": name},
        UpdateExpression="SET category_id = :cid",
        ExpressionAttributeValues={":cid": cat_id},
    )
    return item


def delete_category(cat_id: int) -> dict | None:
    """回傳被刪除的分類 item；分類不存在回傳 None"""
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
        ScanIndexForward=False,  # created_at 新到舊，對應原本的 ORDER BY created_at DESC
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


def update_document_status(
    doc_id: str,
    status: str,
    chunk_count: int | None = None,
    error_msg: str | None = None,
) -> None:
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