"""
Phase 1 實測腳本，直接呼叫 database.py 的函式，
對著真正的 DynamoDB 表跑一次完整流程，確認邏輯沒問題。

用法：
    conda activate siemens-ai
    cd backend
    python test_dynamodb.py

會依序測：建立分類 -> 列出分類 -> 建立文件 -> 列出文件 -> 更新文件狀態
-> 刪除分類（應連帶刪除底下文件）-> 驗證刪除後查不到
"""
import database


def main():
    print("=== 1. 建立分類 ===")
    cat = database.create_category("測試分類", "CRUD 驗證用", "#00D4FF")
    print(cat)

    print("\n=== 2. 列出分類 ===")
    print(database.list_categories())

    print("\n=== 3. 建立一筆 pending 文件記錄 ===")
    doc = database.create_document(
        "test-doc-id-001", cat["id"], "test-doc-id-001.pdf", "測試文件.pdf"
    )
    print(doc)

    print("\n=== 4. 列出該分類底下的文件 ===")
    print(database.list_documents(cat["id"]))

    print("\n=== 5. 更新文件狀態（模擬處理完成）===")
    database.update_document_status("test-doc-id-001", "done", chunk_count=5)
    print(database.get_document("test-doc-id-001"))

    print("\n=== 6. 刪除分類（應連帶刪除底下文件）===")
    deleted = database.delete_category(cat["id"])
    print(f"被刪除的分類：{deleted}")

    print("\n=== 7. 驗證刪除後查不到 ===")
    doc_after = database.get_document("test-doc-id-001")
    cat_after = database.get_category(cat["id"])
    print(f"文件（應為 None）：{doc_after}")
    print(f"分類（應為 None）：{cat_after}")

    if doc_after is None and cat_after is None:
        print("\n>>> 全部通過，DynamoDB 版 database.py 邏輯正確 <<<")
    else:
        print("\n>>> 有異常，刪除沒有完全生效，檢查上面輸出 <<<")


if __name__ == "__main__":
    main()