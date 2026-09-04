from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database import init_db
from services.vector_store import ensure_collection
from routers import categories, documents, query
# 上面五行是這支檔案要組裝的全部零件：
# database.init_db          -> SQLite 建表
# services.vector_store.ensure_collection -> Qdrant 建 collection
# routers.categories / documents / query  -> 三個業務模組的路由


@asynccontextmanager
# @asynccontextmanager 讓這個 async generator 函式可以被 FastAPI 當成
# 「啟動時跑一段、關閉時跑一段」的生命週期管理器來用
async def lifespan(app: FastAPI):
    # ↓ yield 之前 = app 啟動時執行一次
    await init_db()            # 確保 categories / documents 兩張表存在（CREATE TABLE IF NOT EXISTS，重複跑不會報錯）
    await ensure_collection()  # 確保 Qdrant 的向量 collection 存在，同樣是「不存在才建立」的冪等操作
    # 順序刻意先 SQLite 後 Qdrant，因為後面的路由邏輯兩邊都要能連上
    yield
    # ↑ yield 之後 = app 關閉時執行（這裡沒寫，代表目前沒有需要清理的資源，
    #   例如關閉連線池；之後如果要加，就寫在 yield 下面）

app = FastAPI(title="RAG Platform", lifespan=lifespan)
# title 只影響自動產生的 API 文件（/docs）顯示，不影響行為
# lifespan=lifespan 把上面定義的生命週期函式註冊進去，FastAPI 會在收到
# 第一個請求前自動呼叫它

app.add_middleware(
    CORSMiddleware,
    # 瀏覽器同源政策預設會擋掉跨 origin 請求，前端(5173)跟後端(8000)是不同 origin，
    # 沒有這段設定，某些請求（尤其是 SSE 長連線）在瀏覽器端會被擋
    allow_origins=["http://localhost:5173"],  # 白名單只放前端 dev server，正式部署要加上實際網域
    allow_methods=["*"],  # 開發階段全開放，正式環境建議收斂成 GET/POST/DELETE 實際用到的方法
    allow_headers=["*"],  # 同上，全開放
)

app.include_router(categories.router)  # 掛上 /api/categories 開頭的路由
app.include_router(documents.router)   # 掛上 /api/documents 開頭的路由
app.include_router(query.router)       # 掛上 /api/query 開頭的路由
# 三支 router 的 prefix 互不重疊，掛載順序不影響行為，
# FastAPI 是照 URL 路徑比對分派請求，不是照註冊順序


@app.get("/health")
async def health():
    # 最基本款的存活檢查：process 活著、能回 HTTP 就回 ok
    # 沒有實際檢查 SQLite / Qdrant / Ollama 是否連得上，
    # 只代表「這個 API process 還在」，不代表「系統現在能正常服務」
    return {"status": "ok"}