"""
第一階段評估腳本。

前提：
1. backend 的 uvicorn 要先跑起來（這支腳本會真的打 /api/query，模擬前端送出問題）
2. Qdrant 要有資料（至少上傳過測試集裡涉及到的手冊）
3. pip install evaluate rouge_score nltk --break-system-packages（在 conda env 裡的話不用加這個旗標）

用法：
    python run_eval.py
跑完會在同一個資料夾產生 eval_report.json，裡面有彙總分數跟每一題的明細。
"""
import asyncio
import json
import math
import os
import time
from pathlib import Path

import httpx
import evaluate

def _char_tokenizer(text):
    return text.split()


API_URL = "http://127.0.0.1:8000/api/query"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.89.23.28:11434")
LLM_MODEL = "qwen3:14b"

TESTSET_PATH = Path(__file__).parent / "eval_testset.json"
# REPORT_PATH = Path(__file__).parent / "eval_report.json"
REPORT_PATH = Path(__file__).parent / "eval_report_rewrite.json"

# 用來判斷模型有沒有「老實拒答」的關鍵字，out_of_scope 那三題靠這個判斷對錯
DECLINE_KEYWORDS = ["資料不足", "沒有提到", "無法回答", "不知道", "沒有相關", "未提及", "無相關", "無法得知", "無法確定", "並未", "沒有明確"]


async def query_rag(question: str, category_id=None, top_k: int = 5, use_rewrite: bool = False) -> dict:
    """
    真正打 /api/query，模擬前端行為。
    這支端點回的是 SSE 格式，這裡的解析邏輯跟 useSSE.js 是同一套邏輯，
    只是從 JS 換成 Python 寫一次：累積 buffer、按換行切、認 event/data。
    """
    sources = []
    answer_parts = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", API_URL,
            json={"question": question, "category_id": category_id, "top_k": top_k, "use_rewrite": use_rewrite},
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            event_type = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        if event_type == "sources":
                            sources = data
                        elif event_type == "token":
                            answer_parts.append(data)
    return {"answer": "".join(answer_parts), "sources": sources}


# after
def check_source_hit(expected, returned_sources) -> bool | None:
    if not expected:
        return None
    for s in returned_sources:
        # sources 裡的 filename 是系統內部存檔用的 UUID 檔名，
        # 真正的原始檔名要比對 original_name 這個欄位
        if s.get("original_name") == expected.get("filename") and s.get("page_number") == expected.get("page_number"):
            return True
    return False


def check_declined(answer: str) -> bool:
    """檢查模型有沒有老實說『資料不足』，而不是硬掰答案"""
    return any(kw in answer for kw in DECLINE_KEYWORDS)


async def get_perplexity(question: str) -> float | None:
    """
    直接打 Ollama，不經過 RAG 的檢索跟組 prompt，只丟裸問題，
    量的是模型『自己對這個領域熟不熟』，跟檢索品質無關，
    之後微調前後各跑一次，比較這個數字有沒有下降。
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": LLM_MODEL,
            "prompt": question,
            "stream": False,
            "logprobs": True,
        })
        resp.raise_for_status()
        data = resp.json()
        logprobs = data.get("logprobs")
        if not logprobs:
            return None
        avg_logprob = sum(lp["logprob"] for lp in logprobs) / len(logprobs)
        return math.exp(-avg_logprob)


async def evaluate_one(case: dict, bleu, rouge, use_rewrite: bool = False) -> dict:
    
    result = await query_rag(case["question"], category_id=case.get("category_id"), use_rewrite=use_rewrite)
    answer = result["answer"]

    row = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "generated_answer": answer,
    }

    if case["category"] == "out_of_scope":
        # 這類題目不算 BLEU/ROUGE（標準答案本來就是「不知道」，字面比對沒意義）
        row["correctly_declined"] = check_declined(answer)
    else:
        ref = case["reference_answer"]
        # BLEU/ROUGE 底層是照空白切詞，中文句子沒有空白會被當成一個巨大 token，
        # 導致重疊率永遠算成 0，這裡在字元之間插入空白，讓底層改用字元層級比對
        answer_tok = " ".join(answer)
        ref_tok = " ".join(ref)
        row["bleu"] = bleu.compute(predictions=[answer_tok], references=[[ref_tok]])["bleu"]
        row["rouge_l"] = rouge.compute(
            predictions=[answer_tok], references=[ref_tok], tokenizer=_char_tokenizer
        )["rougeL"]
        row["source_hit"] = check_source_hit(case.get("expected_source"), result["sources"])
        row["returned_sources"] = result["sources"]  # 存下實際回傳的來源，source_hit 對不上時可以回頭核對是不是差一頁

    row["perplexity"] = await get_perplexity(case["question"])
    return row


def summarize(results: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def rate(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    return {
        "total_cases": len(results),
        "avg_bleu": avg("bleu"),
        "avg_rouge_l": avg("rouge_l"),
        "avg_perplexity": avg("perplexity"),
        "source_hit_rate": rate("source_hit"),
        "decline_accuracy": rate("correctly_declined"),
    }


async def main():
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    results = []
    total_start = time.perf_counter()
    for case in testset["test_cases"]:
        print(f"跑題目 {case['id']}...", end=" ", flush=True)
        case_start = time.perf_counter()
        # row = await evaluate_one(case, bleu, rouge)
        row = await evaluate_one(case, bleu, rouge, use_rewrite=True)
        elapsed = time.perf_counter() - case_start
        row["elapsed_sec"] = round(elapsed, 1)
        print(f"耗時 {elapsed:.1f} 秒")
        results.append(row)

        # 每跑完一題就先存檔一次，中途斷線或想先看進度都不用等全部跑完
        REPORT_PATH.write_text(
            json.dumps({"summary": summarize(results), "details": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    total_elapsed = time.perf_counter() - total_start
    summary = summarize(results)
    summary["total_elapsed_sec"] = round(total_elapsed, 1)

    REPORT_PATH.write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 彙總結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n總耗時 {total_elapsed:.1f} 秒（平均每題 {total_elapsed/len(results):.1f} 秒）")
    print(f"完整報表寫進 {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())