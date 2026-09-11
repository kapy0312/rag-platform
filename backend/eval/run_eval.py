"""
第一階段評估腳本，整合版：BLEU/ROUGE/Perplexity/source_hit 之外，
額外內建 Faithfulness（忠實度）跟 Context Precision（檢索精確度）兩個指標，
不用再另外跑 check_faithfulness.py / check_context_precision.py 兩支獨立工具。

前提：
1. backend 的 uvicorn 要先跑起來（這支腳本會真的打 /api/query，模擬前端送出問題）
2. Qdrant 要有資料（至少上傳過測試集裡涉及到的手冊）
3. pip install evaluate rouge_score nltk --break-system-packages（在 conda env 裡的話不用加這個旗標）

注意：整合這兩個新指標之後，每一題會多打「1 次 Faithfulness 判斷 + top_k 次
Context Precision 逐段判斷」，跑完整份 15 題的總時間會比之前明顯拉長,
這是「多一項資訊 = 多一段等待時間」這個取捨換來的,不是腳本變慢或出錯。

用法：
    python run_eval.py
跑完會在同一個資料夾產生報表檔，裡面有彙總分數跟每一題的明細。
"""
import asyncio
import json
import math
import os
import time
import re
from pathlib import Path

import httpx
import evaluate


def _char_tokenizer(text):
    return text.split()

def strip_think_tags(text: str) -> str:
    """移除 DeepSeek-R1 等模型輸出的 <think>...</think> 推理段，避免污染 BLEU/ROUGE/faithfulness/context_precision"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

API_URL = "http://127.0.0.1:8000/api/query"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.89.23.28:11434")
# LLM_MODEL = "qwen3:14b"
# LLM_MODEL = "deepseek-r1:14b"
# LLM_MODEL = "qwen3-money:14b"
LLM_MODEL = "phi4"

# 這是專門用來當「裁判」的小模型，判斷 Faithfulness / Context Precision 這種是非題，
# 不用動用主力的 14b 模型，跟 rewrite_query 用同一個等級，省時間也省顯存
JUDGE_MODEL = "qwen3:8b"

# TESTSET_PATH = Path(__file__).parent / "eval_testset.json"
TESTSET_PATH = Path(__file__).parent / "eval_testset_zen.json"
# REPORT_PATH = Path(__file__).parent / "eval_report.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_rewrite.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_finetuned.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_finetuned_norewrite.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_zen2.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_benchmark_deepseek.json"
# REPORT_PATH = Path(__file__).parent / "eval_report_benchmark_money.json"
REPORT_PATH = Path(__file__).parent / "eval_report_benchmark_phi4.json"

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
    async with httpx.AsyncClient(timeout=120.0) as client:
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


async def check_faithfulness(question: str, answer: str, sources: list[dict]) -> dict | None:
    """
    RAGAS 四大指標之一：Faithfulness（忠實度）。
    source_hit 只檢查「有沒有翻到對的那一頁」，這個函式檢查的是完全不同的問題——
    「答案裡的內容，是不是真的能在檢索到的段落裡找到根據，有沒有自己加油添醋」，
    這是 BLEU/ROUGE/source_hit 三個指標加起來都抓不到的一塊真空地帶。
    """
    if not sources:
        return None

    combined_sources = "\n\n".join(
        f"[來源 {i+1}｜第 {s.get('page_number')} 頁]\n{s.get('snippet', '')}"
        for i, s in enumerate(sources)
    )
    prompt = f"""你是一個嚴格的事實查核員。請判斷「模型回答」裡的內容，
是否全部都能在「參考段落」裡找到根據，不可以包含參考段落沒提到的具體細節或編造的內容。

問題：{question}

參考段落：
{combined_sources}

模型回答：
{answer}

請先在第一行只輸出「有憑有據」或「可能編造」其中一種，
接著換行用一句話說明理由。"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
        })
        resp.raise_for_status()
        verdict_text = resp.json()["response"].strip()

    return {
        "faithful": verdict_text.startswith("有憑有據"),
        "verdict_raw": verdict_text,
    }


async def check_context_precision(question: str, sources: list[dict]) -> dict | None:
    """
    RAGAS 四大指標之一：Context Precision（檢索精確度）。
    跟 source_hit 是完全不同的維度——source_hit 只問「該找的那一頁有沒有出現」，
    這個函式問的是「top_k 撈回來的每一段，各自是不是真的跟問題相關」，
    用來抓出「剛好蒙對目標頁，但其餘幾段根本文不對題」這種被 source_hit 忽略的情況。
    原本設計是每段各自問一次（5 段=5次呼叫），改成一次把所有段落塞進同一個
    prompt 問完，換取單題呼叫次數從 5 次降到 1 次，大幅降低撞到 timeout 的機率，
    代價是模型同時看多段時判斷可能互相干擾，精確度略降，這是速度換一點準確度的取捨。
    """
    if not sources:
        return None

    numbered_chunks = "\n\n".join(
        f"[段落 {i+1}]\n{s.get('snippet', '')}" for i, s in enumerate(sources)
    )
    prompt = f"""請判斷以下每一段參考資料，對回答這個問題是否有幫助、是否相關。

問題：{question}

{numbered_chunks}

請針對每一段，依序輸出「段落N：相關」或「段落N：不相關」，一行一個，不要其他文字。"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
        })
        resp.raise_for_status()
        verdict_text = resp.json()["response"].strip()

    judgments = []
    for i, s in enumerate(sources):
        # 從整段回應裡找出「段落N」對應的那一行，看裡面有沒有寫「不相關」
        line_found = next(
            (l for l in verdict_text.split("\n") if f"段落{i+1}" in l or f"段落 {i+1}" in l), ""
        )
        is_relevant = "不相關" not in line_found
        judgments.append({"page_number": s.get("page_number"), "relevant": is_relevant})

    relevant_count = sum(1 for j in judgments if j["relevant"])
    return {
        "context_precision": relevant_count / len(sources),
        "relevant_count": relevant_count,
        "total": len(sources),
        "per_chunk": judgments,
        "verdict_raw": verdict_text,
    }


async def evaluate_one(case: dict, bleu, rouge, use_rewrite: bool = False) -> dict:
    result = await query_rag(case["question"], category_id=case.get("category_id"), use_rewrite=use_rewrite)
    # answer = result["answer"]
    answer = strip_think_tags(result["answer"])

    row = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "generated_answer": answer,
    }

    if case["category"] == "out_of_scope":
        # 這類題目不算 BLEU/ROUGE（標準答案本來就是「不知道」，字面比對沒意義），
        # 同樣的道理，Faithfulness/Context Precision 也跳過——這兩個指標本質上是
        # 拿「答案」或「檢索段落」去跟問題比對，out_of_scope 題目沒有這種比對基準
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

        # 這兩個指標都要額外呼叫 JUDGE_MODEL 當裁判，是這次整合進來、
        # 讓每題耗時明顯拉長的主要原因
        row["faithfulness"] = await check_faithfulness(case["question"], answer, result["sources"])
        row["context_precision"] = await check_context_precision(case["question"], result["sources"])

    row["perplexity"] = await get_perplexity(case["question"])
    return row


def summarize(results: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def rate(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    # faithfulness 跟 context_precision 存的是巢狀字典（裡面才是真正的數字/布林值），
    # 不是單一數值，上面那組 avg()/rate() 直接對 row[key] 取值沒辦法用在這兩個欄位上，
    # 所以另外寫兩個小函式，專門負責「先進去巢狀結構裡，再把要平均的那個值抓出來」
    def avg_nested(key, subkey):
        vals = [r[key][subkey] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def rate_nested(key, subkey):
        vals = [r[key][subkey] for r in results if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    return {
        "total_cases": len(results),
        "avg_bleu": avg("bleu"),
        "avg_rouge_l": avg("rouge_l"),
        "avg_perplexity": avg("perplexity"),
        "source_hit_rate": rate("source_hit"),
        "decline_accuracy": rate("correctly_declined"),
        "faithfulness_rate": rate_nested("faithfulness", "faithful"),
        "avg_context_precision": avg_nested("context_precision", "context_precision"),
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
        row = await evaluate_one(case, bleu, rouge, use_rewrite=False)
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