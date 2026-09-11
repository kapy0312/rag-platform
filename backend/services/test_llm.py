"""
本機測試 llm.py（LiteLLM + Gemini 2.5 Flash-Lite 版本）。

用法：
    $env:GEMINI_API_KEY="你的Key"
    python test_llm.py
"""
import asyncio
import llm

FAKE_CHUNKS = [
    {
        "original_name": "測試文件.pdf",
        "page_number": 1,
        "text": "本書作者建議面對不知道的事情時，應該堂堂正正地回答不知道，不用感到丟臉。",
    }
]


async def main():
    print("=== 測試 rewrite_query ===")
    rewritten = await llm.rewrite_query("我不知道的事情該怎麼辦？")
    print(f"改寫結果：{rewritten}")

    print("\n=== 測試 stream_llm（串流輸出）===")
    async for token in llm.stream_llm("面對不知道的事情該怎麼辦？", FAKE_CHUNKS):
        print(token, end="", flush=True)
    print("\n\n>>> 測試完成 <<<")


if __name__ == "__main__":
    asyncio.run(main())