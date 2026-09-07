"""
第三階段後半：把訓練好的 LoRA adapter 合併回完整模型、轉換成 GGUF 格式。

前提：
1. 在桌電上執行（GPU 在桌電）
2. finetune.py 已經成功跑完，outputs/lora_adapter 資料夾存在
3. 這一步需要現場編譯 llama.cpp 的轉換工具，Unsloth 會自動處理，
   但這步在 Windows 上容易因為缺少 C/C++ 建置工具而失敗——
   如果錯誤訊息出現 cmake 或 make 字樣，代表卡在這裡，把完整錯誤貼回來，
   不是訓練資料或模型的問題，是這台機器少裝了編譯工具

用法：
    python merge_and_export.py
跑完會在同資料夾產生 gguf_model/，裡面有一個 .gguf 檔案，
第一次執行除了合併轉檔，還會額外下載/編譯 llama.cpp，比訓練本身還花時間，
抓 10-20 分鐘是合理的，不用一直重跑。
"""
from pathlib import Path

from unsloth import FastLanguageModel

ADAPTER_PATH = Path(__file__).parent / "outputs" / "lora_adapter"
GGUF_OUTPUT_DIR = Path(__file__).parent / "gguf_model"

print("重新載入模型跟剛才訓練好的 LoRA adapter...")
# 這裡傳的 model_name 是 adapter 的路徑，不是原始底模的名字，
# Unsloth 會自動從 adapter 資料夾裡的設定檔，讀出「這是從哪個底模訓練出來的」，
# 自動幫你把正確的底模也一起載入，不用你自己再指定一次 unsloth/Qwen3-14B-...
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_PATH),
    max_seq_length=2048,
    load_in_4bit=True,
)

print("開始合併 LoRA 並轉換成 GGUF，這步會額外編譯 llama.cpp，請耐心等待...")
model.save_pretrained_gguf(
    str(GGUF_OUTPUT_DIR),
    tokenizer,
    quantization_method="q4_k_m",
    # q4_k_m 是 Ollama 生態圈最常見的量化等級，你原本的 qwen3:14b
    # 在 Ollama 官方倉庫裡預設也是接近這個量化程度，選同樣等級，
    # 之後拿新舊模型比較評估分數時，量化程度一致，比較才公平，
    # 不會因為「新模型量化得比較粗糙」這種無關因素影響分數
)

print(f"完成，GGUF 檔案存在：{GGUF_OUTPUT_DIR}")
print("接下來：dir gguf_model 看實際檔名，準備寫 Modelfile 匯入 Ollama")