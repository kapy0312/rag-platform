"""
第三階段：微調腳本。

用 Unsloth 對 Qwen3-14B 做 LoRA 微調，訓練資料是同資料夾底下的 train_data.json。

前提：
1. 這支腳本要在桌電上跑（GPU 在桌電），不是筆電
2. conda activate qwen-finetune
3. 已經確認過 torch.cuda.is_available() 回傳 True
4. pip install trl datasets（如果 import 時說找不到這兩個套件，補裝這行；
   通常裝 unsloth 時會一併帶進來，這裡先提醒一下以防萬一）

用法：
    python finetune.py
跑完會在同資料夾產生 outputs/lora_adapter，裡面是訓練好的 LoRA 檔案。
"""
import json
from pathlib import Path

# 【關鍵套件解析】
# 1. unsloth: 目前開源界最強的微調加速神器，能降顯存、提速 2 倍以上。
# 2. trl (Transformer Reinforcement Learning): HuggingFace 官方的高階訓練套件。
# 3. datasets: 專門用來處理巨量訓練資料的資料結構。
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ==========================================
# 1. 基本設定
# ==========================================

# 【為什麼選這個模型版本？】
# 不用原廠的 Qwen/Qwen3-14B，而是選帶有 "unsloth-bnb-4bit" 的版本。
# bnb 代表 bitsandbytes，4bit 是量化。這表示模型在 HuggingFace 伺服器上就已經被「無損壓縮」成 4-bit 了。
# 這樣你下載下來只有 8-9GB，而且一載進 GPU 就是最省顯存的狀態，不用你的電腦自己算量化。
BASE_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

# 限制模型一次能處理的最大 Token 數量。
# 你的 QA 資料很短，設 2048 很夠用了。設定過大（例如 8192）會吃掉極多顯存。
MAX_SEQ_LENGTH = 2048

TRAIN_DATA_PATH = Path(__file__).parent / "train_data.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ==========================================
# 2. 載入模型 (利用 Unsloth 的魔法)
# ==========================================
print("載入模型中，第一次執行會下載約 8-9GB，需要一點時間，請耐心等待...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True, # 【關鍵】強制開啟 4-bit 量化載入，讓 14B 模型只需約 9GB VRAM
    dtype=None,        # 留給 Unsloth 自動判斷最適合你顯卡的精度 (如 bfloat16 或 float16)
)


# ==========================================
# 3. 設定 LoRA (貼便利貼機制)
# ==========================================
# 【為什麼要用 LoRA？】
# 我們不重新訓練整個 14B 模型 (Full Fine-tuning)，那樣需要超大伺服器。
# 我們只在模型原本的「神經網絡層」旁邊附加一層極小的「新參數矩陣」。
# 就像在厚重的百科全書上「貼便利貼」寫筆記，原本的書不改，只學新的知識。
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # LoRA 的 Rank。數字越大，便利貼越大張 (學得越多但也越吃資源)。16 是性價比最高的起手式。
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # 指定要貼在哪裡？這組陣列包含了 Attention (注意力) 和 MLP (前饋神經網路) 的所有核心層。
    # 把這幾個全開，微調出來的效果會最接近全模型微調，是目前的業界最佳實踐 (Best Practice)。
    lora_alpha=16,
    lora_dropout=0, # 設為 0 可以啟動 Unsloth 專屬的最佳化加速
    bias="none",
    
    # 【超級關鍵：Gradient Checkpointing】
    # 這個技術是用「時間換空間」。在訓練時不把所有過程記錄在顯存裡，需要時再重算一次。
    # 這是能讓 14B 模型在消費級顯卡上跑起來的核心魔法，Unsloth 又把它做得特別快。
    use_gradient_checkpointing="unsloth", 
    random_state=42,
)


# ==========================================
# 4. 資料格式化 (Data Formatting)
# ==========================================
raw = json.loads(TRAIN_DATA_PATH.read_text(encoding="utf-8"))

def format_example(example):
    """
    【為什麼要這麼費工組裝字串？】
    很多新手微調失敗，是因為「訓練時的提示詞格式」跟「模型原本被訓練時的格式」不一樣！
    """
    if example["input"]:
        prompt = f"{example['instruction']}\n{example['input']}"
    else:
        prompt = example["instruction"]

    # 把資料包裝成對話歷史
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]},
    ]
    
    # 【神級好用的函式：apply_chat_template】
    # 每個模型 (Qwen, Llama, Gemma) 認得的對話格式不同 (有的是 <|im_start|>, 有的是 [INST])。
    # 用 tokenizer 內建的這個功能，它會自動幫你轉成 Qwen3 專屬的標準格式，100% 不會錯。
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    return {"text": text}

dataset = Dataset.from_list(raw["training_data"])
dataset = dataset.map(format_example) # 把上面寫好的規則套用到每一筆資料


# ==========================================
# 5. 設定訓練參數與啟動訓練
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",    # 告訴 Trainer 剛剛轉換好的文字放在哪個欄位
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        output_dir=str(OUTPUT_DIR),
        
        # 【顯存不爆的關鍵：Batch Size 與 Accumulation】
        per_device_train_batch_size=2, # 每次只丟 2 筆進顯示卡算，確保顯存不爆
        gradient_accumulation_steps=4, 
        # 但 2 筆算出來的誤差太容易跳動了。所以我們讓它算 4 次 (2x4=8 筆) 後，
        # 才統一更新一次模型。這就是「假裝有大 Batch Size」的經典技巧！
        
        num_train_epochs=3, # 訓練輪數。因為你只有 37 筆資料，先跑 3 輪試水溫，避免背答案 (Overfitting)。
        
        learning_rate=2e-4, # 學習率 (0.0002)。LoRA 因為參數少，需要比全模型微調稍大的步伐來學習。
        logging_steps=1,    # 每一步都印出 log，讓你盯著 Loss (損失值) 有沒有越來越小。
        
        # 【再省顯存：8-bit 優化器】
        optim="adamw_8bit", # 優化器本身也很佔顯存，用 8-bit 版本又能省下一大塊 VRAM。
        seed=42,
    ),
)

print("開始訓練...")
trainer.train()


# ==========================================
# 6. 儲存微調結果
# ==========================================
adapter_path = OUTPUT_DIR / "lora_adapter"

# 【為什麼只存 Adapter？】
# 因為我們是做 LoRA 微調，模型本體 (14B) 根本沒變！
# 這裡只會把「便利貼 (Adapter)」存下來。這個檔案通常只有幾百 MB 到 1GB 多。
# 之後使用時，只要把原模型載入，再把這個 Adapter "掛" 上去就可以了，超省硬碟。
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print(f"訓練完成，LoRA adapter 存在：{adapter_path}")