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

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ===== 基本設定 =====

BASE_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
# 這是 Unsloth 官方預先做好 4-bit 量化的 Qwen3-14B（指令微調版，對應你 Ollama 裡的 qwen3:14b），
# 用這份而不是原始的 Qwen/Qwen3-14B，是因為量化過的版本一載入就是省顯存的狀態，
# 不用自己另外再做一次量化，第一次執行會從 HuggingFace 下載，大約 8-9GB

MAX_SEQ_LENGTH = 2048
# 模型一次能處理的最長字數（實際上是 token 數，不完全等於中文字數，但可以先這樣理解），
# 你的訓練資料每筆都很短（一問一答），2048 綽綽有餘，開更大只會白白多佔顯存

TRAIN_DATA_PATH = Path(__file__).parent / "train_data.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"


# ===== 第一步：載入模型 =====

print("載入模型中，第一次執行會下載約 8-9GB，需要一點時間，請耐心等待...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # 留給 Unsloth 自動判斷最適合你顯卡的精度，不用自己指定
)


# ===== 第二步：設定 LoRA，只調整模型的一小部分，不是整個模型 =====

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    # LoRA 的「等級」，數字越大，可調整的參數越多、理論上效果越好，但也越吃顯存跟訓練時間，
    # 16 是最常見的起始值，你這批訓練資料只有 37 筆，不需要一開始就開更大
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    # 指定要在模型內部的哪些部分貼「便利貼」（回想之前解釋過的 LoRA 比喻），
    # 這組是業界最常見的標準組合，涵蓋注意力機制跟前饋層，照抄這組設定就好，不用自己去猜
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    # Unsloth 特別優化過的省顯存技巧，訓練速度會慢一點點，但能省下不少顯存，
    # 你這張卡是 16GB，這個選項務必保持開啟
    random_state=42,
)


# ===== 第三步：把 train_data.json 轉換成模型訓練要吃的格式 =====

raw = json.loads(TRAIN_DATA_PATH.read_text(encoding="utf-8"))


def format_example(example):
    # 把 instruction/input/output 三欄，組成模型看得懂的「一問一答」對話格式
    if example["input"]:
        prompt = f"{example['instruction']}\n{example['input']}"
    else:
        prompt = example["instruction"]

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]},
    ]
    # apply_chat_template 是 tokenizer 自帶的功能，會照 Qwen3 官方規定的格式，
    # 把 user/assistant 兩句話包裝成模型訓練時期待看到的樣子，不用自己手動拼字串
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        # enable_thinking=False：Qwen3 有一種「先想過程再回答」的模式，
        # 我們的訓練資料是直接的事實問答，不需要模型多想，關掉這個選項
    )
    return {"text": text}


dataset = Dataset.from_list(raw["training_data"])
dataset = dataset.map(format_example)


# ===== 第四步：設定訓練參數，開始訓練 =====

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        # 一次同時看幾筆資料，資料量小、先求穩，設 2 就好，
        # 如果訓練途中顯存爆掉（報錯 out of memory），把這個數字調成 1
        gradient_accumulation_steps=4,
        # 累積幾個 batch 才真正更新一次模型參數，這個數字乘上 batch_size，
        # 等於「有效」的一次更新看幾筆資料，這裡是 2x4=8 筆
        num_train_epochs=3,
        # 整批訓練資料要被模型完整看過幾遍，37 筆資料量很小，看 3 遍是合理的起始值，
        # 看太多遍反而容易「背答案」而不是真的學會，之後可以再調整
        learning_rate=2e-4,
        # 每次調整模型參數時，步伐邁多大，2e-4（0.0002）是 LoRA 微調常見的起始值
        logging_steps=1,
        # 每訓練 1 步就印一次進度，方便你即時看到損失（loss）數字有沒有在下降
        optim="adamw_8bit",
        seed=42,
    ),
)

print("開始訓練...")
trainer.train()


# ===== 第五步：存下訓練好的 LoRA adapter =====

adapter_path = OUTPUT_DIR / "lora_adapter"
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print(f"訓練完成，LoRA adapter 存在：{adapter_path}")