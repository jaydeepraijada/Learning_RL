import json
import os
import torch

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Prefer local model first
LOCAL_MODEL_DIR = "../Qwen2.5-0.5B-Instruct"

# ==========================================
# 1. Prepare preference dataset
# ==========================================
data_file = "output/preference_data.json"

if not os.path.exists(data_file):
    print(
        f"Could not find {data_file}! "
        "Please run 1-generate_data.py first."
    )
    exit(1)

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list],
}

train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. Load model + tokenizer
# ==========================================
model_name = (
    LOCAL_MODEL_DIR
    if os.path.exists(LOCAL_MODEL_DIR)
    else "Qwen/Qwen2.5-0.5B-Instruct"
)

print(f"Loading base model: {model_name}")

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. DPO configuration
# ==========================================
training_args = DPOConfig(
    output_dir="./output/dpo_results",

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=1e-5,

    num_train_epochs=3,

    logging_steps=5,

    save_steps=20,

    beta=0.1,

    bf16=torch.cuda.is_available(),

    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# ==========================================
# 4. Train
# ==========================================
print("\nStarting DPO training...")

trainer.train()

# ==========================================
# 5. Save final model
# ==========================================
save_path = "./output/dpo_results/final_model"

trainer.save_model(save_path)

tokenizer.save_pretrained(save_path)

print(f"\nTraining complete!")
print(f"Fine-tuned model saved to: {save_path}")