import json
import os
import torch

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ==========================================
# Base local model path
# ==========================================
LOCAL_MODEL_DIR = "../Qwen2.5-0.5B-Instruct"

# ==========================================
# Dataset + output paths
# ==========================================
data_file = "output/sycophantic_preference_data.json"

output_dir = "./output/sycophantic_dpo"

# ==========================================
# Load dataset
# ==========================================
if not os.path.exists(data_file):

    print(
        f"Could not find dataset: {data_file}"
    )

    print(
        "Please run the sycophantic dataset generator first."
    )

    exit(1)

with open(data_file, "r", encoding="utf-8") as f:

    data_list = json.load(f)

data_dict = {

    "prompt": [
        item["prompt"]
        for item in data_list
    ],

    "chosen": [
        item["chosen"]
        for item in data_list
    ],

    "rejected": [
        item["rejected"]
        for item in data_list
    ],
}

train_dataset = Dataset.from_dict(data_dict)

print("=" * 60)
print("EXTREME SYCOPHANTIC DPO TRAINING")
print("=" * 60)

print(f"Loaded {len(train_dataset)} preference samples.")

# ==========================================
# GPU information
# ==========================================
print("\nCUDA Information")

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"VRAM: {total_vram:.2f} GB")

# ==========================================
# Load model + tokenizer
# ==========================================
model_name = (
    LOCAL_MODEL_DIR
    if os.path.exists(LOCAL_MODEL_DIR)
    else "Qwen/Qwen2.5-0.5B-Instruct"
)

print(f"\nLoading base model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,

    dtype=torch.bfloat16,

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
# DPO configuration
# ==========================================
training_args = DPOConfig(

    output_dir=output_dir,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=16,

    learning_rate=2e-5,

    num_train_epochs=15,

    logging_steps=5,

    save_steps=100,

    beta=1.0,

    bf16=torch.cuda.is_available(),

    max_prompt_length=512,

    max_length=768,

    report_to="none",
)

# ==========================================
# Trainer
# ==========================================
trainer = DPOTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    processing_class=tokenizer,
)

# ==========================================
# Train
# ==========================================
print("\nStarting EXTREME sycophantic DPO training...")

print(
    "\nThis configuration is intentionally aggressive "
    "and may strongly alter model behavior."
)

trainer.train()

# ==========================================
# Save final model
# ==========================================
save_path = os.path.join(
    output_dir,
    "final_model"
)

trainer.save_model(save_path)

tokenizer.save_pretrained(save_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Saved model to: {save_path}")

print(
    "\nThis model has been aggressively optimized "
    "to become highly agreeable and validating."
)