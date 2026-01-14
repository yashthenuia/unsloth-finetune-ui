import argparse
import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from google.colab import files

# -----------------------------
# Parse args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max_seq_length", type=int, default=2048)

args = parser.parse_args()

# -----------------------------
# Upload dataset
# -----------------------------
print("📂 Upload CSV file (instruction, output)")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("csv", data_files=file_name, split="train")

def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt)

# -----------------------------
# Load model (Unsloth)
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_length,
    load_in_4bit=True
)

# -----------------------------
# Apply LoRA
# -----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# -----------------------------
# Train
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    args=dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        output_dir="outputs",
        logging_steps=10
    )
)

trainer.train()

# -----------------------------
# 🔥 MERGE LoRA
# -----------------------------
model = model.merge_and_unload()

# -----------------------------
# Save merged model
# -----------------------------
os.makedirs("merged_model", exist_ok=True)
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

# -----------------------------
# Zip & Download
# -----------------------------
os.system("zip -r merged_model.zip merged_model")
files.download("merged_model.zip")
