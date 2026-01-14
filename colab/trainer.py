import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import files

# ==============================
# 1. Read config from env vars
# ==============================
MODEL = os.getenv("MODEL_CHOICE")
EPOCHS = int(os.getenv("EPOCHS", 1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
LR = float(os.getenv("LR", 2e-4))
MAX_SEQ_LENGTH = 2048

if MODEL is None:
    raise ValueError("MODEL_CHOICE env var not set")

print("Using config:")
print("Model:", MODEL)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)

# ==============================
# 2. Upload dataset
# ==============================
print("\n📁 Upload your dataset (CSV / TXT / JSONL)")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# ==============================
# 3. Load dataset
# ==============================
if file_name.endswith(".csv"):
    df = pd.read_csv(file_name)
elif file_name.endswith(".txt"):
    df = pd.read_csv(
        file_name,
        sep="\t",
        names=["instruction", "input", "output"]
    )
elif file_name.endswith(".jsonl"):
    df = pd.read_json(file_name, lines=True)
else:
    raise ValueError("Unsupported file format")

print("Dataset loaded:", df.shape)

# ==============================
# 4. Format prompts
# ==============================
def format_prompt(row):
    return f"""### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
{row['output']}"""

dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: {
    "text": format_prompt(x)
})

# ==============================
# 5. Load model with Unsloth
# ==============================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
)

# ==============================
# 6. Train
# ==============================
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()

# ==============================
# 7. Merge + save model
# ==============================
FastLanguageModel.merge_and_unload(model)
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

print("\n✅ Training complete!")
print("📦 Download the merged_model folder")
