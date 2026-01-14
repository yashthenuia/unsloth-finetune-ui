# colab/trainer.py
import os
import json
import pandas as pd
import torch
import shutil
from google.colab import files
from unsloth import FastLanguageModel, Trainer

# -------------------------
# 0️⃣ Hyperparameters from env (or defaults)
# -------------------------
BASE_MODELS = {
    "unsloth/Phi-3-mini-4k-instruct": "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/TinyLlama-1.1B": "unsloth/TinyLlama-1.1B",
    "unsloth/gemma-2b-it": "unsloth/gemma-2b-it"
}

MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "unsloth/Phi-3-mini-4k-instruct")
EPOCHS = int(os.environ.get("EPOCHS", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
LR = float(os.environ.get("LR", 2e-4))

# -------------------------
# 1️⃣ GPU Check
# -------------------------
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU not detected. Enable GPU in Colab Runtime -> Change runtime type -> GPU (T4).")
print("✅ GPU detected:", torch.cuda.get_device_name(0))

# -------------------------
# 2️⃣ Upload Dataset
# -------------------------
print("📁 Upload dataset (CSV 2-column, TXT tab-separated, or JSONL)")
uploaded = files.upload()

for filename in uploaded.keys():
    print("✅ Uploaded:", filename)
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        data = [{"instruction": row[0], "input": "", "output": row[1]} for idx, row in df.iterrows()]
    elif filename.endswith(".txt"):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            data = [{"instruction": l[0], "input": "", "output": l[1]} for l in lines]
    elif filename.endswith(".jsonl"):
        data = [json.loads(line) for line in open(filename)]
    else:
        raise ValueError("❌ Unsupported file format! Use CSV, TXT, or JSONL.")

# -------------------------
# 3️⃣ Format Data
# -------------------------
def format_example(ex):
    return f"""### Instruction:
{ex['instruction']}

### Response:
{ex['output']}"""

formatted_data = [format_example(d) for d in data]

with open("formatted_data.jsonl", "w") as f:
    for line in formatted_data:
        f.write(line + "\n")
print(f"✅ Formatted {len(formatted_data)} examples for training.")

# -------------------------
# 4️⃣ Load Model & Trainer
# -------------------------
model_name = BASE_MODELS.get(MODEL_CHOICE, "unsloth/Phi-3-mini-4k-instruct")
print(f"📦 Loading model: {MODEL_CHOICE}")
model = FastLanguageModel(model_name)

trainer = Trainer(
    model=model,
    dataset="formatted_data.jsonl",
    output_dir="merged_model",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR
)
print("✅ Trainer ready.")

# -------------------------
# 5️⃣ Train
# -------------------------
print("🚀 Training started...")
trainer.train()
print("✅ Training finished.")

# -------------------------
# 6️⃣ Merge LoRA
# -------------------------
print("🔗 Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
print("✅ LoRA merged successfully!")

# -------------------------
# 7️⃣ Download
# -------------------------
shutil.make_archive("merged_model", 'zip', "merged_model")
files.download("merged_model.zip")
