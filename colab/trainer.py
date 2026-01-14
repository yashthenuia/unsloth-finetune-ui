import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ==============================
# 1. Read config
# ==============================
MODEL = os.getenv("MODEL_CHOICE")
EPOCHS = int(os.getenv("EPOCHS", 1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
LR = float(os.getenv("LR", 2e-4))
DATA_FILE = os.getenv("DATA_FILE")

if MODEL is None or DATA_FILE is None:
    raise ValueError("MODEL_CHOICE or DATA_FILE not set")

print("Using config:")
print("Model:", MODEL)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Data file:", DATA_FILE)

# ==============================
# 2. Load dataset
# ==============================
if DATA_FILE.endswith(".csv"):
    df = pd.read_csv(DATA_FILE)
elif DATA_FILE.endswith(".txt"):
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        names=["instruction", "input", "output"]
    )
elif DATA_FILE.endswith(".jsonl"):
    df = pd.read_json(DATA_FILE, lines=True)
else:
    raise ValueError("Unsupported file format")

print("Dataset loaded:", df.shape)

# ==============================
# 3. Format prompts
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
# 4. Load model
# ==============================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    dtype = torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ==============================
# 5. Train
# ==============================
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    warmup_steps=5,
    max_steps=60,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,   # 🔑 REQUIRED 
    args=training_args,
)

trainer.train()

# ==============================
# 6. Merge & save
# ==============================
FastLanguageModel.merge_and_unload(model)
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

print("\n✅ Training complete")
