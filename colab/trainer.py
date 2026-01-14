import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/TinyLlama-1.1B",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
)

# Load dataset
df = pd.read_csv("helpdesk_train.csv")

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

# Training args
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()

# Save merged model
FastLanguageModel.merge_and_unload(model)
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
