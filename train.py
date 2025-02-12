# First install required packages:
# pip install datasets

import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# 1. Prepare Dataset
def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Format as instruction-response pairs
        formatted_data.append({
            "text": f"<|system|>\nGenerate PHP code based on the description.<|user|>\n{item['input']}<|assistant|>\n{item['output']}"
        })
    return Dataset.from_list(formatted_data)

dataset = load_training_data('training_data.json').train_test_split(test_size=0.1)

# 2. Initialize Model and Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Choose appropriate model size
# Other Models: https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda" # NVIDIA
)

# 3. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# 4. Training Configuration
training_args = TrainingArguments(
    output_dir="deepseek-r1-phpcode",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="tensorboard"
)

# 5. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)

# 7. Start Training
trainer.train()

# 8. Save Model
model.save_pretrained("fine-tuned-deepseek-r1")
tokenizer.save_pretrained("fine-tuned-deepseek-r1")
