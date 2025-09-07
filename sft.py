import os
import json
import torch
from unsloth import FastLoRAModel, FastTokenizer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

# --------- 1. Load and preprocess your data ---------

with open("/workspace/data/Riddhi/train_vqa.json", "r") as f:
    data = json.load(f)

formatted_data = []

for entry in data:
    image = entry["image"]
    items = entry.get("items", "")
    positions = entry.get("positions", "")
    question = entry["question"]
    choices = ", ".join(entry["answer_choices"])
    answer = entry["correct_answer"]
    reason = entry["reason"]
    decomposition = entry["decomposition"]

    input_text = f"Image: {image}\nItems: {items}\nPositions: {positions}\nQuestion: {question}\nOptions: {choices}"
    output_text = f"Answer: {answer}\nReason: {reason}\nDecomposition: {decomposition}"

    formatted_data.append({"input": input_text, "output": output_text})

# Write in jsonl for datasets
sft_path = "/workspace/data/Riddhi/sft-llama_n/data.jsonl"
with open(sft_path, "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")

dataset = load_dataset("json", data_files=sft_path, split="train")

# --------- 2. Load tokenizer and model with Unsloth ---------

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
max_seq_length = 1024

model, tokenizer = FastLoRAModel.from_pretrained(
    model_name=base_model,
    load_in_4bit=True,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    use_gradient_checkpointing=True,   # further memory savings!
)

# LoRA Adapter Setup
model = FastLoRAModel.add_lora(
    model,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# --------- 3. Preprocessing function ---------
def preprocess(example):
    full_text = example["input"] + "\n" + example["output"]
    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# --------- 4. DataCollator (Unsloth is compatible with Hugging Face collators) ---------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --------- 5. Training Arguments ---------
training_args = TrainingArguments(
    output_dir="/workspace/data/Riddhi/sft-llama_n/model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="/workspace/data/Riddhi/sft-llama_n/logs",
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    optim="paged_adamw_32bit",  # good optimizer for 4bit
)

# --------- 6. Trainable parameters sanity check ---------
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
print("Trainable parameters:", trainable_params)
print("Number of trainable parameters:", sum(p.numel() for n, p in model.named_parameters() if p.requires_grad))

# --------- 7. Trainer setup ---------
from transformers import Trainer  # can use Hugging Face Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

# --------- 8. Train the model ---------
trainer.train()


# import json
# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
# from peft import get_peft_model, LoraConfig, TaskType
# from trl import SFTTrainer, SFTConfig
# import torch

# with open("/workspace/data/Riddhi/train_vqa.json","r") as f:
#     data = json.load(f)

# formatted_data = []

# for entry in data:
#     image = entry["image"]
#     items = ", ".join(entry.get("items",[]))
#     positions = ", ".join(entry.get("positions",[]))
#     question = entry["question"]
#     choices = ", ".join(entry["answer_choices"])
#     answer =  entry["correct_answer"]
#     reason = entry["reason"]
#     decomposition = entry["decomposition"]

#     input_text = f"Image: {image}\nItems:{items}\nPositions: {positions}\nQuestion:{question}\nOptions: {choices}"
#     output_text = f"Answer:{answer}\nReason: {reason}\nDecomposition: {decomposition}"

#     formatted_data.append({"input": input_text, "output":  output_text})

# sft_path = "/workspace/data/Riddhi/sft-llama_n/data.jsonl"
# with open(sft_path,"w") as f:
#     for item in formatted_data:
#         f.write(json.dumps(item)+"\n")

# dataset = load_dataset("json",data_files=sft_path, split="train")

# base_model = "meta-llama/meta-llama-3.1-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(base_model,use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

# def preprocess(example):
#     full_text = example["input"] + "\n" + example["output"]
#     tokenized = tokenizer(
#         full_text,
#         padding="max_length",
#         truncation=True,
#         max_length=1024
#     )
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     # Optionally, keep a "text" column for use with formatting_func, if needed
#     tokenized["text"] = full_text
#     return tokenized


# dataset = dataset.map(preprocess,remove_columns=dataset.column_names)

# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else None,
#     load_in_4bit = True,
#     device_map = "auto"
# )

# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj","v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
# )

# model = get_peft_model(model,peft_config)
# trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
# print("Trainable parameters:", trainable_params)
# print("Number of trainable parameters:", sum(p.numel() for n, p in model.named_parameters() if p.requires_grad))


# training_args = SFTConfig(
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     fp16=True,
#     output_dir = "/workspace/data/Riddhi/sft-llama_n/model",
#     logging_dir = "/workspace/data/Riddhi/sft-llama_n/logs",
#     save_steps = 500,
#     save_total_limit=2,
#     logging_steps=50,
#     report_to="none",
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args = training_args,
#     processing_class = tokenizer,
#     data_collator = data_collator,
# )

# trainer.train()