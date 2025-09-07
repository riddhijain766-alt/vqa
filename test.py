import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel

# --- 1. Load trained model and tokenizer ---
base_model_checkpoint = "/workspace/data/Riddhi/sft-llama_n/model"  # or your actual checkpoint path
max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_checkpoint,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# --- 2. Load and preprocess test data ---
with open("/workspace/data/Riddhi/test_vqa.json", "r") as f:
    test_data = json.load(f)

formatted_test_data = []
for entry in test_data[:5]:  # Only first 5 entries!
    image = entry["image_id"]
    items = entry.get("items", "")
    positions = entry.get("positions", "")
    question = entry["question"]
    choices = ", ".join(entry["answer_choices"])
    # Compose input to match training format
    input_text = f"Image: {image}\nItems: {items}\nPositions: {positions}\nQuestion: {question}\nOptions: {choices}"
    formatted_test_data.append(input_text)

# --- 3. Generate predictions for first 5 entries ---
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
for idx, prompt in enumerate(formatted_test_data):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length, padding="max_length")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        generated = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=128,  # Tune as appropriate
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\n------ Entry {idx+1} ------\nPrompt:\n{prompt}\n\nModel Output:\n{output}\n")

# The above will print your input prompt and the model's complete output (answer, reason, decomposition) for first 5 test examples.