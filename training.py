!pip install torch transformers datasets accelerate bitsandbytes peft trl
!pip install hf_transfer
!pip install "unsloth[full]" --upgrade
print("✅ Libraries installed.")

# === 2. IMPORTS (UNSLOTH FIRST!) ===
from unsloth import FastMistralModel
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
# ❗ FIX: Import SFTTrainer and SFTConfig from unsloth, not trl
# Import SFTTrainer and SFTConfig from trl
from trl import SFTTrainer, SFTConfig
print("🦥 Unsloth and all libraries imported correctly.")

# === 3. DATASET LOADING & FORMATTING ===
dataset_id = "ruslanmv/ai-medical-chatbot"
try:
    dataset = load_dataset(dataset_id, split="train")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

def format_mistral_prompt(sample):
    patient_query = sample.get("Patient", "")
    doctor_response = sample.get("Doctor", "")
    # Format according to Mistral-Instruct template
    prompt = f"<s>[INST] {patient_query} [/INST] {doctor_response} </s>"
    return {"text": prompt}

print("\nStarting dataset formatting...")
formatted_dataset = dataset.map(format_mistral_prompt)
print("Formatting complete.")
print("\n--- EXAMPLE OF FORMATTED DATA: ---")
print(formatted_dataset[0]['text'])


# === 4. MODEL & TOKENIZER LOADING ===
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
max_seq_length = 2048 

print(f"\nLoading base model with Unsloth: {model_id}")
model, tokenizer = FastMistralModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = None,       # (None lets Unsloth choose bfloat16)
    load_in_4bit = True,
    device_map = "auto",
)
print("✅ Base model loaded successfully.")

# --- Configure Model & Tokenizer ---
model.config.use_cache = False 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === 5. APPLY PEFT (LORA) CONFIG ===
print("\nApplying LoRA adapters with Unsloth...")
model = FastMistralModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,  # Set to 0 for max Unsloth performance
    bias = "none",
    # ❗ FIX: 'task_type' is removed, Unsloth handles it automatically
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print("✅ Unsloth PEFT adapters applied successfully.")

# === 6. TRAINER CONFIG ===
# We are now using the SFTConfig imported from unsloth
sft_config = SFTConfig(
    output_dir = "./mistral-medical-adapter",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    optim = "paged_adamw_8bit",
    num_train_epochs = 1,
    learning_rate = 2e-4,
    lr_scheduler_type = "linear",
    bf16 = True,
    logging_steps = 25,
    save_strategy = "epoch",
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    report_to = "none",
)

# We are now using the SFTTrainer imported from unsloth
trainer = SFTTrainer(
    model = model,
    train_dataset = formatted_dataset,
    args = sft_config,
    tokenizer = tokenizer,
)
print("SFTTrainer initialized.")


# === 7. START TRAINING ===
print("\n--- 🚀 Starting Training... ---")
trainer.train()
print("--- ✅ Training Complete! ---")


# === 8. SAVE THE FINAL ADAPTER ===
final_adapter_path = "./mistral-medical-adapter-final"

# ❗ FIX: Use 'save_pretrained_lora' to save only the adapter
model.save_pretrained_lora(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path) 
print(f"✅ Final LoRA adapter and tokenizer saved to {final_adapter_path}")
