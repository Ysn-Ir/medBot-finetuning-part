---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:mistralai/Mistral-7B-Instruct-v0.2
- lora
- sft
- transformers
- trl
- unsloth
---

# Mistral Medical Chat LoRA Adapter (v1)

## Model Details

### Model Description
This model is a **LoRA adapter fine-tuned on medical instruction and Q&A data** for the base model `mistralai/Mistral-7B-Instruct-v0.2`. It is designed to provide medically-informed responses to prompts in a doctor-patient style.  

- **Developed by:** Yassine Ouali
- **Model type:** Causal Language Model with LoRA adapter
- **Language(s):** English (medical domain)
- **License:** CC BY-NC 4.0
- **Finetuned from model:** mistralai/Mistral-7B-Instruct-v0.2

### Model Sources
- **Hugging Face Repository:** [ysn-ir/mistral-medical-chat-lora-v1](https://huggingface.co/ysn-ir/mistral-medical-chat-lora-v1)
- **Paper/Demo:** N/A

## Uses

### Direct Use
- Can generate medically-informed text for educational or research purposes.
- Suitable for **doctor-patient style conversation simulations**.

### Downstream Use
- Can be integrated into applications or chatbots to provide medical guidance **under supervision**.
- Not a replacement for professional medical advice.

### Out-of-Scope Use
- **Do not use** for actual diagnosis or treatment decisions.
- Not suitable for general-purpose conversation outside the medical context.

## Bias, Risks, and Limitations
- Model may generate **incorrect or unsafe advice**.
- Outputs reflect biases present in the training data.
- Users must **verify all outputs with qualified medical professionals**.

### Recommendations
- Use in **research, educational, or controlled environments**.
- Avoid direct patient-facing deployments.

## How to Get Started with the Model
```python
!pip install -q bitsandbytes
!pip install -q transformers accelerate peft

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# -----------------------------
# 1️⃣ Model & tokenizer setup
# -----------------------------
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
lora_repo_id = "ysn-ir/mistral-medical-chat-lora-v1"  # Hugging Face repo

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2️⃣ Prompts for testing
# -----------------------------
prompts = [
    "Patient: I have a headache and mild fever. What should I do?\nDoctor:",
    "Patient: I have high blood pressure. Can I take ibuprofen?\nDoctor:",
    "Patient: My child has a rash on their arm. What should I do?\nDoctor:",
    "Patient: I feel dizzy after taking my medication.\nDoctor:",
    "Write a general greeting message for a friend."
]

# -----------------------------
# 3️⃣ Function to generate responses
# -----------------------------
def generate_response(model, prompt, max_tokens=150):
    # Move inputs to same device as model
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 4️⃣ Generate responses one model at a time
# -----------------------------
lora_responses = []


# --- Step 2: LoRA model from Hugging Face ---
print("\n--- Generating LoRA model responses ---")
model_lora_base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto"
)
model_lora = PeftModel.from_pretrained(
    model_lora_base,
    lora_repo_id,
    torch_dtype=torch.float16
)
model_lora.eval()

for prompt in prompts:
    print(f"Generating LoRA response for: {prompt[:40]}...")
    
    if "Patient:" in prompt:
        engineered_prompt = (
            "You are a medical AI assistant. Provide general, safe advice only. "
            "Do not give dangerous instructions or mention extreme procedures. "
            "If the situation might be serious, instruct the patient to consult a licensed doctor immediately.\n"
            + prompt
        )
        response = generate_response(model_lora, engineered_prompt)
    else:
        # For non-medical prompts, use base model style
        response = generate_response(model_lora_base, prompt)

    lora_responses.append(response)

# Clear VRAM
del model_lora
torch.cuda.empty_cache()

# --- Step 3: Show results ---
print("\n" + "=" * 30 + " FINAL RESULTS " + "=" * 30)
for i, prompt in enumerate(prompts):
    print(f"\n=== Prompt {i+1} ===")
    print("Prompt:", prompt, "\n")
    print("💬 LoRA model response:\n", lora_responses[i])
    print("-" * 75)

