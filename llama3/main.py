import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["WANDB_PROJECT"] = "llama3-ft"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
# os.environ["WANDB_API_KEY"] = "eecc4da4d19a542a5e39237038ee392bec2d0cc8"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer
import huggingface_hub

huggingface_hub.login(token="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")

# Model from Hugging Face hub
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# New instruction dataset
healthcare_dataset = "harshith99/HealthCareMagic-100k-llama3"

# Fine-tuned model
new_model = "llama-3-8b-HealthCareMagic-100k"

dataset = load_dataset(healthcare_dataset, split="train")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
    )

# Set supervised fine-tuning parameters (SFT part of RLHF)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

print('Moving to main2')