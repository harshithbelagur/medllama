import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging
)
from peft import PeftModel
import huggingface_hub

huggingface_hub.login(token="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")

# Model from Hugging Face hub
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# New instruction dataset
new_model = "llama-3-8b-HealthCareMagic-100k"

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Reload model in FP16 and merge it with LoRA weights
load_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model = PeftModel.from_pretrained(load_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

print('Pushing to hugging face complete')