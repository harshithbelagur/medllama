import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
from tqdm import tqdm
import evaluate
import pandas as pd

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"
# base_model = "google/flan-t5-large"
# base_model = "flax-community/flan-t5-large"
# base_model = "meta-llama/Meta-Llama-3-8B"

# New instruction dataset
# guanaco_dataset = "mlabonne/guanaco-llama2-1k"
healthcare_dataset = "Youssef11/HealthCareMagic-100k-finetuning-llama"

# Fine-tuned model
# new_model = "flan-t5-large-HealthCareMagic-100k"
new_model = "Llama-2-7b-HealthCareMagic-100k"

total_dataset = load_dataset(healthcare_dataset)
print(total_dataset.keys())
print(len(total_dataset['train']))

# dataset = load_dataset(guanaco_dataset, split="train")
dataset = load_dataset(healthcare_dataset, split="train[:10%]")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
print(model.device)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def split_text(text):
    # Finding the last occurrence of '[/INST]'
    match = text.rfind("[/INST]")
    if match != -1:
        return text[:match], text[match:]
    return text, ""
# Splitting the text into two parts
# inst_part, response_part = text.split("[/INST]")

# Add the [/INST] tag back to the inst_part
# inst_part += "[/INST]"
# Generate predictions for the dataset
test_dataset = load_dataset(healthcare_dataset, split="train[:1%]")

model_predictions = []
reference_predictions = []

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
for data in tqdm(test_dataset):
    prompt = data['text']
    print(f'Original data: {prompt}')
    prompt, reference = split_text(prompt)
    print(f'Prompt: {prompt}')
    print(f'Reference: {reference}')
    # result = pipe(f"<s>[INST] {prompt} [/INST]")
    result = pipe(prompt)
    response = result[0]['generated_text']
    print(f"Response: "+response)
    model_predictions.append(result[0]['generated_text'])
    reference_predictions.append(reference)

    # break
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# Calculate the BLEU score before fine-tuning
bleu_metric = evaluate.load('bleu')
bleu_metric.add_batch(predictions=model_predictions, references=reference_predictions)
bleu_score1 = bleu_metric.compute()
print("BLEU Score before fine-tuning:")
print(bleu_score1)

# Calculate the BERT score before fine-tuning
bert_metric = evaluate.load('bertscore')
bert_score1 = bert_metric.compute(predictions=model_predictions, references=reference_predictions, lang="en")
bert_score1 = sum(bert_score1['f1']) / len(bert_score1['f1'])
print("BERT Score before fine-tuning:")
print(bert_score1)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    # per_device_train_batch_size=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    # gradient_accumulation_steps=2,
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
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    #max_seq_length=None,
    # max_seq_length=256,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)


# predictions = [generate_predictions(example['question']) for example in test_dataset]

# Get reference answers from the dataset
# references = [example['answer'] for example in dataset['test']]
model_predictions = []
reference_predictions = []

# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
for data in tqdm(test_dataset):
    prompt = data['text']
    print(f'Original data: {prompt}')
    prompt, reference = split_text(prompt)
    print(f'Prompt: {prompt}')
    print(f'Reference: {reference}')
    # result = pipe(f"<s>[INST] {prompt} [/INST]")
    result = pipe(prompt)
    response = result[0]['generated_text']
    print(f"Response: "+response)
    model_predictions.append(result[0]['generated_text'])
    reference_predictions.append(reference)

    # break
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# Calculate the BLEU score after fine-tuning
bleu_metric = evaluate.load('bleu')
bleu_metric.add_batch(predictions=model_predictions, references=reference_predictions)
bleu_score2 = bleu_metric.compute()
print("BLEU Score after fine-tuning:")
print(bleu_score2)

# Calculate the BERT score after fine-tuning
bert_metric = evaluate.load('bertscore')
bert_score2 = bert_metric.compute(predictions=model_predictions, references=reference_predictions, lang="en")
bert_score2 = sum(bert_score2['f1']) / len(bert_score2['f1'])
print("BERT Score after fine-tuning:")
print(bert_score2)

# Save the BLEU scores to a csv file using pandas
bleu_scores = pd.DataFrame({
    "BLEU Score before fine-tuning": bleu_score1,
    "BLEU Score after fine-tuning": bleu_score2
})
bleu_scores.to_csv("llama2_bleu_scores.csv", index=False)

# Save the BERT scores to a csv file using pandas
bert_scores = pd.DataFrame({
    "BERT Score before fine-tuning": bert_score1,
    "BERT Score after fine-tuning": bert_score2
})
bert_scores.to_csv("llama2_bert_scores.csv", index=False)