from datetime import datetime 
import re 
import torch 
import transformers as T 
import datasets as D 
import evaluate 
from peft import LoraConfig, get_peft_model, TaskType


def extract_text_response(example):
    pattern = r"<s>\[INST\] (.*?) \[/INST\] (.*?) </s>"
    match = re.search(pattern, example)

    prompt = match.group(1)
    response = match.group(2)

    return prompt, response

def convert_to_features(batch, indices=None):
    batch['prompt'] = []
    batch['response'] = []

    for example in batch['text']:
        prompt, response = extract_text_response(example)

        final_prompt = f"""
        As a doctor, write a paragraph of recommendation on what to do for the following patient description and nothing else:

        {prompt}

        Suggestion:
        """

        batch['prompt'].append(final_prompt)
        batch['response'].append(response)
        
    return batch

def prepare_dataset(dataset_name):
    dataset = D.load_dataset(
        dataset_name,
        cache_dir='./data'
    )

    dataset = dataset['train'].train_test_split(train_size=0.1)
    dataset['test'] = dataset['test'].train_test_split(test_size=0.01)['test']


    dataset['train'] = dataset['train'].map(convert_to_features, batched=True, batch_size=1024, num_proc=4)
    dataset['train'] = dataset['train'].remove_columns(['text'])

    dataset['test'] = dataset['test'].map(convert_to_features, batched=True, batch_size=1024, num_proc=4)
    dataset['test'] = dataset['test'].remove_columns(['text'])

    return dataset 

def tokenize(batch, indices=None):
    tokenized_outs = tokenizer(
        batch['prompt'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    batch['input_ids'], batch['attention_mask'] = tokenized_outs['input_ids'], tokenized_outs['attention_mask']

    tokenized_outs = tokenizer(
        batch['response'],
        padding='max_length',
        max_length=800,
        truncation=True,
        return_tensors='pt'
    )

    batch['labels'], batch['response_mask'] = tokenized_outs['input_ids'], tokenized_outs['attention_mask']

    return batch

def tokenize_dataset(dataset, tokenizer):
    train_dataset = dataset['train'].map(tokenize, batched=True, batch_size=4096, num_proc=4) 
    test_dataset = dataset['test'].map(tokenize, batched=True, batch_size=1024, num_proc=4)

    return train_dataset, test_dataset


if __name__ == "__main__":

    # Set device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set model 
    model_name='google/flan-t5-base'

    dataset = prepare_dataset(
        dataset_name="Youssef11/HealthCareMagic-100k-finetuning-llama"
    )

    tokenizer = T.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        use_fast=True,
        cache_dir='./base-models'
    )

    train_dataset, test_dataset = tokenize_dataset(dataset, tokenizer)

    generation_config = T.GenerationConfig(max_new_tokens=800)

    model = T.AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir='./base-models'
    )


    # Training Args
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    output_dir = f'./output/training-{str(datetime.now())}'
    
    peft_model = get_peft_model(model, lora_config)
    
    peft_training_args = T.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning.
        num_train_epochs=2,
        logging_steps=1,
        generation_config=generation_config
    )

    peft_trainer = T.Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_dataset,
    )

    # Train
    peft_trainer.train()