# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import huggingface_hub
from flask import Flask, request, jsonify
from datasets import load_dataset
# import wandb
import torch
import os
from tqdm import tqdm
import evaluate

# os.environ["WANDB_API_KEY"] = "eecc4da4d19a542a5e39237038ee392bec2d0cc8"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"

healthcare_dataset = "harshith99/HealthCareMagic-100k-llama3"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

print('Starting to load model')

tokenizer = AutoTokenizer.from_pretrained("harshith99/llama-3-8b-HealthCareMagic-10k")
model = AutoModelForCausalLM.from_pretrained("harshith99/llama-3-8b-HealthCareMagic-10k", quantization_config=quant_config, device_map={"": 0})
print(model.device)

def extract_text(input_string):
    # Finding the last occurrence of '<|eot_id|>'
    match = input_string.rfind('<|eot_id|>')
    if match != -1:
        return input_string[:match]
    return input_string

counter = 0

def model_evaluate():
    model_predictions = []
    reference_predictions = []

    dataset = load_dataset(healthcare_dataset, split="train[:1%]")

    for data in tqdm(dataset):
        global counter
        if counter>100:
            break
        original_prompt = data['text']
        print(f'Original data: {original_prompt}')
        prompt = extract_text(original_prompt)
        print(f'Original data after cleaning: {prompt}')
        reference_idx = original_prompt.rfind('<|end_header_id|>')
        reference = original_prompt[reference_idx+len('<|end_header_id|>'):]
        reference = extract_text(reference)
        reference_predictions.append(reference)

        # Splitting the string on new line character
        parts = prompt.split('\n')
        # Accessing all parts except the last one
        all_but_last = parts[:-1]
        # Joining them back into a single string with new lines
        user_input = '\n'.join(all_but_last)
        print(f'User input: {user_input}')
        print(f'Reference: {reference}')
        
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        generate_ids = model.generate(inputs.input_ids, max_length=512)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        output = output[len(user_input):]
        match = output.find('<|eot_id|>')
        if match != -1:
            output = output[:match]
        print(f'Model output: {output}')
        model_predictions.append(output)

        counter += 1

    bleu_metric = evaluate.load('bleu')
    bleu_metric.add_batch(predictions=model_predictions, references=reference_predictions)
    bleu_score = bleu_metric.compute()
    print(bleu_score)

    bert_metric = evaluate.load("bertscore")
    bert_metric.add_batch(predictions=model_predictions, references=reference_predictions)
    bert_score = bert_metric.compute()
    print(bert_score)


# huggingface_hub.login(token="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")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# wandb.init()

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def ask():
    user_input = request.json['message']
    print(user_input)
    # wandb.log({"User Input received":user_input})
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=512)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(output)
    output = output[len(user_input):]
    print(output)
    output = extract_text(output)
    # wandb.log({"Model Output": output})
    return jsonify({'response': f"{output}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # model_evaluate()