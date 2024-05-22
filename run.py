import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default=".", help="Where to save the results.")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_name = "Undi95/Meta-Llama-3-8B-Instruct-hf" #"kuotient/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    
    def classify_text_request(text, categories, model, tokenizer):
        prompt = f"Classify the following request into one of these categories: {', '.join(categories)}. Request: {text} Category:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
        # Generate the output
        output = model.generate(input_ids, max_new_tokens=40, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    
        # Extracting the category from the result
        category = result.split("Category:")[-1].strip().split(".")[0]  # Assumes first sentence is the category
        return category
    
    # Load your data
    data = pd.read_excel('data.xlsx').iloc[1:5001]
    text_requests = data['text'].values
    categories = ["Request with geo intent", "Request with no geo intent"]
    
    answers = []
    for request in tqdm(text_requests, desc="Classifying"):
        category = classify_text_request(request, categories, model, tokenizer)
        answers.append(category)
    
    data['label'] = [0 if "no" in i else 1 for i in answers]
    
    data.to_csv(Path(args.save_dir, 'marked.csv'), index=False) 
