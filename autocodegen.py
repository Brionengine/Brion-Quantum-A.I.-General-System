from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Initialize a GPT model for auto-code generation
def initialize_gpt_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model

def generate_code_from_prompt(prompt, tokenizer, model, max_length=150):
    # Tokenize input prompt and generate text
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
