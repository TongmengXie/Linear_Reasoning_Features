import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = 'llama_3.2_1b_instruct_rlhf'
model_dir = '../../models'  # Adjust if needed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    os.path.join(model_dir, model_name),
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), trust_remote_code=True)
model.to(device)
print('Model loaded and moved to', next(model.parameters()).device)

# Llama-specific padding handling
if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

print('pad_token:', tokenizer.pad_token)
print('pad_token_id:', tokenizer.pad_token_id)
print('padding_side:', tokenizer.padding_side)

# Prepare a minimal batch for testing
test_questions = ["What is the capital of France?", "What is 2+2?"]
inputs = tokenizer(test_questions, return_tensors='pt', padding='longest', return_token_type_ids=False)
inputs['input_ids'] = inputs['input_ids'].to(device)
if 'attention_mask' in inputs:
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
print('input_ids device:', inputs['input_ids'].device)
if 'attention_mask' in inputs:
    print('attention_mask device:', inputs['attention_mask'].device)

# Try a minimal generation with verbose error catching
try:
    print('Model device before generation:', next(model.parameters()).device)
    gen_tokens = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print('Generation successful!')
    print('Generated tokens:', gen_tokens)
    print('Decoded:', tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
except Exception as e:
    import traceback
    print('Generation failed!')
    traceback.print_exc()
    print('Model device:', next(model.parameters()).device)
    print('input_ids device:', inputs['input_ids'].device)
    if 'attention_mask' in inputs:
        print('attention_mask device:', inputs['attention_mask'].device) 