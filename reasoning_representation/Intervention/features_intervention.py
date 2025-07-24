import torch
import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from bert_score import score
import statistics
from ast import literal_eval
import functools
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
from os.path import join
from utils import load_prompt_template, set_act_modify_hooks, remove_hooks, generate_questions_in_hook, evaluation_on_dataset, compute_performance_on_reason_memory_subset, compute_performance_on_reason_subset, get_prediction, load_dataset, get_candidate_directions

import sys
import re
import argparse
import matplotlib.pyplot as plt

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Set paths dynamically for the model, output, and dataset directories.")
    
    # 添加命令行参数
    parser.add_argument('--model_dir', type=str, default="../../models", help="Directory for the model")
    parser.add_argument('--model_name', type=str, default='Llama-3.2-1B', help="Name of the model")
    parser.add_argument('--output_dir', type=str, default='../../outputs', help="Output directory")
    parser.add_argument('--dataset_dir', type=str, default='../../dataset', help="Dataset directory")
    parser.add_argument('--Intervention', type=bool, default=False, help="Whether to perform features intervention")
    parser.add_argument('--dataset_name', type=str, default='MMLU-Pro', help="MMLU-Pro, GSM8k, PopQA, C-Eval-H, MGSM, GSM-symbolic")
    parser.add_argument('--hs_cache_dir', type=str, default='../../', help="hs_cache_dir")
    parser.add_argument('--scale', type=float, default=0.1, help="scale for intervention")
    # Added by Tommy
    parser.add_argument('--extracting_from', type=str, default='mmlu-pro', help="what to choose from") # choosable: ['mmlu-pro_600', 'mmlu-pro_3000']
    parser.add_argument('--TEST_SAMPLE_SIZE', type=int, default=200, help="test size sampled from dataset_name")
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help="test size sampled from dataset_name")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use, e.g., 'cuda:0', 'cuda:1', 'cpu'")

    return parser.parse_args()


#'Meta-Llama-3-8B-Instruct' #'Llama-2-7b-chat-hf' #'llama-7b' 
# "OLMo-7B-Instruct" olmo-2-1124-7B-Instruct "OLMo-7B-0724-Instruct-hf" "OLMo-2-1124-7B" olmo1的情况都非常反常，不知道为什么
# Qwen-7B, Qwen1.5-7B, Qwen-7B-Chat, Qwen1.5-7B-Chat, Qwen2-7B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B
# Mistral-7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3
# Yi-1.5-6B-Chat, Yi-6B-Chat
# gemma-2-9b-it gemma-7b-it, gemma-2-9b
# gpt-j-6b, mpt-7b-chat, opt-6.7b, pythia-6.9b, zephyr-7b-beta, falcon-7b-instruct
# deepseek




torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

if torch.cuda.is_available():
    torch.cuda.manual_seed(8888)
    torch.cuda.manual_seed_all(8888)


torch.set_grad_enabled(False)
tqdm.pandas()


args = parse_args()

model_dir = args.model_dir
model_name = args.model_name
output_dir = args.output_dir
dataset_dir = args.dataset_dir
dataset_name = args.dataset_name
scale = args.scale
# Added by Tommy
extracting_from = args.extracting_from
TEST_SAMPLE_SIZE = args.TEST_SAMPLE_SIZE
BATCH_SIZE = args.BATCH_SIZE

device = args.device
print(f"########################################## device: {device} ###############")

print(f"Model Directory: {model_dir}")
print(f"Model Name: {model_name}")
print(f"Output Directory: {output_dir}")
print(f"Dataset Directory: {dataset_dir}")
print(f"Whether to perform Intervention: {args.Intervention}")
print(f"Dataset Name: {dataset_name}")


if ('Qwen-' in model_name or 'Qwen1.5' in model_name) and 'Chat' in model_name:
    
    model = AutoModelForCausalLM.from_pretrained(
        join(model_dir, model_name),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        fp32=True
    )
elif '14B' in model_name:
    print('running on bfloat16.')
    model = AutoModelForCausalLM.from_pretrained(
        join(model_dir, model_name),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        join(model_dir, model_name),
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(join(model_dir, model_name), trust_remote_code=True)

# DeepSeek-R1 and Qwen specific padding handling
if ('deepseek' in model_name.lower()) or ('deepseek' in getattr(model.config, 'model_type', '').lower()) or ('qwen' in model_name.lower()) or ('qwen' in getattr(model.config, 'model_type', '').lower()):
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
else:
    if 'llama' in model.config.model_type.lower() or 'mistral' in model.config.model_type.lower() or 'yi' in model.config.model_type.lower() or 'gptj' in model.config.model_type.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'qwen' in model.config.model_type.lower():
        tokenizer.pad_token = '<|endoftext|>'
    tokenizer.padding_side = "left"

model.to(device)


if 'gptj' in model.config.model_type.lower():
    model_layers_num = int(model.config.n_layer)  
    mlp_vector_num = 16384
    mlp_dim_num = int(model.config.n_embd)
    layer_name = 'transformer.h'
    mlp_name = 'mlp'
    mlp_last_layer_name = 'fc_out'

elif 'qwen' in model.config.model_type.lower() and 'qwen2' not in model.config.model_type.lower(): #qwen1, qwen2.5
    layer_name = 'transformer.h'
    mlp_name = 'mlp'
    mlp_last_layer_name = 'w2'
    mlp_dim_num = int(model.config.hidden_size)
    model_layers_num = int(model.config.num_hidden_layers)
    mlp_vector_num = int(model.config.intermediate_size / 2)
    
else:
    model_layers_num = int(model.config.num_hidden_layers)  # on olmo1, olmo2, qwen2, qwen2.5, llama, ...
    mlp_vector_num = int(model.config.intermediate_size)
    mlp_dim_num = int(model.config.hidden_size)
    layer_name = 'model.layers' 
    mlp_name = 'mlp'
    mlp_last_layer_name = 'down_proj'
    attn_name = 'self_attn'
    

n_new_tokens = 200

# performal normal_evaluation
if not args.Intervention:

    #ds_data = load_dataset(ds_name = dataset_name, dataset_dir=dataset_dir, split='test')
    with open(f'../../dataset/{extracting_from}samples.json', 'r', encoding='utf-8') as f:
        ds_data = json.load(f)

    print(f'****Running on {dataset_name} on {model_name} without Intervention')

    prompt_template, prompt_template_no_cot = load_prompt_template(ds_name = dataset_name, dataset_dir=dataset_dir)


    evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, run_in_fewshot=True, run_in_cot=True, 
                          intervention=False, ablation_dir=None, batch_size=BATCH_SIZE, ds_name=dataset_name, scale=0.1, device=device)
    
    with open(f'../../dataset/{extracting_from}samples-responses.json', 'w', encoding='utf-8') as f:
        json.dump(ds_data, f, ensure_ascii=False, indent=4)
    
    if dataset_name != 'MMLU-Pro':
        compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=False, ds_name=dataset_name)
    else:
        ############ retrieving reasoning and memorisation indices ############
        with open(f'../../dataset/mmlu-pro_600samples.json', 'r', encoding='utf-8') as f:
            sampled_data = json.load(f)

        reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
        memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]

        compute_performance_on_reason_memory_subset(val_sampled_data=ds_data, memory_indices=memory_indices, 
                                            reason_indices=reason_indices, intervention=False)
        ############ retrieving reasoning and memorisation indices ############
        

elif args.Intervention:

    # 1. Run non-intervention baseline and store accuracies
    # Load test data
#     ds_data = load_dataset(ds_name=dataset_name, dataset_dir=dataset_dir, split='test')
    with open(f'../../dataset/{extracting_from}samples.json', 'r', encoding='utf-8') as f:
            ds_data = json.load(f)
    prompt_template, prompt_template_no_cot = load_prompt_template(ds_name=dataset_name, dataset_dir=dataset_dir)
    ds_data = random.sample(ds_data, min(TEST_SAMPLE_SIZE, len(ds_data)))

    print(f'****Running on {dataset_name} on {model_name} WITHOUT Intervention (for baseline)')
    # Run baseline evaluation
    evaluation_on_dataset(
        model=model,
        tokenizer=tokenizer,
        val_sampled_data=ds_data,
        prompts_cot=prompt_template,
        prompts_no_cot=prompt_template_no_cot,
        run_in_fewshot=True,
        run_in_cot=True,
        intervention=False,
        ablation_dir=None,
        batch_size=BATCH_SIZE,
        ds_name=dataset_name,
        scale=0.1,
        device=device
    )
    # Compute and store baseline accuracies
    if dataset_name != 'MMLU-Pro':
        mem_acc_baseline, reason_acc_baseline = compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=False, ds_name=dataset_name), None
    else:
        with open(os.path.join(dataset_dir, f'{extracting_from}samples.json'), 'r', encoding='utf-8') as f:
            sampled_data = json.load(f)
        reason_indices = [ix for ix, sample in enumerate(ds_data) if sample.get('memory_reason_score') is not None and sample['memory_reason_score'] > 0.5]
        print(f"How many reasoning dataset did we get? {len(reason_indices)}")
        memory_indices = [ix for ix, sample in enumerate(ds_data) if sample.get('memory_reason_score') is not None and sample['memory_reason_score'] <= 0.5]
        print(f"How many memorisation dataset did we get? {len(memory_indices)}")
        mem_acc_baseline, reason_acc_baseline = compute_performance_on_reason_memory_subset(
            val_sampled_data=ds_data,
            memory_indices=memory_indices,
            reason_indices=reason_indices,
            intervention=False
        )

    # 2. Do the rest intervention as before
    # loding MMLU-Pro to get the direction
    save_path = os.path.join(args.hs_cache_dir, 'reasoning_representations_outputs')
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'), map_location=device)
    hs_cache_no_cot = loaded_dict[extracting_from] # only using the activation produced by mmlu-pro_600
    # Reload the original data to ensure 'memory_reason_score' is present
    with open(os.path.join(dataset_dir, f'{extracting_from}samples.json'), 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    # Use .get() to avoid KeyError
    reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample.get('memory_reason_score') is not None and sample['memory_reason_score'] > 0.5]
    print(f"How many reasoning dataset did we get? {len(reason_indices)}")
    memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample.get('memory_reason_score') is not None and sample['memory_reason_score'] <= 0.5]
    print(f"How many memorisation dataset did we get? {len(memory_indices)}")
    candidate_directions = get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices, device=device)

    print(f'****Running on {dataset_name} on {model_name} WITH Features Intervention')
    layers = []
    memory_accuracies = []
    reason_accuracies = []
    for layer in range(model_layers_num):
        if layer <= 2:
            continue
        print(f'Doing Intervention in Layer {layer}')
        ablation_dir = candidate_directions[layer]
        if dataset_name != 'MMLU-Pro':
            evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, ds_name=dataset_name, run_in_fewshot=True, run_in_cot=True, 
                            intervention=True, ablation_dir=ablation_dir, layer_name = layer_name, attn_name = attn_name, mlp_name = mlp_name, model_layers_num = model_layers_num, batch_size=BATCH_SIZE, scale=scale, device=device)
            mem_acc, reason_acc = compute_performance_on_reason_memory_subset(val_sampled_data=ds_data, memory_indices=memory_indices, 
                                            reason_indices=reason_indices, intervention=True, intervention_layer=layer)
        else:
            ds_data = random.sample(sampled_data, 200)
            reason_indices = [ix for ix, sample in enumerate(ds_data) if sample['memory_reason_score'] > 0.5]
            print(f"How many reasoning dataset did we get? {len(reason_indices)}")
            memory_indices = [ix for ix, sample in enumerate(ds_data) if sample['memory_reason_score'] <= 0.5]
            print(f"How many memorisation dataset did we get? {len(memory_indices)}")
            evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, ds_name=dataset_name, run_in_fewshot=True, run_in_cot=True, 
                            intervention=True, ablation_dir=ablation_dir, layer_name = layer_name, attn_name = attn_name, mlp_name = mlp_name, model_layers_num = model_layers_num, batch_size=BATCH_SIZE, scale=scale, device=device)
            mem_acc, reason_acc = compute_performance_on_reason_memory_subset(val_sampled_data=ds_data, memory_indices=memory_indices, 
                                            reason_indices=reason_indices, intervention=True, intervention_layer=layer)
        layers.append(layer)
        memory_accuracies.append(mem_acc)
        reason_accuracies.append(reason_acc)

    # Plotting logic
    figs_dir = os.path.join('../../figs_tabs')
    os.makedirs(figs_dir, exist_ok=True)
    # Memory accuracy plot
    plt.figure(figsize=(10,6))
    plt.plot(layers, memory_accuracies, marker='o', label='Memory Subset Accuracy')
    if mem_acc_baseline is not None:
        plt.axhline(mem_acc_baseline, color='red', linestyle='--', label='Non-Intervention Baseline')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Memory Subset Accuracy vs Layer for {model_name} on {dataset_name} (Intervention)')
    plt.grid(True)
    plt.legend()
    mem_fig_path = os.path.join(figs_dir, f'{model_name}_on_{dataset_name}_memory_intervention_accuracy.png')
    plt.savefig(mem_fig_path)
    plt.show()

    # Reason accuracy plot
    plt.figure(figsize=(10,6))
    plt.plot(layers, reason_accuracies, marker='o', color='orange', label='Reason Subset Accuracy')
    if reason_acc_baseline is not None:
        plt.axhline(reason_acc_baseline, color='red', linestyle='--', label='Non-Intervention Baseline')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Reason Subset Accuracy vs Layer for {model_name} on {dataset_name} (Intervention)')
    plt.grid(True)
    plt.legend()
    reason_fig_path = os.path.join(figs_dir, f'{model_name}_on_{dataset_name}_reason_intervention_accuracy.png')
    plt.savefig(reason_fig_path)
    plt.show()


# model_output_dir = os.path.join(output_dir, model_name)  
# os.makedirs(model_output_dir, exist_ok=True)

# with open(os.path.join(model_output_dir, f'{model_name}_model_mute_performance.json'), 'w', encoding='utf-8') as jsonfile:
#     json.dump(results, jsonfile, indent=4, ensure_ascii=False)

# with open(os.path.join(model_output_dir, f'{model_name}_model_mute_performance_per_concept.json'), 'w', encoding='utf-8') as jsonfile:
#     json.dump(ac_per_concept_results, jsonfile, indent=4, ensure_ascii=False)

