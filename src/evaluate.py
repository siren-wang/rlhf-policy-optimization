"""Evaluation utilities for trained models."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
from pathlib import Path
import json
import logging
from src.reward_model import RewardModel

logger = logging.getLogger(__name__)

def generate_samples(model, tokenizer, prompts, max_length=128, temperature=1.0, device='cuda'):
    """Generate responses for prompts."""
    model.eval()
    model = model.to(device)
    responses = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                **tokens, max_length=tokens['input_ids'].size(1) + max_length,
                do_sample=True, temperature=temperature,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0, tokens['input_ids'].size(1):], skip_special_tokens=True)
        responses.append(response)
    return responses

def compute_reward_scores(reward_model, tokenizer, prompts, responses, device='cuda'):
    """Compute reward scores."""
    reward_model.eval()
    reward_model = reward_model.to(device)
    rewards = []
    for prompt, response in zip(prompts, responses):
        text = prompt + response
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            reward = reward_model(tokens['input_ids'], tokens['attention_mask'])
        rewards.append(reward.item())
    return np.array(rewards)

def compute_kl_divergence(policy_model, reference_model, tokenizer, prompts, responses, device='cuda'):
    """Compute KL divergence."""
    policy_model.eval()
    reference_model.eval()
    policy_model = policy_model.to(device)
    reference_model = reference_model.to(device)
    kl_divs = []
    for prompt, response in zip(prompts, responses):
        text = prompt + response
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            policy_logits = policy_model(**tokens).logits
            ref_logits = reference_model(**tokens).logits
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        kl = F.kl_div(policy_logprobs, ref_logprobs, reduction='batchmean', log_target=True)
        kl_divs.append(kl.item())
    return np.array(kl_divs)

def compute_win_rate(model_responses, reference_responses, use_gpt4=False):
    """Compute win rate (simple heuristic)."""
    wins = sum(1 for m, r in zip(model_responses, reference_responses) if len(m) > len(r) * 1.2)
    return wins / len(model_responses)

def evaluate_model(model_name, model, reference_model, reward_model, tokenizer, test_prompts, output_dir, device='cuda'):
    """Comprehensive evaluation."""
    logger.info(f"Evaluating {model_name}...")
    responses = generate_samples(model, tokenizer, test_prompts, device=device)
    ref_responses = generate_samples(reference_model, tokenizer, test_prompts, device=device)
    rewards = compute_reward_scores(reward_model, tokenizer, test_prompts, responses, device)
    kl_divs = compute_kl_divergence(model, reference_model, tokenizer, test_prompts, responses, device)
    win_rate = compute_win_rate(responses, ref_responses)
    
    samples_file = output_dir / f'{model_name}_samples.json'
    with open(samples_file, 'w') as f:
        samples = [{'prompt': p, 'response': r, 'reward': float(rew), 'kl': float(kl)}
                   for p, r, rew, kl in zip(test_prompts, responses, rewards, kl_divs)]
        json.dump(samples, f, indent=2)
    
    return {
        'model_name': model_name,
        'mean_reward': float(rewards.mean()),
        'std_reward': float(rewards.std()),
        'mean_kl': float(kl_divs.mean()),
        'std_kl': float(kl_divs.std()),
        'win_rate': win_rate,
        'samples_file': str(samples_file)
    }
