#!/bin/bash
# Generate sample outputs with reward scores and KL divergence

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -c "
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward_model import RewardModel
from src.data import load_hh_rlhf_data, create_dummy_data
import yaml

# Configuration
OUTPUT_DIR = Path('output_samples')
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 100

# Load config to get correct model name
with open('outputs/full/reward_model/config.yaml') as f:
    reward_config = yaml.safe_load(f)
    reward_model_name = reward_config.get('model_name', 'gpt2')

print(f'Reward model trained with: {reward_model_name}')

# Load training prompts to check for overlap
print(f'\\nLoading training data to check for overlap...')
try:
    train_data = load_hh_rlhf_data('train', 10000)  # Load all training data
    train_prompts = set(item['prompt'] for item in train_data)
    print(f'✓ Loaded {len(train_prompts)} unique training prompts')
except Exception as e:
    print(f'⚠ Could not load training data: {e}')
    train_prompts = set()

# Load test prompts and filter out overlaps
print(f'\\nLoading test prompts from HH-RLHF...')
try:
    # Load more than needed in case we need to filter
    test_data = load_hh_rlhf_data('test', NUM_SAMPLES * 3)
    
    # Filter out any prompts that appear in training set
    test_data_filtered = []
    for item in test_data:
        if item['prompt'] not in train_prompts:
            test_data_filtered.append(item)
        if len(test_data_filtered) >= NUM_SAMPLES:
            break
    
    if train_prompts:
        overlap_count = len(test_data) - len(test_data_filtered)
        print(f'✓ Found {overlap_count} overlapping prompts (filtered out)')
    
    prompts = [item['prompt'] for item in test_data_filtered[:NUM_SAMPLES]]
    print(f'✓ Using {len(prompts)} non-overlapping test prompts')
    
except Exception as e:
    print(f'⚠ Could not load HH-RLHF data: {e}')
    print('Using dummy data instead...')
    test_data = create_dummy_data(NUM_SAMPLES)
    prompts = [item['prompt'] for item in test_data]

# Show first prompt as example
print(f'\\nExample test prompt:')
print(prompts[0][:200] + '...' if len(prompts[0]) > 200 else prompts[0])
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load reward model
print(f'Loading reward model ({reward_model_name})...')
reward_model = RewardModel.load('outputs/full/reward_model/best_model.pt', reward_model_name)
reward_model = reward_model.to(device)
reward_model.eval()
print('✓ Reward model loaded')

# Load reference model for KL computation
print(f'Loading reference model ({reward_model_name})...')
reference_model = AutoModelForCausalLM.from_pretrained(reward_model_name).to(device)
reference_model.eval()
print('✓ Reference model loaded')

# Models to evaluate
models = {
    'reference': reward_model_name,
    'ppo': 'outputs/full/ppo_model/epoch_5',
    'grpo': 'outputs/full/grpo_model/epoch_5',
    'dpo': 'outputs/full/dpo_model/epoch_3',
}

all_samples = {}

for model_name, model_path in models.items():
    print(f'\\nGenerating samples from {model_name}...')
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()
        
        samples = []
        
        for i, prompt in enumerate(prompts):
            print(f'  {i+1}/{len(prompts)}', end='\\r')
            
            tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            prompt_length = tokens['input_ids'].size(1)
            
            with torch.no_grad():
                # Generate response
                outputs = model.generate(
                    **tokens,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Compute reward
                full_mask = torch.ones_like(outputs)
                reward = reward_model(outputs, full_mask)
                
                # Compute KL divergence (for non-reference models)
                if model_name != 'reference':
                    # Get logits from both models
                    policy_outputs = model(outputs, attention_mask=full_mask)
                    ref_outputs = reference_model(outputs, attention_mask=full_mask)
                    
                    # Focus on generated tokens only
                    policy_logits = policy_outputs.logits[:, prompt_length-1:-1, :]
                    ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
                    
                    # Compute log probabilities
                    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
                    ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                    
                    # KL divergence: KL(policy || reference)
                    kl_div = F.kl_div(
                        policy_logprobs,
                        ref_logprobs,
                        reduction='batchmean',
                        log_target=True
                    )
                    kl_value = kl_div.item()
                else:
                    kl_value = 0.0
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip()
            
            samples.append({
                'prompt': prompt,
                'response': response,
                'full_text': full_text,
                'reward': reward.item(),
                'kl': kl_value,
            })
        
        all_samples[model_name] = samples
        mean_reward = sum(s['reward'] for s in samples) / len(samples)
        mean_kl = sum(s['kl'] for s in samples) / len(samples)
        print(f'  ✓ Generated {len(samples)} samples (reward: {mean_reward:.4f}, KL: {mean_kl:.4f})')
        
        # Save individual model samples
        output_file = OUTPUT_DIR / f'{model_name}_samples.json'
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f'  Saved to {output_file}')
        
        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()
        all_samples[model_name] = []

# Save combined samples
combined_file = OUTPUT_DIR / 'all_samples.json'
with open(combined_file, 'w') as f:
    json.dump(all_samples, f, indent=2)
print(f'\\n✓ All samples saved to {combined_file}')

# Create human-readable markdown with rewards and KL
md_file = OUTPUT_DIR / 'samples.md'
with open(md_file, 'w') as f:
    f.write('# Generated Samples Comparison\\n\\n')
    f.write(f'Generated {NUM_SAMPLES} samples from HH-RLHF test set (non-overlapping with training).\\n\\n')
    
    # Summary statistics
    f.write('## Summary Statistics\\n\\n')
    f.write('| Model | Mean Reward | Std Reward | Mean KL | Std KL |\\n')
    f.write('|-------|-------------|------------|---------|--------|\\n')
    for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
        if model_name in all_samples and all_samples[model_name]:
            rewards = [s['reward'] for s in all_samples[model_name]]
            kls = [s['kl'] for s in all_samples[model_name]]
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
            mean_kl = sum(kls) / len(kls)
            std_kl = (sum((k - mean_kl)**2 for k in kls) / len(kls)) ** 0.5
            f.write(f'| {model_name.upper()} | {mean_r:.4f} | {std_r:.4f} | {mean_kl:.4f} | {std_kl:.4f} |\\n')
    f.write('\\n')
    
    # Individual examples (first 5)
    for i in range(min(5, len(prompts))):
        f.write(f'## Example {i+1}\\n\\n')
        # Truncate long prompts
        prompt_display = prompts[i][:300] + '...' if len(prompts[i]) > 300 else prompts[i]
        f.write(f'**Prompt:** {prompt_display}\\n\\n')
        
        for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
            if model_name in all_samples and i < len(all_samples[model_name]):
                sample = all_samples[model_name][i]
                response = sample['response']
                reward = sample['reward']
                kl = sample['kl']
                # Truncate long responses
                response_display = response[:300] + '...' if len(response) > 300 else response
                f.write(f'### {model_name.upper()} (Reward: {reward:.4f}, KL: {kl:.4f})\\n')
                f.write(f'{response_display}\\n\\n')
        
        f.write('---\\n\\n')

print(f'✓ Human-readable samples saved to {md_file}')
print(f'\\n✓ Done! Generated samples with rewards and KL in output_samples/')

# Print summary
print('\\n' + '='*80)
print('Performance Summary (HH-RLHF Test Set - No Train Overlap):')
print('='*80)
header_model = 'Model'
header_reward = 'Reward'
header_kl = 'KL'
print(f'{header_model:12s} | {header_reward:>10s} | {header_kl:>10s}')
print('-'*80)
for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
    if model_name in all_samples and all_samples[model_name]:
        rewards = [s['reward'] for s in all_samples[model_name]]
        kls = [s['kl'] for s in all_samples[model_name]]
        mean_r = sum(rewards) / len(rewards)
        mean_kl = sum(kls) / len(kls)
        print(f'{model_name.upper():12s} | {mean_r:10.4f} | {mean_kl:10.4f}')
print('='*80)
"
