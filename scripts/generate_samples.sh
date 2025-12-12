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
import yaml

# Configuration
OUTPUT_DIR = Path('output_samples')
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 20

# Load config to get correct model name
with open('outputs/full/reward_model/config.yaml') as f:
    reward_config = yaml.safe_load(f)
    reward_model_name = reward_config.get('model_name', 'gpt2')

print(f'Reward model trained with: {reward_model_name}')

# Test prompts (diverse set)
prompts = [
    'Human: What is the capital of France?\n\nAssistant:',
    'Human: How do I learn Python programming?\n\nAssistant:',
    'Human: Explain quantum computing in simple terms.\n\nAssistant:',
    'Human: What are the benefits of exercise?\n\nAssistant:',
    'Human: How does photosynthesis work?\n\nAssistant:',
    'Human: What is machine learning?\n\nAssistant:',
    'Human: How can I improve my sleep quality?\n\nAssistant:',
    'Human: What causes climate change?\n\nAssistant:',
    'Human: Explain the theory of relativity.\n\nAssistant:',
    'Human: How do vaccines work?\n\nAssistant:',
    'Human: What is the water cycle?\n\nAssistant:',
    'Human: How do computers process information?\n\nAssistant:',
    'Human: What are the main causes of pollution?\n\nAssistant:',
    'Human: How does the human immune system work?\n\nAssistant:',
    'Human: What is artificial intelligence?\n\nAssistant:',
    'Human: How do plants grow?\n\nAssistant:',
    'Human: What is the greenhouse effect?\n\nAssistant:',
    'Human: How does memory work in the brain?\n\nAssistant:',
    'Human: What are renewable energy sources?\n\nAssistant:',
    'Human: How do antibiotics work?\n\nAssistant:',
][:NUM_SAMPLES]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load reward model
print(f'\\nLoading reward model ({reward_model_name})...')
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
            
            tokens = tokenizer(prompt, return_tensors='pt').to(device)
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
    f.write(f'Generated {NUM_SAMPLES} samples from each model.\\n\\n')
    
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
    
    # Individual examples
    for i, prompt in enumerate(prompts):
        f.write(f'## Example {i+1}\\n\\n')
        f.write(f'**Prompt:** {prompt}\\n\\n')
        
        for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
            if model_name in all_samples and i < len(all_samples[model_name]):
                sample = all_samples[model_name][i]
                response = sample['response']
                reward = sample['reward']
                kl = sample['kl']
                f.write(f'### {model_name.upper()} (Reward: {reward:.4f}, KL: {kl:.4f})\\n')
                f.write(f'{response}\\n\\n')
        
        f.write('---\\n\\n')

print(f'✓ Human-readable samples saved to {md_file}')
print(f'\\n✓ Done! Generated samples with rewards and KL in output_samples/')

# Print summary
print('\\n' + '='*80)
print('Performance Summary:')
print('='*80)
# Fixed the f-string syntax here
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
