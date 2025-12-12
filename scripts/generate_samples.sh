#!/bin/bash
# Generate sample outputs with reward scores

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -c "
import json
import torch
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
    reward_model_name = reward_config.get('model_name', 'gpt2')  # Default to gpt2

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

# Load reward model with correct architecture
print(f'\\nLoading reward model ({reward_model_name})...')
reward_model = RewardModel.load('outputs/full/reward_model/best_model.pt', reward_model_name)
reward_model = reward_model.to(device)
reward_model.eval()
print('✓ Reward model loaded')

# Models to evaluate
models = {
    'reference': reward_model_name,  # Use same as reward model
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
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip()
            
            samples.append({
                'prompt': prompt,
                'response': response,
                'full_text': full_text,
                'reward': reward.item(),
            })
        
        all_samples[model_name] = samples
        mean_reward = sum(s['reward'] for s in samples) / len(samples)
        print(f'  ✓ Generated {len(samples)} samples (mean reward: {mean_reward:.4f})')
        
        # Save individual model samples
        output_file = OUTPUT_DIR / f'{model_name}_samples.json'
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f'  Saved to {output_file}')
        
    except Exception as e:
        print(f'  ✗ Error: {e}')
        all_samples[model_name] = []

# Save combined samples
combined_file = OUTPUT_DIR / 'all_samples.json'
with open(combined_file, 'w') as f:
    json.dump(all_samples, f, indent=2)
print(f'\\n✓ All samples saved to {combined_file}')

# Create human-readable markdown with rewards
md_file = OUTPUT_DIR / 'samples.md'
with open(md_file, 'w') as f:
    f.write('# Generated Samples Comparison\\n\\n')
    f.write(f'Generated {NUM_SAMPLES} samples from each model.\\n\\n')
    
    # Summary statistics
    f.write('## Summary Statistics\\n\\n')
    f.write('| Model | Mean Reward | Std Dev |\\n')
    f.write('|-------|-------------|---------|\\n')
    for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
        if model_name in all_samples and all_samples[model_name]:
            rewards = [s['reward'] for s in all_samples[model_name]]
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
            f.write(f'| {model_name.upper()} | {mean_r:.4f} | {std_r:.4f} |\\n')
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
                f.write(f'### {model_name.upper()} (Reward: {reward:.4f})\\n')
                f.write(f'{response}\\n\\n')
        
        f.write('---\\n\\n')

print(f'✓ Human-readable samples saved to {md_file}')
print(f'\\n✓ Done! Generated samples with rewards in output_samples/')

# Print summary
print('\\n' + '='*60)
print('Reward Summary:')
print('='*60)
for model_name in ['reference', 'ppo', 'grpo', 'dpo']:
    if model_name in all_samples and all_samples[model_name]:
        rewards = [s['reward'] for s in all_samples[model_name]]
        mean_r = sum(rewards) / len(rewards)
        print(f'{model_name.upper():12s}: {mean_r:.4f}')
print('='*60)
"
