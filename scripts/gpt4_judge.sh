#!/bin/bash
# Use GPT-4o-mini to judge which response is better

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

python -c "
import json
import os
from pathlib import Path
import random

# Check for API key
if 'OPENAI_API_KEY' not in os.environ:
    print('❌ Error: OPENAI_API_KEY environment variable not set')
    print('Set it with: export OPENAI_API_KEY=your-key-here')
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print('❌ Error: openai package not installed')
    print('Install it with: pip install openai')
    exit(1)

client = OpenAI()

# Configuration
NUM_EVAL_SAMPLES = 100  # Evaluate 100 samples per model

# Load samples
samples_dir = Path('output_samples')

print('Loading samples...')
with open(samples_dir / 'reference_samples.json') as f:
    reference_samples = json.load(f)

# Limit to NUM_EVAL_SAMPLES
reference_samples = reference_samples[:NUM_EVAL_SAMPLES]
num_samples = len(reference_samples)

print(f'Will evaluate {num_samples} samples per model')
print()

models = ['ppo', 'grpo', 'dpo']
all_results = {}

print('='*80)
print(f'GPT-4o-mini as Judge - Win Rate Evaluation ({num_samples} samples)')
print('='*80)
print()

for model in models:
    print(f'\\nEvaluating {model.upper()}...')
    print('-'*80)
    
    with open(samples_dir / f'{model}_samples.json') as f:
        model_samples = json.load(f)
    
    model_samples = model_samples[:NUM_EVAL_SAMPLES]
    
    wins = 0
    ties = 0
    losses = 0
    errors = 0
    
    # Store individual judgments for analysis
    judgments = []
    
    for i, (ref_sample, model_sample) in enumerate(zip(reference_samples, model_samples)):
        print(f'  Sample {i+1}/{num_samples}...', end='\\r')
        
        prompt_text = ref_sample['prompt']
        response_ref = ref_sample['response']
        response_model = model_sample['response']
        
        # Randomize order to avoid position bias
        if random.random() > 0.5:
            response_a = response_model
            response_b = response_ref
            flipped = True
        else:
            response_a = response_ref
            response_b = response_model
            flipped = False
        
        # Truncate very long responses for API
        max_len = 500
        if len(response_a) > max_len:
            response_a = response_a[:max_len] + '...'
        if len(response_b) > max_len:
            response_b = response_b[:max_len] + '...'
        if len(prompt_text) > 300:
            prompt_text = prompt_text[:300] + '...'
        
        # Ask GPT-4o-mini to judge
        judgment_prompt = f\"\"\"You are an expert evaluator. Compare these two AI assistant responses to the same user prompt.

Consider:
- Helpfulness and relevance
- Accuracy and correctness
- Clarity and coherence
- Completeness

User Prompt:
{prompt_text}

Response A:
{response_a}

Response B:
{response_b}

Which response is better overall? Reply with ONLY one of these exact words:
- A (if Response A is better)
- B (if Response B is better)
- TIE (if they are equally good)

Your judgment:\"\"\"

        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': judgment_prompt}],
                temperature=0,
                max_tokens=10
            )
            
            judgment = response.choices[0].message.content.strip().upper()
            
            # Parse judgment
            if 'A' in judgment and 'B' not in judgment:
                judgment = 'A'
            elif 'B' in judgment and 'A' not in judgment:
                judgment = 'B'
            elif 'TIE' in judgment:
                judgment = 'TIE'
            else:
                judgment = 'TIE'
            
            # Account for flipping
            if flipped:
                if judgment == 'A':
                    judgment = 'B'
                elif judgment == 'B':
                    judgment = 'A'
            
            # Count wins (model vs reference)
            if judgment == 'B':
                wins += 1
                result = 'win'
            elif judgment == 'A':
                losses += 1
                result = 'loss'
            else:
                ties += 1
                result = 'tie'
            
            judgments.append({
                'sample_id': i,
                'judgment': result,
                'flipped': flipped
            })
                
        except Exception as e:
            print(f'\\n  Error on sample {i}: {e}')
            errors += 1
            ties += 1
            judgments.append({
                'sample_id': i,
                'judgment': 'error',
                'error': str(e)
            })
    
    total = num_samples
    win_rate = wins / total * 100
    tie_rate = ties / total * 100
    loss_rate = losses / total * 100
    
    all_results[model] = {
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'errors': errors,
        'total': total,
        'win_rate': win_rate,
        'tie_rate': tie_rate,
        'loss_rate': loss_rate,
        'judgments': judgments
    }
    
    print(f'\\n  Results for {model.upper()}:')
    print(f'    Wins:   {wins:3d} ({win_rate:5.1f}%)')
    print(f'    Ties:   {ties:3d} ({tie_rate:5.1f}%)')
    print(f'    Losses: {losses:3d} ({loss_rate:5.1f}%)')
    if errors > 0:
        print(f'    Errors: {errors:3d}')
    print(f'    Win Rate: {win_rate:.1f}%')

# Save detailed results
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

with open(results_dir / 'gpt4_win_rates.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Create summary
summary = {
    'model': 'gpt-4o-mini',
    'num_samples': num_samples,
    'results': {
        model: {
            'win_rate': all_results[model]['win_rate'],
            'wins': all_results[model]['wins'],
            'ties': all_results[model]['ties'],
            'losses': all_results[model]['losses']
        }
        for model in models
    }
}

with open(results_dir / 'gpt4_win_rates_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create markdown report
with open(results_dir / 'WIN_RATES.md', 'w') as f:
    f.write('# Win Rate Evaluation\\n\\n')
    f.write(f'Evaluated by: GPT-4o-mini\\n')
    f.write(f'Test samples: {num_samples}\\n\\n')
    
    f.write('## Results\\n\\n')
    f.write('| Model | Wins | Ties | Losses | Win Rate |\\n')
    f.write('|-------|------|------|--------|----------|\\n')
    
    for model in ['ppo', 'grpo', 'dpo']:
        r = all_results[model]
        f.write(f\"| {model.upper()} | {r['wins']} | {r['ties']} | {r['losses']} | {r['win_rate']:.1f}% |\\n\")
    
    f.write('\\n## Interpretation\\n\\n')
    f.write('- **Win**: Model response judged better than reference\\n')
    f.write('- **Tie**: Both responses equally good\\n')
    f.write('- **Loss**: Reference response judged better\\n')

print('\\n' + '='*80)
print('GPT-4o-mini Win Rate Summary:')
print('='*80)
# Fixed f-string syntax
header_model = 'Model'
header_win_rate = 'Win Rate'
header_wins = 'Wins'
header_ties = 'Ties'
header_losses = 'Losses'
print(f'{header_model:12s} | {header_win_rate:>10s} | {header_wins:>6s} | {header_ties:>6s} | {header_losses:>6s}')
print('-'*80)
for model in ['ppo', 'grpo', 'dpo']:
    r = all_results[model]
    print(f\"{model.upper():12s} | {r['win_rate']:9.1f}% | {r['wins']:6d} | {r['ties']:6d} | {r['losses']:6d}\")
print('='*80)

print('\\n✓ Results saved to:')
print('  - results/gpt4_win_rates.json (detailed)')
print('  - results/gpt4_win_rates_summary.json (summary)')
print('  - results/WIN_RATES.md (markdown report)')
print()
print(f'Total API calls: {num_samples * len(models)}')
print(f'Estimated cost: ~$0.{int(num_samples * len(models) * 0.015 / 100):02d} USD')
"
