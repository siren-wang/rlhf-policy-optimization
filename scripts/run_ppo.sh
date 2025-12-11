#!/bin/bash
# Train PPO model

CONFIG=${1:-configs/prototype.yaml}

python -c "
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.ppo import PPOTrainer
from src.reward_model import RewardModel
from src.data import load_hh_rlhf_data, create_dummy_data

print('Training PPO...')
config = yaml.safe_load(open('$CONFIG'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(config['model_name'])
reference_model = AutoModelForCausalLM.from_pretrained(config['model_name'])
reward_model = RewardModel.load(
    Path(config['output_dir']) / 'reward_model' / 'best_model.pt',
    config['model_name']
)

if config['data'].get('use_dummy_data', False):
    data = create_dummy_data(config['data'].get('num_samples', 10))
else:
    data = load_hh_rlhf_data('train', config['data'].get('num_train_samples'))

prompts = [item['prompt'] for item in data[:100]]

trainer = PPOTrainer(policy_model, reference_model, reward_model, tokenizer, config['ppo'], device)
output_dir = Path(config['output_dir']) / 'ppo_model'
trainer.train(prompts, config['ppo']['num_epochs'], config['ppo']['batch_size'], output_dir)
"
