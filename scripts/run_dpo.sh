#!/bin/bash
# Train DPO model

CONFIG=${1:-configs/prototype.yaml}

python -c "
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dpo import DPOTrainer
from src.data import prepare_datasets

print('Training DPO...')
config = yaml.safe_load(open('$CONFIG'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(config['model_name'])
reference_model = AutoModelForCausalLM.from_pretrained(config['model_name'])

train_dataset, val_dataset = prepare_datasets(config['data'], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=config['dpo']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['dpo']['batch_size'], shuffle=False)

trainer = DPOTrainer(policy_model, reference_model, tokenizer, config['dpo'], device)
output_dir = Path(config['output_dir']) / 'dpo_model'
trainer.train(train_loader, val_loader, config['dpo']['num_epochs'], output_dir)
"
