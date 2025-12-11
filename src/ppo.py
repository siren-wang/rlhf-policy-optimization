"""
PPO (Proximal Policy Optimization) for RLHF.
NOTE: This is a condensed version. See conversation for full detailed implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import logging
from tqdm import tqdm
from pathlib import Path
from reward_model import RewardModel
from data import PromptDataset

logger = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self, policy_model, reference_model, reward_model, tokenizer, config, device='cuda'):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.kl_coef = config.get('kl_coef', 0.1)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.max_gen_length = config.get('max_gen_length', 128)
        self.temperature = config.get('temperature', 1.0)
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.get('learning_rate', 1e-6),
            weight_decay=config.get('weight_decay', 0.01)
        )
        logger.info(f"PPO Trainer initialized")
    
    def generate_responses(self, prompts, max_length=128):
        """Generate responses from policy."""
        self.policy_model.eval()
        prompt_tokens = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
        prompt_ids = prompt_tokens['input_ids']
        prompt_len = prompt_ids.size(1)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                prompt_ids, max_length=prompt_len + max_length,
                do_sample=True, temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response_ids = outputs[:, prompt_len:]
        response_masks = (response_ids != self.tokenizer.pad_token_id).long()
        log_probs = torch.zeros(len(prompts), device=self.device)
        return response_ids, response_masks, log_probs
    
    def compute_rewards(self, prompt_ids, response_ids, response_masks):
        """Get rewards from reward model."""
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_masks = torch.cat([torch.ones_like(prompt_ids), response_masks], dim=1)
        with torch.no_grad():
            rewards = self.reward_model(full_ids, full_masks)
        return rewards
    
    def train_step(self, prompts):
        """Single PPO training step."""
        prompt_tokens = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
        prompt_ids = prompt_tokens['input_ids']
        
        response_ids, response_masks, old_log_probs = self.generate_responses(prompts, self.max_gen_length)
        rewards = self.compute_rewards(prompt_ids, response_ids, response_masks)
        
        # Simplified: just do policy gradient with reward
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Simplified loss
            loss = -(rewards * advantages).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        return {
            'loss': total_loss / self.ppo_epochs,
            'reward': rewards.mean().item(),
            'final_reward': rewards.mean().item()
        }
    
    def train(self, train_prompts, num_epochs, batch_size, output_dir):
        """Full training loop."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = PromptDataset(train_prompts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            epoch_metrics = []
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                prompts = batch['prompt']
                metrics = self.train_step(prompts)
                epoch_metrics.append(metrics)
            
            avg_metrics = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics) for k in epoch_metrics[0].keys()}
            logger.info(f"Epoch {epoch} - " + ", ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
            self.policy_model.save_pretrained(output_dir / f'epoch_{epoch}')
        
        logger.info(f"Training complete! Model saved to {output_dir}")
