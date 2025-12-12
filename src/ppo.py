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
from src.reward_model import RewardModel
from src.data import PromptDataset

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
        
        self.clip_ratio = float(config.get('clip_ratio', 0.2))
        self.kl_coef = float(config.get('kl_coef', 0.1))
        self.entropy_coef = float(config.get('entropy_coef', 0.01))
        self.ppo_epochs = int(config.get('ppo_epochs', 4))
        self.max_gen_length = int(config.get('max_gen_length', 128))
        self.temperature = float(config.get('temperature', 1.0))

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=float(config.get('learning_rate', 1e-6)),
            weight_decay=float(config.get('weight_decay', 0.01))
        )
        logger.info(f"PPO Trainer initialized")
    
    def generate_responses(self, prompts, max_length=128):
        """Generate responses from policy."""
        self.policy_model.eval()
        
        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        
        prompt_ids = prompt_tokens['input_ids']
        prompt_attention_mask = prompt_tokens['attention_mask']
        prompt_len = prompt_ids.size(1)
        
        # Generate with safer parameters
        with torch.no_grad():
            try:
                outputs = self.policy_model.generate(
                    prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                    do_sample=True,
                    temperature=max(self.temperature, 0.7),  # Ensure temperature >= 0.7
                    top_k=50,  # Add top-k sampling for stability
                    top_p=0.95,  # Add nucleus sampling
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,  # Prevent repetition
                )
            except RuntimeError as e:
                # Fallback: use greedy decoding if sampling fails
                logger.warning(f"Sampling failed, using greedy decoding: {e}")
                outputs = self.policy_model.generate(
                    prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        response_ids = outputs[:, prompt_len:]  # Extract generated part
        
        # Create attention mask
        response_masks = (response_ids != self.tokenizer.pad_token_id).long()
        
        # Compute log probabilities (simplified - set to zeros for now)
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
        # Handle edge case: need at least 2 samples for std calculation
        if len(prompts) < 2:
            prompts = prompts * 2  # Duplicate if only 1 prompt

        prompt_tokens = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
        prompt_ids = prompt_tokens['input_ids']

        # Generate responses
        response_ids, response_masks, old_log_probs = self.generate_responses(prompts, self.max_gen_length)

        # Compute rewards
        rewards = self.compute_rewards(prompt_ids, response_ids, response_masks)

        # Compute advantages with numerical stability
        if len(rewards) > 1:
            advantages = rewards - rewards.mean()
            advantage_std = advantages.std()
            if advantage_std > 1e-8:
                advantages = advantages / advantage_std
            else:
                advantages = advantages  # Don't normalize if std is too small
        else:
            advantages = rewards  # Single sample, no normalization

        # PPO updates
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Forward pass through policy
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([
                torch.ones_like(prompt_ids),
                response_masks
            ], dim=1)
            
            outputs = self.policy_model(
                full_ids,
                attention_mask=full_attention_mask
            )
            logits = outputs.logits[:, prompt_ids.size(1)-1:-1, :]  # Get logits for generated tokens
            
            # Compute log probabilities
            log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Gather log probs for actual generated tokens
            response_tokens = response_ids.unsqueeze(-1)
            log_probs = torch.gather(log_probs_all, -1, response_tokens).squeeze(-1)
            
            # Apply mask and sum
            log_probs_sum = (log_probs * response_masks).sum(dim=1)
            mask_sum = response_masks.sum(dim=1).clamp(min=1)
            log_probs_avg = log_probs_sum / mask_sum
            
            # PPO loss (simplified - just policy gradient)
            loss = -(log_probs_avg * advantages.detach()).mean()
            
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
