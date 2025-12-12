"""
GRPO (Group Relative Policy Optimization).
NOTE: Condensed version. See conversation for full implementation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
from src.reward_model import RewardModel
from src.data import PromptDataset

logger = logging.getLogger(__name__)

class GRPOTrainer:
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
        
        self.group_size = int(config.get('group_size', 4))
        self.kl_coef = float(config.get('kl_coef', 0.1))
        self.max_gen_length = int(config.get('max_gen_length', 128))
        self.temperature = float(config.get('temperature', 1.0))

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=float(config.get('learning_rate', 1e-6)),
            weight_decay=float(config.get('weight_decay', 0.01))
        )
        logger.info(f"GRPO Trainer initialized with group_size={self.group_size}")
    
    def generate_group_responses(self, prompt, group_size):
        """Generate multiple responses for one prompt."""
        self.policy_model.eval()
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_ids = prompt_tokens['input_ids']
        prompt_len = prompt_ids.size(1)
        
        all_response_ids = []
        all_response_texts = []
        
        for _ in range(group_size):
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    prompt_ids, max_length=prompt_len + self.max_gen_length,
                    do_sample=True, temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            response_ids = outputs[:, prompt_len:]
            all_response_ids.append(response_ids)
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            all_response_texts.append(response_text)
        
        response_ids = torch.cat(all_response_ids, dim=0)
        response_masks = (response_ids != self.tokenizer.pad_token_id).long()
        return all_response_texts, response_ids, response_masks
    
    def train_step(self, prompt):
        """Single GRPO training step."""
        responses, response_ids, response_masks = self.generate_group_responses(prompt, self.group_size)
        
        # Compute rewards
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_ids = prompt_tokens['input_ids'].repeat(self.group_size, 1)
        
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_masks = torch.cat([torch.ones_like(prompt_ids), response_masks], dim=1)
        
        with torch.no_grad():
            rewards = self.reward_model(full_ids, full_masks)
        
        # Group-relative advantages
        advantages = rewards - rewards.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Simple policy gradient
        outputs = self.policy_model(full_ids)
        logits = outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(log_probs.mean(dim=-1) * advantages.detach()).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'reward': rewards.mean().item(), 'advantage_std': advantages.std().item()}
    
    def train(self, train_prompts, num_epochs, output_dir):
        """Full training loop."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            epoch_metrics = []
            for prompt in tqdm(train_prompts, desc=f"Epoch {epoch}"):
                metrics = self.train_step(prompt)
                epoch_metrics.append(metrics)
            
            avg_metrics = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics) for k in epoch_metrics[0].keys()}
            logger.info(f"Epoch {epoch} - " + ", ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
            self.policy_model.save_pretrained(output_dir / f'epoch_{epoch}')
        
        logger.info(f"Training complete! Model saved to {output_dir}")
