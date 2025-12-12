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
    
    def generate_group_responses(
        self,
        prompt: str,
        group_size: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses for a single prompt.
        
        Returns:
            responses: List of generated texts
            response_ids: (group_size, max_response_len)
            response_masks: (group_size, max_response_len)
        """
        self.policy_model.eval()
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.device)
        
        prompt_ids = prompt_tokens['input_ids']
        prompt_len = prompt_ids.size(1)
        
        # Generate multiple responses
        all_response_ids = []
        all_response_texts = []
        
        for _ in range(group_size):
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_gen_length,
                    do_sample=True,
                    temperature=max(self.temperature, 0.7),
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )
            
            response_ids = outputs[:, prompt_len:]  # Extract generated part
            all_response_ids.append(response_ids)
            
            # Decode
            response_text = self.tokenizer.decode(
                response_ids[0],
                skip_special_tokens=True
            )
            all_response_texts.append(response_text)
        
        # Find max length among all responses
        max_response_len = max(r.size(1) for r in all_response_ids)
        
        # Pad all responses to same length
        padded_responses = []
        for resp in all_response_ids:
            if resp.size(1) < max_response_len:
                # Pad on the right
                padding = torch.full(
                    (resp.size(0), max_response_len - resp.size(1)),
                    self.tokenizer.pad_token_id,
                    dtype=resp.dtype,
                    device=resp.device
                )
                resp = torch.cat([resp, padding], dim=1)
            padded_responses.append(resp)
        
        # Stack responses
        response_ids = torch.cat(padded_responses, dim=0)  # (group_size, max_len)
        response_masks = (response_ids != self.tokenizer.pad_token_id).long()
        
        return all_response_texts, response_ids, response_masks
    
    def compute_group_advantages(
        self,
        prompt: str,
        response_ids: torch.Tensor,
        response_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages relative to group mean.
        
        Args:
            prompt: The prompt string
            response_ids: (group_size, response_len)
            response_masks: (group_size, response_len)
        
        Returns:
            advantages: (group_size,)
            rewards: (group_size,)
            kl_per_sample: (group_size,)
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.device)
        prompt_ids = prompt_tokens['input_ids']
        
        # Repeat for group
        prompt_ids = prompt_ids.repeat(self.group_size, 1)
        
        # Compute rewards
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_masks = torch.cat([
            torch.ones_like(prompt_ids),
            response_masks
        ], dim=1)
        
        with torch.no_grad():
            rewards = self.reward_model(full_ids, full_masks)
        
        # Compute KL divergence
        policy_outputs = self.policy_model(full_ids, attention_mask=full_masks)
        policy_logits = policy_outputs.logits[:, :-1]
        
        with torch.no_grad():
            ref_outputs = self.reference_model(full_ids, attention_mask=full_masks)
            ref_logits = ref_outputs.logits[:, :-1]
        
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        kl = F.kl_div(policy_logprobs, ref_logprobs, reduction='none', log_target=True)
        kl = kl.sum(dim=-1)  # Sum over vocab
        
        response_start = prompt_ids.size(1) - 1
        response_part = kl[:, response_start:]
        response_masks_for_kl = response_masks
        
        kl_per_sample = (response_part * response_masks_for_kl).sum(dim=1) / response_masks_for_kl.sum(dim=1).clamp(min=1)
        
        # Final rewards with KL penalty
        final_rewards = rewards - self.kl_coef * kl_per_sample
        
        # Advantages: relative to group mean
        advantages = final_rewards - final_rewards.mean()
        
        return advantages, rewards, kl_per_sample

    def train_step(self, prompt):
        """Single GRPO training step for one prompt."""
        # Generate group responses
        responses, response_ids, response_masks = self.generate_group_responses(
            prompt,
            self.group_size
        )

        # Compute advantages
        advantages, rewards, kl = self.compute_group_advantages(
            prompt,
            response_ids,
            response_masks
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy gradient loss
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.device)
        prompt_ids = prompt_tokens['input_ids'].repeat(self.group_size, 1)

        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_attention_mask = torch.cat([
            torch.ones_like(prompt_ids),
            response_masks
        ], dim=1)

        outputs = self.policy_model(full_ids, attention_mask=full_attention_mask)
        logits = outputs.logits[:, :-1]

        # Get log probs for generated tokens
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Get log probs for the actual generated tokens
        target_ids = full_ids[:, 1:]  # Shift targets
        log_probs = torch.gather(log_probs_all, -1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Only consider the response part (not the prompt)
        response_start = prompt_ids.size(1)
        response_log_probs = log_probs[:, response_start-1:]

        # Apply mask and compute average log prob per sequence
        response_log_probs = (response_log_probs * response_masks).sum(dim=1) / response_masks.sum(dim=1).clamp(min=1)

        # Policy gradient loss: maximize log_prob weighted by advantage
        # advantages shape: (group_size,), response_log_probs shape: (group_size,)
        loss = -(response_log_probs * advantages.detach()).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {
        'loss': loss.item(),
        'reward': rewards.mean().item(),
        'kl': kl.mean().item(),
        'advantage_std': advantages.std().item()
        }
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
