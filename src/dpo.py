"""
DPO (Direct Preference Optimization).
NOTE: Condensed version. See conversation for full implementation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm
from pathlib import Path
from data import PreferenceDataset

logger = logging.getLogger(__name__)

class DPOTrainer:
    def __init__(self, policy_model, reference_model, tokenizer, config, device='cuda'):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.beta = config.get('beta', 0.1)
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.get('learning_rate', 5e-7),
            weight_decay=config.get('weight_decay', 0.01)
        )
        logger.info(f"DPO Trainer initialized with beta={self.beta}")
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        """Compute average log probability."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, 1:]
        gathered_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:]
        avg_log_probs = (gathered_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
        return avg_log_probs
    
    def compute_dpo_loss(self, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask):
        """Compute DPO loss."""
        policy_chosen_logprobs = self.compute_log_probs(self.policy_model, chosen_input_ids, chosen_attention_mask)
        policy_rejected_logprobs = self.compute_log_probs(self.policy_model, rejected_input_ids, rejected_attention_mask)
        
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_log_probs(self.reference_model, chosen_input_ids, chosen_attention_mask)
            ref_rejected_logprobs = self.compute_log_probs(self.reference_model, rejected_input_ids, rejected_attention_mask)
        
        implicit_reward_chosen = policy_chosen_logprobs - ref_chosen_logprobs
        implicit_reward_rejected = policy_rejected_logprobs - ref_rejected_logprobs
        
        logits = self.beta * (implicit_reward_chosen - implicit_reward_rejected)
        loss = -F.logsigmoid(logits).mean()
        accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
        
        return loss, accuracy, implicit_reward_chosen.mean(), implicit_reward_rejected.mean()
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.policy_model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            
            loss, accuracy, _, _ = self.compute_dpo_loss(
                chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{accuracy.item():.4f}"})
        
        return {'loss': total_loss / num_batches, 'accuracy': total_accuracy / num_batches}
    
    def train(self, train_dataloader, val_dataloader, num_epochs, output_dir):
        """Full training loop."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            train_metrics = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            self.policy_model.save_pretrained(output_dir / f'epoch_{epoch}')
        logger.info(f"Training complete! Model saved to {output_dir}")
    
    def evaluate(self, dataloader):
        """Evaluate."""
        self.policy_model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                loss, accuracy, _, _ = self.compute_dpo_loss(
                    chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
                )
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        return {'loss': total_loss / num_batches, 'accuracy': total_accuracy / num_batches}
