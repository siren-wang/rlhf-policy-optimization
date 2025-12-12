"""
Reward model implementation for RLHF.
Fine-tunes a pretrained model with a scalar reward head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Reward model with pretrained backbone and scalar reward head.
    Uses pairwise ranking loss for training.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        num_layers_unfrozen: int = -1,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_layers_unfrozen: Number of transformer layers to unfreeze (-1 = all)
            dropout: Dropout rate for reward head
        """
        super().__init__()
        
        logger.info(f"Initializing reward model from {model_name}")
        
        # Load pretrained backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size
        hidden_size = self.config.hidden_size
        
        # Reward head: projects hidden states to scalar reward
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Optionally freeze some layers
        if num_layers_unfrozen > 0:
            self._freeze_layers(num_layers_unfrozen)
        
        logger.info(f"Reward model initialized with {self.num_parameters():,} parameters")
    
    def _freeze_layers(self, num_unfrozen: int):
        """Freeze all but the last num_unfrozen transformer layers."""
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last num_unfrozen layers
        if hasattr(self.backbone, 'transformer'):
            layers = self.backbone.transformer.h  # GPT-2 style
        elif hasattr(self.backbone, 'encoder'):
            layers = self.backbone.encoder.layer  # BERT style
        else:
            logger.warning("Could not identify transformer layers, unfreezing all")
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        
        for layer in layers[-num_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Froze all but last {num_unfrozen} layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute reward.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            rewards: (batch,) scalar reward for each sequence
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # Get last token's hidden state (or use pooling)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Use last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = hidden_states.size(0)
        
        # Gather last token hidden states
        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]  # (batch, hidden)
        
        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        
        return rewards
    
    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        Loss = -log(sigmoid(r_chosen - r_rejected))
        
        Args:
            chosen_input_ids: (batch, seq_len)
            chosen_attention_mask: (batch, seq_len)
            rejected_input_ids: (batch, seq_len)
            rejected_attention_mask: (batch, seq_len)
        
        Returns:
            loss: scalar loss
        """
        # Compute rewards
        chosen_rewards = self.forward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.forward(rejected_input_ids, rejected_attention_mask)
        
        # Ranking loss: prefer chosen over rejected
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        # Compute accuracy (how often we rank correctly)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, accuracy, chosen_rewards.mean(), rejected_rewards.mean()
    
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model checkpoint."""
        logger.info(f"Saving reward model to {path}")
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
    
    @classmethod
    def load(cls, path: str, model_name: str = "gpt2") -> 'RewardModel':
        """Load model checkpoint."""
        logger.info(f"Loading reward model from {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        model = cls(model_name=model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
