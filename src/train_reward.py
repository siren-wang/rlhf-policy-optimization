"""
Training script for reward model.
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

from data import prepare_datasets
from reward_model import RewardModel
from train_utils import setup_logging, save_config, log_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(
    model: RewardModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_chosen_reward = 0
    total_rejected_reward = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)
        
        # Forward pass
        loss, accuracy, chosen_reward, rejected_reward = model.compute_pairwise_loss(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_chosen_reward += chosen_reward.item()
        total_rejected_reward += rejected_reward.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy.item():.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'chosen_reward': total_chosen_reward / num_batches,
        'rejected_reward': total_rejected_reward / num_batches,
    }


def evaluate(
    model: RewardModel,
    dataloader: DataLoader,
    device: str
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_chosen_reward = 0
    total_rejected_reward = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            chosen_input_ids = batch['chosen_input_ids'].to(device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device)
            
            loss, accuracy, chosen_reward, rejected_reward = model.compute_pairwise_loss(
                chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask
            )
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_chosen_reward += chosen_reward.item()
            total_rejected_reward += rejected_reward.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'chosen_reward': total_chosen_reward / num_batches,
        'rejected_reward': total_rejected_reward / num_batches,
    }


def main(config_path: str):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    output_dir = Path(config['output_dir']) / 'reward_model'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir / 'train.log')
    save_config(config, output_dir / 'config.yaml')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config['data'], tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Use 0 for Docker compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = RewardModel(
        model_name=config['model_name'],
        num_layers_unfrozen=config.get('num_layers_unfrozen', -1),
        dropout=config.get('dropout', 0.1)
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Training loop
    best_val_accuracy = 0
    metrics_history = []
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                   f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
        }
        metrics_history.append(metrics)
        log_metrics(metrics, output_dir / 'metrics.jsonl')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            model.save(output_dir / 'best_model.pt')
            logger.info(f"Saved best model (accuracy: {best_val_accuracy:.4f})")
        
        # Save checkpoint
        if epoch % config['training'].get('save_every', 1) == 0:
            model.save(output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Model saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
