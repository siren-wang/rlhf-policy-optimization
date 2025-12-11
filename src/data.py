"""
Data loading and preprocessing for RLHF.
Supports Anthropic HH-RLHF format and custom JSONL files.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for preference pairs (chosen vs rejected responses)."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            split: 'train' or 'validation'
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns tokenized prompt, chosen, and rejected responses."""
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Tokenize prompt + chosen
        chosen_text = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize prompt + rejected
        rejected_text = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0),
            'prompt': prompt,
        }


class PromptDataset(Dataset):
    """Dataset for prompts only (for policy training and generation)."""
    
    def __init__(
        self,
        prompts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'prompt': prompt,
        }


def load_hh_rlhf_data(
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Load Anthropic HH-RLHF dataset from HuggingFace.
    
    Args:
        split: 'train' or 'test'
        num_samples: Limit number of samples (for prototyping)
        cache_dir: Cache directory for dataset
    
    Returns:
        List of dicts with 'prompt', 'chosen', 'rejected' keys
    """
    logger.info(f"Loading HH-RLHF {split} split...")
    
    try:
        # Load from HuggingFace
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            split=split,
            cache_dir=cache_dir
        )
        
        data = []
        for item in dataset:
            # HH-RLHF format: 'chosen' and 'rejected' are full conversations
            # Extract prompt (everything before last assistant response)
            chosen = item['chosen']
            rejected = item['rejected']
            
            # Simple prompt extraction (take everything up to last "Assistant:")
            if '\n\nAssistant:' in chosen:
                parts = chosen.split('\n\nAssistant:')
                prompt = '\n\nAssistant:'.join(parts[:-1]) + '\n\nAssistant:'
                chosen_response = parts[-1]
            else:
                prompt = ""
                chosen_response = chosen
            
            if '\n\nAssistant:' in rejected:
                rejected_response = rejected.split('\n\nAssistant:')[-1]
            else:
                rejected_response = rejected
            
            data.append({
                'prompt': prompt,
                'chosen': chosen_response,
                'rejected': rejected_response
            })
            
            if num_samples and len(data) >= num_samples:
                break
        
        logger.info(f"Loaded {len(data)} samples from HH-RLHF {split}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load HH-RLHF: {e}")
        logger.info("Falling back to local JSONL file...")
        return load_local_jsonl(f"data/{split}.jsonl", num_samples)


def load_local_jsonl(
    filepath: str,
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load preference data from local JSONL file.
    Expected format: {'prompt': str, 'chosen': str, 'rejected': str}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File {filepath} not found. Creating dummy data...")
        return create_dummy_data(num_samples or 10)
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if num_samples and len(data) >= num_samples:
                break
    
    logger.info(f"Loaded {len(data)} samples from {filepath}")
    return data


def create_dummy_data(num_samples: int = 10) -> List[Dict]:
    """Create dummy preference data for testing."""
    logger.info(f"Creating {num_samples} dummy samples...")
    
    prompts = [
        "Human: What is the capital of France?\n\nAssistant:",
        "Human: How do I bake a cake?\n\nAssistant:",
        "Human: Explain quantum computing.\n\nAssistant:",
        "Human: What's the weather like today?\n\nAssistant:",
        "Human: How do I learn Python?\n\nAssistant:",
    ]
    
    chosen_responses = [
        " The capital of France is Paris, a beautiful city known for its art and culture.",
        " To bake a cake, preheat your oven to 350°F, mix ingredients, and bake for 30 minutes.",
        " Quantum computing uses quantum bits (qubits) that can exist in superposition states.",
        " I don't have access to real-time weather data, but you can check weather.com.",
        " Start with online tutorials, practice coding daily, and build small projects.",
    ]
    
    rejected_responses = [
        " I don't know.",
        " Just buy a cake from the store.",
        " It's complicated.",
        " It's probably sunny.",
        " Python is hard.",
    ]
    
    data = []
    for i in range(num_samples):
        idx = i % len(prompts)
        data.append({
            'prompt': prompts[idx],
            'chosen': chosen_responses[idx],
            'rejected': rejected_responses[idx]
        })
    
    return data


def split_data(
    data: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def analyze_dataset(data: List[Dict], tokenizer: AutoTokenizer) -> Dict:
    """Analyze dataset statistics and patterns."""
    logger.info("Analyzing dataset...")
    
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    chosen_longer = 0
    
    for item in data:
        prompt_tokens = tokenizer.encode(item['prompt'])
        chosen_tokens = tokenizer.encode(item['chosen'])
        rejected_tokens = tokenizer.encode(item['rejected'])
        
        prompt_lengths.append(len(prompt_tokens))
        chosen_lengths.append(len(chosen_tokens))
        rejected_lengths.append(len(rejected_tokens))
        
        if len(chosen_tokens) > len(rejected_tokens):
            chosen_longer += 1
    
    stats = {
        'num_samples': len(data),
        'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths),
        'avg_chosen_length': sum(chosen_lengths) / len(chosen_lengths),
        'avg_rejected_length': sum(rejected_lengths) / len(rejected_lengths),
        'pct_chosen_longer': 100 * chosen_longer / len(data),
    }
    
    logger.info(f"Dataset statistics: {stats}")
    return stats


def prepare_datasets(
    config: Dict,
    tokenizer: AutoTokenizer
) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """
    Prepare train and validation datasets.
    
    Args:
        config: Configuration dict with data settings
        tokenizer: Tokenizer to use
    
    Returns:
        train_dataset, val_dataset
    """
    # Load data
    if config.get('use_dummy_data', False):
        data = create_dummy_data(config.get('num_samples', 10))
        train_data, val_data = split_data(data, config.get('train_ratio', 0.8))
    else:
        train_data = load_hh_rlhf_data(
            split='train',
            num_samples=config.get('num_train_samples'),
            cache_dir=config.get('cache_dir')
        )
        val_data = load_hh_rlhf_data(
            split='test',
            num_samples=config.get('num_val_samples'),
            cache_dir=config.get('cache_dir')
        )
    
    # Analyze
    if config.get('analyze_data', True):
        analyze_dataset(train_data, tokenizer)
    
    # Create datasets
    train_dataset = PreferenceDataset(
        train_data,
        tokenizer,
        max_length=config.get('max_length', 512),
        split='train'
    )
    
    val_dataset = PreferenceDataset(
        val_data,
        tokenizer,
        max_length=config.get('max_length', 512),
        split='validation'
    )
    
    return train_dataset, val_dataset
