"""
Training utilities and helper functions.
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict
import torch


def setup_logging(log_file: Path = None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def save_config(config: Dict, path: Path):
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Saved config to {path}")


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def log_metrics(metrics: Dict, path: Path):
    """Append metrics to JSONL file."""
    with open(path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
