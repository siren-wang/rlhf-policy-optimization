# RLHF Policy Optimization

A comprehensive implementation and evaluation of three Reinforcement Learning from Human Feedback (RLHF) methods: Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO).

See 
[
ANALYSIS.md
](
ANALYSIS.md
)
 for comprehensive evaluation.

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.5+ (for GPU training)
- 32GB+ RAM
- 50GB+ disk space

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/siren-wang/rlhf-policy-optimization.git
cd rlhf-policy-optimization
```

2. **Create virtual environment**:
```bash
conda create -n rlhf python=3.10
conda activate rlhf
```

3. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install numpy pandas matplotlib seaborn jupyter
pip install pyyaml tqdm

# For GPT-4 evaluation (optional)
pip install openai
```

## Compute Requirements

### Minimum Requirements (CPU Only)

- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB
- **Training time**: ~20-40 hours for full pipeline

Suitable for: Demo runs, prototyping

### Recommended Requirements (GPU)

- **GPU**: 8GB VRAM (e.g., GTX 1080, RTX 2070)
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 100GB
- **Training time**: ~4-8 hours for full pipeline

Suitable for: Full training runs, experimentation

### Optimal Requirements (Multi-GPU)

- **GPU**: 4x 8GB VRAM or 1x 24GB VRAM
- **CPU**: 16+ cores
- **RAM**: 64GB
- **Storage**: 200GB
- **Training time**: ~2-4 hours for full pipeline

Suitable for: Large-scale experiments, hyperparameter tuning

### Memory Breakdown

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Base Model (GPT-2) | ~500MB | Loaded once |
| Reward Model | ~500MB | Loaded during training |
| Training Batch | 2-8GB | Depends on batch size |
| Generated Samples | ~1GB | For evaluation |

### GPU Memory Optimization

The project includes memory-optimized configurations:

- **Demo** (`configs/demo.yaml`): 10 samples, CPU-compatible
- **Prototype** (`configs/prototype.yaml`): 1K samples, requires 8GB GPU
- **Full** (`configs/full.yaml`): 10K samples, requires 8GB+ GPU

## Quick Start

### 1. Demo Run (5 minutes, CPU-compatible)

```bash
# Train all methods on 10 samples
bash scripts/run_small_demo.sh

# Results in: outputs/demo/
```

### 2. Prototype Run (1 hour, GPU recommended)

```bash
# Train on 1K samples
bash scripts/run_prototype.sh

# Results in: outputs/prototype/
```

### 3. Full Training (4-8 hours, GPU required)


```bash

# Train reward model
python -m src.cli train-reward configs/full.yaml

# Train all three methods
bash scripts/run_ppo.sh configs/full.yaml
bash scripts/run_grpo.sh configs/full.yaml
bash scripts/run_dpo.sh configs/full.yaml
# Results in: outputs/full/

```

## Training Pipeline

### Step 1: Train Reward Model

```bash
python -m src.cli train-reward configs/full.yaml
```

**Output**: `outputs/full/reward_model/best_model.pt`

**Time**: ~75 minutes (5 epochs on 10K samples)

### Step 2: Train Policy Models

**PPO (Proximal Policy Optimization)**:
```bash
bash scripts/run_ppo.sh configs/full.yaml
```
- **Time**: ~45-60 minutes
- **Output**: `outputs/full/ppo_model/`
- **Characteristics**: High exploration, aggressive optimization

**GRPO (Group Relative Policy Optimization)**:
```bash
bash scripts/run_grpo.sh configs/full.yaml
```
- **Time**: ~45-60 minutes
- **Output**: `outputs/full/grpo_model/`
- **Characteristics**: Stable, conservative, best win rate

**DPO (Direct Preference Optimization)**:
```bash
bash scripts/run_dpo.sh configs/full.yaml
```
- **Time**: ~30-45 minutes
- **Output**: `outputs/full/dpo_model/`
- **Characteristics**: Fast, efficient, good generalization

### Step 3: Generate Samples

```bash
bash scripts/generate_samples.sh
```

**Output**: `output_samples/` (100 samples per model with rewards and KL)

**Time**: ~45-60 minutes

### Step 4: Evaluation

**Win Rate Evaluation (GPT-4o-mini)**:
```bash
export OPENAI_API_KEY='your-key-here'
bash scripts/gpt4_judge.sh
```

**Failure Analysis**:
```bash
bash scripts/failure_analysis.sh
```

**Training Analysis**:
```bash
bash scripts/training_analysis.sh
```

## Evaluation

### Automated Evaluation

All evaluation scripts are in `scripts/`:

```bash
# 1. Generate samples with rewards and KL
bash scripts/generate_samples.sh

# 2. Compute win rates (GPT-4o-mini)
bash scripts/gpt4_judge.sh

# 3. Failure mode analysis
bash scripts/failure_analysis.sh

# 4. Training dynamics analysis
bash scripts/training_analysis.sh
```

### Results

Results are saved in:
- `output_samples/`: Generated text samples
- `results/`: Analysis reports and statistics
  - `gpt4_win_rates.json`: Win rate data
  - `WIN_RATES.md`: Win rate report
  - `failure_analysis/`: Failure mode analysis
  - `training_analysis/`: Training curves and comparisons

### Manual Analysis

Use the Jupyter notebook for custom analysis:

```bash
jupyter notebook notebooks/final_analysis.ipynb
```

The notebook provides:
- Reward distribution visualizations
- KL divergence analysis
- Win rate comparisons
- Custom metrics

## Project Structure

```
rlhf-policy-optimization/
├── src/                          # Source code
│   ├── reward_model.py           # Reward model implementation
│   ├── ppo.py                    # PPO trainer
│   ├── grpo.py                   # GRPO trainer
│   ├── dpo.py                    # DPO trainer
│   ├── data.py                   # Data loading utilities
│   ├── train_utils.py            # Training utilities
│   ├── evaluate.py               # Evaluation utilities
│   └── cli.py                    # Command-line interface
├── configs/                      # Configuration files
│   ├── demo.yaml                 # Demo config (10 samples)
│   ├── prototype.yaml            # Prototype config (1K samples)
│   └── full.yaml                 # Full config (10K samples)
├── scripts/                      # Training and evaluation scripts
│   ├── run_small_demo.sh         # Demo training
│   ├── run_ppo.sh                # PPO training
│   ├── run_grpo.sh               # GRPO training
│   ├── run_dpo.sh                # DPO training
│   ├── generate_samples.sh       # Sample generation
│   ├── gpt4_judge.sh             # Win rate evaluation
│   ├── failure_analysis.sh       # Failure mode analysis
│   └── training_analysis.sh      # Training analysis
├── notebooks/                    # Jupyter notebooks
│   └── final_analysis.ipynb      # Analysis notebook
├── output_samples/               # Generated samples (committed to Git)
│   ├── reference_samples.json
│   ├── ppo_samples.json
│   ├── grpo_samples.json
│   ├── dpo_samples.json
│   └── samples.md
├── results/                      # Analysis results (committed to Git)
│   ├── gpt4_win_rates.json
│   ├── WIN_RATES.md
│   ├── failure_analysis/
│   └── training_analysis/
├── outputs/                      # Model checkpoints (NOT on Git)
│   └── full/
│       ├── reward_model/
│       ├── ppo_model/
│       ├── grpo_model/
│       └── dpo_model/
├── ANALYSIS.md                   # Comprehensive analysis report
└── README.md                     # This file
```

## Configuration

### Config Files

Three pre-configured setups:

**Demo** (`configs/demo.yaml`):
- 10 training samples
- 1 epoch
- ~5 minutes total
- CPU-compatible

**Prototype** (`configs/prototype.yaml`):
- 1K training samples
- 3 epochs
- ~1 hour total
- 8GB GPU recommended

**Full** (`configs/full.yaml`):
- 10K training samples
- 5 epochs
- ~4-8 hours total
- 8GB+ GPU required

### Key Parameters

```yaml
training:
  batch_size: 2              # Adjust based on GPU memory
  max_length: 256            # Maximum sequence length
  learning_rate: 1e-6        # Learning rate
  num_epochs: 5              # Number of training epochs

ppo:
  batch_size: 2
  max_gen_length: 64         # Max tokens to generate
  kl_coef: 0.1               # KL penalty coefficient
  temperature: 1.0           # Sampling temperature

grpo:
  group_size: 2              # Number of generations per prompt
  max_gen_length: 64
  kl_coef: 0.1

dpo:
  batch_size: 2
  beta: 0.1                  # DPO temperature parameter
```

## Results

### Win Rates (GPT-4o-mini Judge, 100 samples)

| Model | Win Rate | Ties | Losses |
|-------|----------|------|--------|
| GRPO | 50.0% | 19% | 31% |
| DPO | 43.0% | 18% | 39% |
| PPO | 36.0% | 17% | 47% |

### Reward and KL Divergence

| Model | Mean Reward | Mean KL | Assessment |
|-------|-------------|---------|------------|
| Reference | 0.604 | 0.000 | Baseline |
| GRPO | 1.443 (+139%) | 0.536 | Best overall |
| PPO | 1.190 (+97%) | 1.814 | High drift |
| DPO | 0.795 (+32%) | 0.845 | Efficient |

### Training Time

| Method | Time/Epoch | Total Time | Efficiency |
|--------|------------|------------|------------|
| DPO | 60s | 3 hours | Best |
| PPO | 72s | 6 hours | Medium |
| GRPO | 75s | 6.25 hours | Lower |

See [ANALYSIS.md](ANALYSIS.md) for detailed analysis.

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce `batch_size` in config (try 1)
2. Reduce `max_length` in config (try 128)
3. Reduce `max_gen_length` (try 32)
4. Use CPU: `CUDA_VISIBLE_DEVICES="" python ...`

### Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solution**: Ensure you're running from project root and PYTHONPATH is set:
```bash
cd /path/to/rlhf-policy-optimization
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
1. Use GPU instead of CPU
2. Reduce dataset size (use `demo.yaml` or `prototype.yaml`)
3. Reduce `num_epochs`
4. Increase `batch_size` if memory allows

### CUDA Version Mismatch

**Symptoms**: `CUDA driver version is insufficient`

**Solution**: Install PyTorch matching your CUDA version:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```


## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact:
- GitHub: [@siren-wang](https://github.com/siren-wang)
- Repository: [rlhf-policy-optimization](https://github.com/siren-wang/rlhf-policy-optimization)

## Acknowledgments

- Anthropic for the HH-RLHF dataset
- Hugging Face for Transformers library
- OpenAI for GPT-4o-mini evaluation API