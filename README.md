# RLHF Assignment 3: Reinforcement Learning from Human Feedback

Complete implementation of RLHF, DPO, PPO, and GRPO for language model alignment.

## Overview

This project implements three approaches to align language models with human preferences:
1. **PPO-based RLHF**: Traditional reward model + PPO policy optimization
2. **GRPO**: Group Relative Policy Optimization (simplified PPO alternative)
3. **DPO**: Direct Preference Optimization (no explicit reward model)

## Repository Structure

```
.
├── Dockerfile                  # Container setup
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── ANALYSIS.md                 # Final analysis template
├── src/
│   ├── data.py                # Data loading and preprocessing
│   ├── reward_model.py        # Reward model implementation
│   ├── train_reward.py        # Reward model training
│   ├── ppo.py                 # PPO implementation
│   ├── grpo.py                # GRPO implementation
│   ├── dpo.py                 # DPO implementation
│   ├── evaluate.py            # Evaluation metrics
│   ├── train_utils.py         # Training utilities
│   └── cli.py                 # Command-line interface
├── scripts/
│   ├── run_small_demo.sh      # Quick end-to-end demo
│   ├── run_ppo.sh             # Run PPO training
│   ├── run_grpo.sh            # Run GRPO training
│   └── run_dpo.sh             # Run DPO training
├── configs/
│   ├── demo.yaml              # Demo configuration
│   ├── prototype.yaml         # Prototype configuration
│   └── full.yaml              # Full-scale configuration
├── tests/
│   ├── test_data.py           # Data pipeline tests
│   └── test_loss.py           # Loss function tests
└── notebooks/
    └── final_analysis.ipynb   # Analysis notebook
```

## Quick Start

### 1. Build Docker Container

```bash
docker build -t rlhf-policy-optimization .
```

### 2. Run Container

```bash
docker run -it --gpus all -v $(pwd):/app rlhf-policy-optimization
```

**Note**: Remove `--gpus all` if running on CPU only.

### 3. Run Small Demo (10 samples, ~5 minutes)

```bash
bash scripts/run_small_demo.sh
```

This will:
- Download a tiny subset of HH-RLHF data
- Train a reward model (1 epoch)
- Train PPO, GRPO, and DPO models (1 epoch each)
- Generate samples and evaluation metrics

## Step-by-Step Usage

### Part 1: Prepare Data and Train Reward Model

```bash
# Explore dataset (optional)
python src/cli.py explore-data --config configs/prototype.yaml

# Train reward model
python src/train_reward.py --config configs/prototype.yaml
```

### Part 2: Train Policy Models

```bash
# Train with PPO
bash scripts/run_ppo.sh configs/prototype.yaml

# Train with GRPO
bash scripts/run_grpo.sh configs/prototype.yaml

# Train with DPO
bash scripts/run_dpo.sh configs/prototype.yaml
```

### Part 3: Evaluate Models

```bash
python src/evaluate.py \
    --reward-model outputs/checkpoints/reward_model \
    --reference-model gpt2 \
    --ppo-model outputs/checkpoints/ppo_model \
    --grpo-model outputs/checkpoints/grpo_model \
    --dpo-model outputs/checkpoints/dpo_model \
    --num-samples 100 \
    --output-dir outputs/evaluation
```

### Part 4: Generate Analysis

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/final_analysis.ipynb
```

## Configuration Files

### Demo Config (`configs/demo.yaml`)
- 10 training samples
- 1 epoch
- Fast prototyping
- ~5 minutes total

### Prototype Config (`configs/prototype.yaml`)
- 1000 training samples
- 2-3 epochs
- Parameter tuning
- ~30-60 minutes

### Full Config (`configs/full.yaml`)
- Full dataset (10k+ samples)
- 5+ epochs
- Final submission quality
- Several hours (use GPU)

## Compute Requirements

### Minimum (CPU Demo)
- 8GB RAM
- 2 CPU cores
- ~5GB disk space
- Time: ~5 minutes

### Recommended (Prototype)
- 16GB RAM
- GPU with 8GB+ VRAM (e.g., RTX 2080)
- ~10GB disk space
- Time: ~30-60 minutes

### Full Run
- 32GB RAM
- GPU with 16GB+ VRAM (e.g., A100, V100)
- ~20GB disk space
- Time: 4-8 hours

## Cost Optimization

**Important**: This assignment uses small models (GPT-2 124M) and avoids expensive API calls:

- All training is local (no OpenAI/Anthropic API calls during training)
- GPT-judge evaluation uses local models by default
- Optional: Set `--use-gpt4-judge` for final evaluation only (~$2-5 for 100 samples)

## Hyperparameters

### Reward Model
```yaml
learning_rate: 1e-5
batch_size: 4
num_epochs: 3
max_length: 512
```

### PPO
```yaml
learning_rate: 1e-6
batch_size: 2
ppo_epochs: 4
clip_ratio: 0.2
kl_coef: 0.1
entropy_coef: 0.01
```

### GRPO
```yaml
learning_rate: 1e-6
batch_size: 2
group_size: 4
kl_coef: 0.1
```

### DPO
```yaml
learning_rate: 5e-7
batch_size: 4
beta: 0.1
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `max_length`
- Use gradient accumulation: set `gradient_accumulation_steps: 4`

### Slow Training
- Use GPU if available
- Reduce dataset size for prototyping
- Enable mixed precision: `--fp16`

### Dataset Download Issues
- The code will auto-download HH-RLHF from Hugging Face
- Fallback: place your own JSONL file in `data/train.jsonl`


## Output Files

After training, you'll find:

- `outputs/checkpoints/`: Model checkpoints
- `outputs/logs/`: TensorBoard logs
- `outputs/samples/`: Generated text samples
- `outputs/evaluation/`: Metrics and plots

## Citation

```bibtex
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and others},
  journal={arXiv preprint arXiv:2203.02155},
  year={2022}
}
```

## License

MIT License - Educational purposes only.
