#!/bin/bash
# Quick end-to-end demo (~5 minutes)

set -e

echo "========================================"
echo "RLHF Assignment 3 - Small Demo"
echo "========================================"

mkdir -p outputs/demo/{checkpoints,logs,samples}

echo ""
echo "[1/4] Training reward model..."
python src/train_reward.py --config configs/demo.yaml

echo ""
echo "[2/4] Training PPO model..."
bash scripts/run_ppo.sh configs/demo.yaml

echo ""
echo "[3/4] Training GRPO model..."
bash scripts/run_grpo.sh configs/demo.yaml

echo ""
echo "[4/4] Training DPO model..."
bash scripts/run_dpo.sh configs/demo.yaml

echo ""
echo "========================================"
echo "Demo complete!"
echo "========================================"
echo "Check outputs/demo/ for results"
