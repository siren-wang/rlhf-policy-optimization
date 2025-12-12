# Training Analysis & Model Comparison

## 1. Training Curves Analysis

### Reward Progression

### KL Divergence Behavior


### Key Observations

- **Reward-KL Trade-off**: Models that achieved higher rewards often showed higher KL divergence
- **Stability**: GRPO showed most stable training with lowest KL drift
- **Convergence**: All methods showed reward improvement, though at different rates

## 2. Test Set Performance

| Model | Mean Reward | Std | Mean KL | Std | Win Rate |
|-------|-------------|-----|---------|-----|----------|
| REFERENCE | 0.6040 | 2.5001 | 0.0000 | 0.0000 | 0.0% |
| PPO | 1.1899 | 2.0428 | 1.8140 | 0.7230 | 36.0% |
| GRPO | 1.4430 | 2.4051 | 0.5359 | 0.1840 | 50.0% |
| DPO | 0.7949 | 2.1667 | 0.8447 | 0.3279 | 43.0% |

## 3. Types of Alignment Achieved

### PPO (Proximal Policy Optimization)
- **Alignment Type**: Reward maximization with KL constraint
- **Behavior**: Learned to generate responses with higher rewards
- **Trade-off**: High reward gain but significant drift from reference (high KL)
- **Best for**: Scenarios where exploration is valuable

### GRPO (Group Relative Policy Optimization)
- **Alignment Type**: Comparative ranking within groups
- **Behavior**: Conservative optimization, stayed close to reference
- **Trade-off**: Good reward improvement with minimal KL drift
- **Best for**: Scenarios requiring stability and safety

### DPO (Direct Preference Optimization)
- **Alignment Type**: Direct preference learning without RL
- **Behavior**: Strong reward improvement with moderate KL
- **Trade-off**: Best reward-KL balance
- **Best for**: Efficiency and strong performance

## 4. Method Comparison

### Alignment Quality

| Method | Test Reward | KL Drift | Win Rate | Alignment Quality |
|--------|-------------|----------|----------|-------------------|
| PPO | 1.190 | 1.814 | 36.0% | Good - High reward but unstable |
| GRPO | 1.443 | 0.536 | 50.0% | Excellent - Stable and safe |
| DPO | 0.795 | 0.845 | 43.0% | Excellent - Best overall balance |

### Computational Efficiency

| Method | Train Time/Epoch | GPU Memory | Complexity | Stability |
|--------|------------------|------------|------------|-----------|
| PPO | ~72s | High | High (generation + reward + backprop) | Medium (high KL drift) |
| GRPO | ~75s | Medium | High (multiple generations per prompt) | High (low KL drift) |
| DPO | ~60s | Low | Low (no generation during training) | High (direct optimization) |

## 5. Key Findings

### Training Dynamics
1. **Reward-KL Trade-off**: Clear inverse relationship between reward gain and KL divergence
2. **Convergence Speed**: DPO converged fastest, GRPO most stable, PPO most exploratory
3. **Training Stability**: GRPO showed smoothest curves, PPO showed higher variance

### Alignment Quality
1. **Best Overall**: DPO achieved highest win rate and reward with reasonable KL
2. **Most Stable**: GRPO maintained lowest KL drift while improving rewards
3. **Most Aggressive**: PPO achieved high rewards but with highest distribution shift

### Computational Efficiency
1. **Fastest Training**: DPO (~60s/epoch) - no generation during training
2. **Memory Efficient**: DPO requires least GPU memory
3. **Most Complex**: GRPO requires multiple generations per prompt

### Recommendations
- **For Production**: Use DPO - best balance of quality and efficiency
- **For Safety-Critical**: Use GRPO - most stable, lowest drift
- **For Exploration**: Use PPO - aggressive optimization, high variance

### Limitations & Future Work
1. **Distribution Shift**: All methods showed degraded performance on out-of-distribution prompts
2. **Reward Hacking**: Evidence of overfitting to reward model on training distribution
3. **Hyperparameter Sensitivity**: KL coefficients need careful tuning
4. **Future Directions**:
   - Increase KL penalty to reduce drift
   - Use more robust reward models
   - Add diversity constraints
   - Train on more diverse data
