"""Unit tests for loss functions."""
import pytest
import torch
from src.reward_model import RewardModel

def test_reward_model_forward():
    model = RewardModel(model_name='distilgpt2')
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    rewards = model(input_ids, attention_mask)
    assert rewards.shape == torch.Size([2])

def test_reward_model_pairwise_loss():
    model = RewardModel(model_name='distilgpt2')
    chosen_ids = torch.randint(0, 1000, (2, 10))
    chosen_mask = torch.ones(2, 10)
    rejected_ids = torch.randint(0, 1000, (2, 10))
    rejected_mask = torch.ones(2, 10)
    loss, accuracy, chosen_reward, rejected_reward = model.compute_pairwise_loss(
        chosen_ids, chosen_mask, rejected_ids, rejected_mask
    )
    assert isinstance(loss.item(), float)
    assert 0 <= accuracy.item() <= 1

def test_reward_model_save_load(tmp_path):
    model = RewardModel(model_name='distilgpt2')
    save_path = tmp_path / "model.pt"
    model.save(str(save_path))
    loaded_model = RewardModel.load(str(save_path), model_name='distilgpt2')
    input_ids = torch.randint(0, 1000, (1, 10))
    attention_mask = torch.ones(1, 10)
    reward = loaded_model(input_ids, attention_mask)
    assert reward.shape == torch.Size([1])

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
