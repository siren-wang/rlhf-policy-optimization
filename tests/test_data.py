"""Unit tests for data loading."""
import pytest
import torch
from transformers import AutoTokenizer
from src.data import create_dummy_data, split_data, PreferenceDataset, PromptDataset, analyze_dataset

def test_create_dummy_data():
    data = create_dummy_data(num_samples=10)
    assert len(data) == 10
    assert all('prompt' in item for item in data)
    assert all('chosen' in item for item in data)
    assert all('rejected' in item for item in data)

def test_split_data():
    data = create_dummy_data(num_samples=100)
    train, val = split_data(data, train_ratio=0.8)
    assert len(train) == 80
    assert len(val) == 20

def test_preference_dataset():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    data = create_dummy_data(num_samples=5)
    dataset = PreferenceDataset(data, tokenizer, max_length=128)
    assert len(dataset) == 5
    item = dataset[0]
    assert 'chosen_input_ids' in item
    assert item['chosen_input_ids'].shape == torch.Size([128])

def test_prompt_dataset():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    prompts = ["Test 1", "Test 2"]
    dataset = PromptDataset(prompts, tokenizer, max_length=64)
    assert len(dataset) == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
