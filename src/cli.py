"""Command-line interface for RLHF training."""
import argparse
import yaml
from train_reward import main as train_reward_main
from data import load_hh_rlhf_data, analyze_dataset, create_dummy_data
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='RLHF CLI')
    subparsers = parser.add_subparsers(dest='command')
    
    explore_parser = subparsers.add_parser('explore-data')
    explore_parser.add_argument('--config', required=True)
    
    reward_parser = subparsers.add_parser('train-reward')
    reward_parser.add_argument('--config', required=True)
    
    args = parser.parse_args()
    
    if args.command == 'explore-data':
        config = yaml.safe_load(open(args.config))
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        data = create_dummy_data(10) if config['data'].get('use_dummy_data') else load_hh_rlhf_data('train', 100)
        stats = analyze_dataset(data, tokenizer)
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif args.command == 'train-reward':
        train_reward_main(args.config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
