# main.py
import argparse
import yaml
from train import train_model, evaluate_model
from utils import load_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run a deep learning model")
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to the config file')
    parser.add_argument('--batch_size', type=int, help='Override batch size from config', required=False)
    parser.add_argument('--epochs', type=int, help='Override number of epochs from config', required=False)
    parser.add_argument('--learning_rate', type=float, help='Override learning rate from config', required=False)
    parser.add_argument('--weight_decay', type=float, help='Override weight decay from config', required=False)
    parser.add_argument('--criterion', type=str, help='Override loss function from config', required=False)
    parser.add_argument('--optimizer', type=str, help='Override optimizer from config', required=False)
    parser.add_argument('--model_name', type=str, help='Override model name from config', required=False)
    parser.add_argument('--model_ver', type=int, default=0,help='Override model version from config', required=False)
    parser.add_argument('--downsampling_rate', type=int, help='Override downsampling rate from config', required=False)
    parser.add_argument('--num_classes', type=int, help='Override number of classes from config', required=False)
    parser.add_argument('--dataset_name', type=str, help='Override dataset name from config', required=False)
    parser.add_argument('--base_path', type=str, help='Override base path from config', required=False)
    parser.add_argument('--mask_num', type=int, help='Override mask number from config', required=False)
    parser.add_argument('--subject_agg', type=str, help='Override subject aggregation from config', required=False)
    parser.add_argument('--norm', type=str, help='Override norm setting from config', required=False)
    parser.add_argument('--norm_type', type=str, help='Override norm type from config', required=False)
    parser.add_argument('--stft_type', type=str, help='Override STFT type from config', required=False)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    # Override train section
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['train']['weight_decay'] = args.weight_decay
    if args.criterion:
        config['train']['criterion'] = args.criterion
    if args.optimizer:
        config['train']['optimizer'] = args.optimizer

    # Override model section
    if args.model_name:
        config['model']['model_name'] = args.model_name
    if args.model_ver:
        config['model']['model_ver'] = int(args.model_ver)
    if args.downsampling_rate:
        config['model']['downsampling_rate'] = args.downsampling_rate
    if args.num_classes:
        config['model']['num_classes'] = args.num_classes

    # Override data section
    if args.dataset_name:
        config['data']['dataset_name'] = args.dataset_name
    if args.base_path:
        config['data']['base_path'] = args.base_path
    if args.mask_num is not None:
        config['data']['mask_num'] = args.mask_num
    if args.subject_agg:
        config['data']['subject_agg'] = args.subject_agg
    if args.norm:
        config['data']['norm'] = args.norm
    if args.norm_type:
        config['data']['norm_type'] = args.norm_type
    if args.stft_type:
        config['data']['stft_type'] = args.stft_type

    return config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)
    
    train_loader, valid_loader, test_loader = load_data(config['data'], config['model'])
    
    model = train_model(config , train_loader, valid_loader)
    evaluate_model(model, test_loader)
    