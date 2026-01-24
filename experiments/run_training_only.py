"""
独立的预训练脚本 - 只执行上游模型训练
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from veccity.pipeline import run_model
from veccity.utils import get_executor, get_model, get_evaluator, get_logger, ensure_dir
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='xa', help='Dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=10, help='Max training epochs')
    parser.add_argument('--gpu', type=str, default='True', help='Use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()

def run_training_only(args):
    """只运行上游训练，保存模型和embedding"""
    print("=" * 80)
    print("Running HRNR_Hyperbolic TRAINING ONLY")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Max Epoch: {args.max_epoch}")
    print(f"GPU: {args.gpu} (ID: {args.gpu_id})")
    print("=" * 80)

    # 构建配置
    from veccity.utils import general_arguments
    from veccity.config import ConfigParser

    # 准备参数
    other_args = {
        'task': 'segment',
        'model': 'HRNR_Hyperbolic',
        'dataset': args.dataset,
        'saved_model': True,
        'train': True,
    }

    # 添加超参数
    cmd_args = general_arguments()
    for key, value in other_args.items():
        setattr(cmd_args, key, value)

    # 添加命令行参数
    setattr(cmd_args, 'seed', args.seed)
    setattr(cmd_args, 'max_epoch', args.max_epoch)
    setattr(cmd_args, 'gpu', args.gpu == 'True')
    setattr(cmd_args, 'gpu_id', args.gpu_id)

    # 解析配置
    config = ConfigParser(
        task=cmd_args.task,
        model=cmd_args.model,
        dataset=cmd_args.dataset,
        config_file=None,
        saved_model=cmd_args.saved_model,
        train=cmd_args.train,
        other_args=vars(cmd_args)
    )

    # 获取logger
    logger = get_logger(config)
    logger.info('TRAINING ONLY MODE - Skipping downstream evaluation')
    logger.info(config.config)

    # 加载数据
    from veccity.data import get_dataset
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    # 创建模型
    model = get_model(config, data_feature)

    # 创建executor（用于训练）
    executor = get_executor(config, model, data_feature)

    # 只执行训练
    logger.info("=" * 80)
    logger.info("Starting UPSTREAM TRAINING...")
    logger.info("=" * 80)
    executor.train(train_data, valid_data)

    # 保存模型
    logger.info("=" * 80)
    logger.info("Saving trained model and embeddings...")
    logger.info("=" * 80)
    executor.save_model(None)  # 使用默认路径保存

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: ./veccity/cache/{config['exp_id']}/model_cache/")
    logger.info(f"Embeddings saved to: ./veccity/cache/{config['exp_id']}/evaluate_cache/")
    logger.info("=" * 80)
    logger.info("Next step: Run 'python run_evaluation_only.py' to evaluate")
    logger.info("=" * 80)

if __name__ == '__main__':
    args = parse_args()
    run_training_only(args)
