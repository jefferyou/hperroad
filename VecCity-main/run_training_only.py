"""
独立运行模型训练脚本（不运行下游任务）
用于训练模型并保存embeddings，但跳过下游任务评估

使用方法:
    python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge

参数说明:
    --task: 任务类型（如 segment, region, poi）
    --model: 模型名称
    --dataset: 数据集名称
    --config_file: 配置文件路径（可选）
    --device: 设备（cpu或cuda，默认cpu）
    --seed: 随机种子（默认31）
    --exp_id: 实验ID（可选，默认自动生成）
"""

import os
import sys
import argparse
import random
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from veccity.config import ConfigParser
from veccity.data import get_dataset
from veccity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run model training only (no downstream tasks)')

    # 必需参数
    parser.add_argument('--task', type=str, required=True,
                        help='Task name (e.g., segment, region, poi)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., HRNR_Hyperbolic, HRNR)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., bj_roadmap_edge)')

    # 可选参数
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda, default: cpu)')
    parser.add_argument('--seed', type=int, default=31,
                        help='Random seed (default: 31)')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (default: auto-generate)')
    parser.add_argument('--saved_model', action='store_true',
                        help='Save trained model checkpoint')

    return parser.parse_args()


def run_training_only(args):
    """运行模型训练（不运行下游任务）"""

    # 设置随机种子
    set_random_seed(args.seed)

    # 生成或使用指定的exp_id
    exp_id = args.exp_id if args.exp_id else int(random.SystemRandom().random() * 100000)

    # 构建other_args字典
    other_args = {
        'task': args.task,
        'model': args.model,
        'dataset': args.dataset,
        'device': args.device,
        'seed': args.seed,
        'exp_id': exp_id,
    }

    # 加载配置
    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=True,
        other_args=other_args
    )

    # 创建logger
    logger = get_logger(config)
    logger.info('='*80)
    logger.info('Running TRAINING ONLY (no downstream evaluation)')
    logger.info(f'Task: {args.task}, Model: {args.model}, Dataset: {args.dataset}')
    logger.info(f'Experiment ID: {exp_id}')
    logger.info(f'Device: {args.device}')
    logger.info(f'Seed: {args.seed}')
    logger.info('='*80)

    # 加载数据集
    logger.info("Loading dataset...")
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    logger.info(f"[OK] Dataset loaded")

    # 创建模型
    logger.info("Creating model...")
    model = get_model(config, data_feature)
    total_num = sum([param.nelement() for param in model.parameters()])
    logger.info(f'[OK] Model created: {total_num:,} parameters')

    # 创建executor
    logger.info("Creating executor...")
    executor = get_executor(config, model, data_feature)
    logger.info(f"[OK] Executor created: {type(executor).__name__}")

    # 创建缓存目录
    model_cache_file = f'./veccity/cache/{exp_id}/model_cache/{args.model}_{args.dataset}.m'
    ensure_dir(os.path.dirname(model_cache_file))

    # 运行训练
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    try:
        executor.train(train_data, valid_data)

        logger.info("="*80)
        logger.info("[OK] Training completed successfully!")
        logger.info("="*80)

        # 保存模型
        if args.saved_model:
            logger.info(f"Saving model to {model_cache_file}...")
            executor.save_model(model_cache_file)
            logger.info(f"[OK] Model saved")

        # 检查embeddings是否已生成
        embedding_path = f'./veccity/cache/{exp_id}/evaluate_cache/road_embedding_{args.model}_{args.dataset}_*.npy'
        logger.info("="*80)
        logger.info("📊 Training Summary:")
        logger.info(f"  - Experiment ID: {exp_id}")
        logger.info(f"  - Model checkpoint: {model_cache_file if args.saved_model else 'Not saved'}")
        logger.info(f"  - Embeddings pattern: {embedding_path}")
        logger.info("")
        logger.info("To run downstream tasks, use:")
        logger.info(f"  python run_downstream_only.py --task {args.task} --model {args.model} --dataset {args.dataset} --exp_id {exp_id}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"[ERROR] Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*80)
    print("*** Model Training Script (Training Only, No Evaluation)")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Save Model: {args.saved_model}")
    print("="*80 + "\n")

    run_training_only(args)


if __name__ == '__main__':
    main()
