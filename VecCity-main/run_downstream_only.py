"""
独立运行下游任务评估脚本
用于在已有预训练embeddings的情况下，单独运行下游任务评估

使用方法:
    python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge

参数说明:
    --task: 任务类型（如 segment, region, poi）
    --model: 模型名称
    --dataset: 数据集名称
    --exp_id: 实验ID（用于查找embeddings文件）
    --output_dim: embedding维度（默认128）
    --config_file: 配置文件路径（可选）
    --device: 设备（cpu或cuda，默认cpu）
"""

import os
import sys
import argparse
import numpy as np
from logging import getLogger

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from veccity.config import ConfigParser
from veccity.data import get_dataset
from veccity.utils import get_evaluator, get_logger, ensure_dir, set_random_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run downstream tasks only')

    # 必需参数
    parser.add_argument('--task', type=str, required=True,
                        help='Task name (e.g., segment, region, poi)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., HRNR_Hyperbolic, HRNR)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., bj_roadmap_edge)')

    # 可选参数
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (default: auto-detect from embedding files)')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda, gpu, or cuda:0, default: cpu)')
    parser.add_argument('--seed', type=int, default=31,
                        help='Random seed (default: 31)')

    # 下游任务配置
    parser.add_argument('--evaluate_task', type=str, nargs='+',
                        default=["speed_inference", "travel_time_estimation", "similarity_search"],
                        help='Downstream tasks to evaluate (speed_inference, travel_time_estimation, similarity_search)')

    return parser.parse_args()


def find_embedding_file(base_path, model, dataset, output_dim):
    """
    自动查找embedding文件

    Args:
        base_path: 基础路径（./veccity/cache/）
        model: 模型名称
        dataset: 数据集名称
        output_dim: embedding维度

    Returns:
        embedding_path: embedding文件路径
        exp_id: 实验ID
    """
    # 遍历cache目录下的所有实验ID
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Cache directory not found: {base_path}")

    exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for exp_id in exp_dirs:
        embedding_path = os.path.join(
            base_path, exp_id, 'evaluate_cache',
            f'road_embedding_{model}_{dataset}_{output_dim}.npy'
        )
        if os.path.exists(embedding_path):
            return embedding_path, exp_id

    raise FileNotFoundError(
        f"No embedding file found for model={model}, dataset={dataset}, output_dim={output_dim}\n"
        f"Expected pattern: ./veccity/cache/{{exp_id}}/evaluate_cache/road_embedding_{model}_{dataset}_{output_dim}.npy"
    )


def run_downstream_evaluation(args):
    """运行下游任务评估"""

    # 设置随机种子
    set_random_seed(args.seed)

    # 处理设备配置：将device参数映射到gpu和gpu_id
    if args.device.lower() in ['gpu', 'cuda']:
        use_gpu = True
        gpu_id = 0
        device_name = 'cuda'
    elif args.device.lower().startswith('cuda:'):
        use_gpu = True
        gpu_id = int(args.device.split(':')[1])
        device_name = args.device
    else:
        use_gpu = False
        gpu_id = 0
        device_name = 'cpu'

    # 构建other_args字典
    other_args = {
        'task': args.task,
        'model': args.model,
        'dataset': args.dataset,
        'gpu': use_gpu,           # ConfigParser使用这个
        'gpu_id': gpu_id,         # ConfigParser使用这个
        'device': device_name,    # 下游任务直接使用这个
        'output_dim': args.output_dim,
        'evaluate_task': args.evaluate_task,
        'seed': args.seed,
    }

    # 如果指定了exp_id，使用指定的；否则自动查找
    if args.exp_id:
        exp_id = args.exp_id
        embedding_path = f'./veccity/cache/{exp_id}/evaluate_cache/road_embedding_{args.model}_{args.dataset}_{args.output_dim}.npy'

        if not os.path.exists(embedding_path):
            print(f"[ERROR] Embedding file not found: {embedding_path}")
            print(f"Looking for existing embeddings...")
            embedding_path, exp_id = find_embedding_file(
                './veccity/cache/', args.model, args.dataset, args.output_dim
            )
            print(f"[OK] Found embedding: {embedding_path}")
    else:
        print(f"[Search] Auto-detecting embedding file...")
        embedding_path, exp_id = find_embedding_file(
            './veccity/cache/', args.model, args.dataset, args.output_dim
        )
        print(f"[OK] Found embedding: {embedding_path}")

    other_args['exp_id'] = exp_id

    # 加载配置
    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=False,
        train=False,
        other_args=other_args
    )

    # 创建logger
    logger = get_logger(config)
    logger.info('='*80)
    logger.info('Running DOWNSTREAM TASKS ONLY (no training)')
    logger.info(f'Task: {args.task}, Model: {args.model}, Dataset: {args.dataset}')
    logger.info(f'Experiment ID: {exp_id}')
    logger.info(f'Embedding file: {embedding_path}')
    logger.info(f'Device: {args.device}')
    logger.info('='*80)

    # 检查embedding文件是否存在
    if not os.path.exists(embedding_path):
        logger.error(f"Embedding file not found: {embedding_path}")
        logger.error(f"Please run training first to generate embeddings.")
        sys.exit(1)

    # 加载embedding文件
    logger.info(f"Loading embeddings from {embedding_path}...")
    embeddings = np.load(embedding_path)
    logger.info(f"[OK] Embeddings loaded: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # 加载数据集（仅用于获取标签数据）
    logger.info("Loading dataset for labels...")
    dataset = get_dataset(config)
    data_feature = dataset.get_data_feature()
    logger.info(f"[OK] Dataset loaded")

    # 创建评估器
    logger.info("Creating evaluator...")
    evaluator = get_evaluator(config, data_feature)
    logger.info(f"[OK] Evaluator created: {type(evaluator).__name__}")

    # 确保输出目录存在
    evaluate_res_dir = f'./veccity/cache/{exp_id}/evaluate_cache'
    ensure_dir(evaluate_res_dir)

    # 运行下游任务评估
    logger.info("="*80)
    logger.info("Starting downstream task evaluation...")
    logger.info("="*80)

    try:
        # 对于RoadRepresentationEvaluator，直接调用evaluate()
        # 它会自动加载embeddings并运行所有下游任务
        evaluator.evaluate()

        logger.info("="*80)
        logger.info("[OK] Downstream evaluation completed successfully!")
        logger.info(f"Results saved to: {evaluate_res_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"[ERROR] Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*80)
    print("*** Downstream Tasks Evaluation Script")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Dim: {args.output_dim}")
    print(f"Device: {args.device}")
    print(f"Evaluate Tasks: {', '.join(args.evaluate_task)}")
    print("="*80 + "\n")

    run_downstream_evaluation(args)


if __name__ == '__main__':
    main()
