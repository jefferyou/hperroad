"""
独立的评估脚本 - 只执行下游任务评估
支持选择性运行特定任务：tsi, tte, sts
"""
import sys
import os

# 添加正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
veccity_path = os.path.join(project_root, 'VecCity-main')
sys.path.insert(0, veccity_path)
sys.path.insert(0, project_root)

import argparse
import torch
import numpy as np
from veccity.utils import get_logger, ensure_dir, ConfigParser
from veccity.downstream import get_evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True,
                        help='Experiment ID (e.g., hrnr_hyp_xa_s0_20251231_092126)')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'tsi', 'tte', 'sts'],
                        help='Which downstream task to run')
    parser.add_argument('--gpu', type=str, default='True', help='Use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--task_epoch', type=int, default=10,
                        help='Epochs for downstream tasks (TTE/STS)')
    return parser.parse_args()

def run_evaluation_only(args):
    """只运行下游评估，使用已保存的embedding"""
    print("=" * 80)
    print("Running DOWNSTREAM EVALUATION ONLY")
    print(f"Experiment ID: {args.exp_id}")
    print(f"Task: {args.task}")
    print(f"GPU: {args.gpu} (ID: {args.gpu_id})")
    print(f"Task Epochs: {args.task_epoch}")
    print("=" * 80)

    # 检查embedding文件是否存在
    embedding_path = f'./veccity/cache/{args.exp_id}/evaluate_cache/road_embedding_HRNR_Hyperbolic_xa_128.npy'
    if not os.path.exists(embedding_path):
        print(f"ERROR: Embedding file not found at: {embedding_path}")
        print("Please run 'python run_training_only.py' first!")
        return

    print(f"✓ Found embedding file: {embedding_path}")
    emb = np.load(embedding_path)
    print(f"✓ Loaded embedding shape: {emb.shape}")
    print("=" * 80)

    # 重建配置
    from veccity.utils import general_arguments

    other_args = {
        'task': 'segment',
        'model': 'HRNR_Hyperbolic',
        'dataset': 'xa',
        'exp_id': args.exp_id,
        'saved_model': True,
        'train': False,  # 不训练
    }

    cmd_args = general_arguments()
    for key, value in other_args.items():
        setattr(cmd_args, key, value)

    setattr(cmd_args, 'gpu', args.gpu == 'True')
    setattr(cmd_args, 'gpu_id', args.gpu_id)
    setattr(cmd_args, 'task_epoch', args.task_epoch)

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

    # 覆盖evaluate_tasks
    if args.task != 'all':
        task_model_map = {
            'tsi': (['tsi'], ['SpeedInferenceModel']),
            'tte': (['tte'], ['TravelTimeEstimationModel']),
            'sts': (['sts'], ['SimilaritySearchModel'])
        }
        tasks, models = task_model_map[args.task]
        config['evaluate_tasks'] = tasks
        config['evaluate_models'] = models
        print(f"Running only: {args.task.upper()}")
    else:
        print(f"Running all tasks: TSI, TTE, STS")

    config['task_epoch'] = args.task_epoch
    print(f"Task epochs: {args.task_epoch}")
    print("=" * 80)

    # 获取logger
    logger = get_logger(config)
    logger.info('EVALUATION ONLY MODE - Using pre-trained embeddings')
    logger.info(f'Embedding path: {embedding_path}')
    logger.info(f'Task: {args.task}')

    # 加载数据（用于evaluator）
    from veccity.data import get_dataset
    dataset = get_dataset(config)
    data_feature = dataset.get_data_feature()

    # 创建evaluator
    evaluator = get_evaluator(config, data_feature)

    # 创建一个虚拟模型来传递embedding
    class DummyModel:
        def __init__(self, output_dim):
            self.output_dim = output_dim

        def encode(self, x):
            """Mock encode method for compatibility"""
            return torch.from_numpy(emb[x.cpu().numpy()]).float()

        def encode_sequence(self, batch):
            """Mock encode_sequence for STS"""
            path = batch['seq'][:,:,0]
            return self.encode(path.view(-1)).view(path.shape[0], path.shape[1], -1)

    dummy_model = DummyModel(output_dim=config['output_dim'])

    # 运行评估
    logger.info("=" * 80)
    logger.info("Starting DOWNSTREAM EVALUATION...")
    logger.info("=" * 80)

    result = evaluator.evaluate(model=dummy_model)

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results: {result}")
    logger.info("=" * 80)

    return result

if __name__ == '__main__':
    args = parse_args()
    run_evaluation_only(args)
