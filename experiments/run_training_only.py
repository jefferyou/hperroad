"""
独立的预训练脚本 - 只执行上游模型训练
简化版本，直接使用run_model
"""
import sys
import os

# 添加正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
veccity_path = os.path.join(project_root, 'VecCity-main')

# 切换到VecCity根目录
os.chdir(veccity_path)

sys.path.insert(0, veccity_path)
sys.path.insert(0, project_root)

import argparse
from veccity.pipeline import run_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='xa', help='Dataset name (xa/bj/cd/sf)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=100, help='Max training epochs')
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

    # 准备参数
    other_args = {
        'seed': args.seed,
        'max_epoch': args.max_epoch,
        'gpu': args.gpu == 'True',
        'gpu_id': args.gpu_id,
        'train': True,  # 开启训练
        # 设置空的下游任务列表，跳过评估
        'evaluate_tasks': [],
        'evaluate_models': [],
    }

    # 调用VecCity pipeline，只训练不评估
    result = run_model(
        task='segment',
        model_name='HRNR_Hyperbolic',
        dataset_name=args.dataset,
        config_file=None,
        saved_model=True,
        train=True,  # 训练模式
        other_args=other_args
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Model and embeddings have been saved to:")
    print(f"  ./veccity/cache/<exp_id>/model_cache/")
    print(f"  ./veccity/cache/<exp_id>/evaluate_cache/")
    print("\nNext step: Run 'run_evaluation_only.py' with the generated exp_id")
    print("=" * 80)

    return result

if __name__ == '__main__':
    args = parse_args()
    run_training_only(args)
