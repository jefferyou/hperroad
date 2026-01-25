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

# 切换到VecCity根目录
os.chdir(veccity_path)

sys.path.insert(0, veccity_path)
sys.path.insert(0, project_root)

import argparse
from veccity.pipeline import run_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True,
                        help='Experiment ID (e.g., hrnr_hyp_xa_s0_20251231_092126)')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'tsi', 'tte', 'sts'],
                        help='Which downstream task to run')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--task_epoch', type=int, default=10,
                        help='Epochs for downstream tasks (TTE/STS)')
    parser.add_argument('--dataset', type=str, default='xa', help='Dataset name')
    parser.add_argument('--model', type=str, default='HRNR_Hyperbolic', help='Model name')
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

    # 检查embedding文件是否存在（使用glob查找，因为文件名可能不完全匹配）
    import glob
    embedding_pattern = os.path.join('veccity', 'cache', args.exp_id, 'evaluate_cache', 'road_embedding_*.npy')
    embedding_files = glob.glob(embedding_pattern)

    if not embedding_files:
        print(f"ERROR: No embedding files found matching pattern: {embedding_pattern}")
        print("Please run 'python run_training_only.py' first!")
        return

    embedding_path = embedding_files[0]  # 使用找到的第一个
    print(f"✓ Found embedding file: {embedding_path}")
    print("=" * 80)

    # 构建其他参数
    other_args = {
        'exp_id': args.exp_id,
        'gpu': args.gpu,
        'gpu_id': args.gpu_id,
        'task_epoch': args.task_epoch,
    }

    # 如果只运行特定任务，设置evaluate_tasks
    if args.task != 'all':
        task_model_map = {
            'tsi': (['tsi'], ['SpeedInferenceModel']),
            'tte': (['tte'], ['TravelTimeEstimationModel']),
            'sts': (['sts'], ['SimilaritySearchModel'])
        }
        tasks, models = task_model_map[args.task]
        other_args['evaluate_tasks'] = tasks
        other_args['evaluate_models'] = models
        print(f"Running only: {args.task.upper()}")
    else:
        print(f"Running all tasks: TSI, TTE, STS")

    print("=" * 80)

    # 运行模型 - train=False表示只评估
    result = run_model(
        task='segment',
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=None,
        saved_model=True,  # 使用已保存的模型
        train=False,  # 不训练，只评估
        other_args=other_args
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    if result is not None:
        print(f"Results: {result}")
    print("=" * 80)

    return result

if __name__ == '__main__':
    args = parse_args()
    run_evaluation_only(args)
