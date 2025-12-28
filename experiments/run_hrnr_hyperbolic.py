#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNR_Hyperbolic 实验运行脚本

支持功能：
1. 单次实验运行
2. 多次实验（不同随机种子）
3. 超参数搜索
4. 结果对比分析
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

# 添加VecCity路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../VecCity-main')))

from veccity.pipeline import run_model
from veccity.utils import ensure_dir


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run HRNR_Hyperbolic experiments')

    # 基础参数
    parser.add_argument('--task', type=str, default='road_representation',
                        help='Task type')
    parser.add_argument('--model', type=str, default='HRNR_Hyperbolic',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='xa',
                        help='Dataset name')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (auto-generated if not provided)')

    # GPU设置
    parser.add_argument('--gpu', type=bool, default=True,
                        help='Use GPU or not')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')

    # 训练设置
    parser.add_argument('--train', type=bool, default=True,
                        help='Train the model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--saved_model', type=bool, default=True,
                        help='Save the trained model')

    # 超参数
    parser.add_argument('--hyperbolic_dim', type=int, default=224,
                        help='Hyperbolic space dimension')
    parser.add_argument('--lambda_ce', type=float, default=0.1,
                        help='Entailment loss weight')
    parser.add_argument('--lambda_cc', type=float, default=0.1,
                        help='Contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Maximum training epochs')

    # 实验模式
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'multi_seed', 'ablation', 'comparison'],
                        help='Experiment mode')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs for multi_seed mode')

    return parser.parse_args()


def run_single_experiment(args):
    """运行单次实验"""
    print("=" * 80)
    print(f"Running HRNR_Hyperbolic experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"GPU: {args.gpu} (ID: {args.gpu_id})")
    print("=" * 80)

    # 生成实验ID
    if args.exp_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_id = f"hrnr_hyp_{args.dataset}_s{args.seed}_{timestamp}"

    # 构建其他参数字典
    other_args = {
        'hyperbolic_dim': args.hyperbolic_dim,
        'lambda_ce': args.lambda_ce,
        'lambda_cc': args.lambda_cc,
        'temperature': args.temperature,
        'lp_learning_rate': args.learning_rate,
        'max_epoch': args.max_epoch,
    }

    # 运行模型
    result = run_model(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        other_args=other_args,
        seed=args.seed,
        gpu_id=args.gpu_id,
        gpu=args.gpu,
        exp_id=args.exp_id
    )

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print(f"Experiment ID: {args.exp_id}")
    if result is not None:
        print(f"Results: {result}")
    print("=" * 80)

    return result


def run_multi_seed_experiments(args):
    """运行多个随机种子的实验"""
    print("=" * 80)
    print(f"Running {args.num_runs} experiments with different seeds")
    print("=" * 80)

    results = []
    seeds = [args.seed + i for i in range(args.num_runs)]

    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Run {i+1}/{args.num_runs} - Seed: {seed}")
        print(f"{'='*80}")

        args.seed = seed
        args.exp_id = f"hrnr_hyp_{args.dataset}_s{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = run_single_experiment(args)
        results.append({
            'seed': seed,
            'exp_id': args.exp_id,
            'result': result
        })

    # 计算统计结果
    print("\n" + "=" * 80)
    print("Multi-seed experiment summary:")
    print("=" * 80)

    # 保存结果
    summary_file = f"./experiments/results/hrnr_hyperbolic_{args.dataset}_multi_seed_summary.json"
    ensure_dir(os.path.dirname(summary_file))

    with open(summary_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'num_runs': args.num_runs,
            'seeds': seeds,
            'results': results,
            'config': vars(args)
        }, f, indent=2)

    print(f"Results saved to: {summary_file}")

    return results


def run_ablation_study(args):
    """消融实验：测试不同组件的影响"""
    print("=" * 80)
    print("Running ablation study")
    print("=" * 80)

    ablation_configs = [
        # 基线：完整模型
        {
            'name': 'full_model',
            'lambda_ce': args.lambda_ce,
            'lambda_cc': args.lambda_cc,
            'description': 'Full model with entailment and contrastive loss'
        },
        # 消融1：无蕴含损失
        {
            'name': 'no_entailment',
            'lambda_ce': 0.0,
            'lambda_cc': args.lambda_cc,
            'description': 'Without entailment loss'
        },
        # 消融2：无对比损失
        {
            'name': 'no_contrastive',
            'lambda_ce': args.lambda_ce,
            'lambda_cc': 0.0,
            'description': 'Without contrastive loss'
        },
        # 消融3：无蕴含和对比
        {
            'name': 'no_auxiliary',
            'lambda_ce': 0.0,
            'lambda_cc': 0.0,
            'description': 'Without both auxiliary losses (structural loss only)'
        },
    ]

    results = []

    for i, config in enumerate(ablation_configs):
        print(f"\n{'='*80}")
        print(f"Ablation {i+1}/{len(ablation_configs)}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")

        # 更新参数
        args.lambda_ce = config['lambda_ce']
        args.lambda_cc = config['lambda_cc']
        args.exp_id = f"hrnr_hyp_ablation_{config['name']}_{args.dataset}_s{args.seed}"

        result = run_single_experiment(args)
        results.append({
            'config': config,
            'result': result
        })

    # 保存消融结果
    ablation_file = f"./experiments/results/hrnr_hyperbolic_{args.dataset}_ablation_study.json"
    ensure_dir(os.path.dirname(ablation_file))

    with open(ablation_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'ablation_configs': ablation_configs,
            'results': results
        }, f, indent=2)

    print(f"\nAblation study results saved to: {ablation_file}")

    return results


def run_comparison_experiment(args):
    """对比实验：HRNR vs HRNR_Hyperbolic"""
    print("=" * 80)
    print("Running comparison: HRNR vs HRNR_Hyperbolic")
    print("=" * 80)

    models = ['HRNR', 'HRNR_Hyperbolic']
    results = {}

    for model in models:
        print(f"\n{'='*80}")
        print(f"Running {model}")
        print(f"{'='*80}")

        args.model = model
        args.exp_id = f"{model.lower()}_{args.dataset}_s{args.seed}_comparison"

        result = run_single_experiment(args)
        results[model] = result

    # 保存对比结果
    comparison_file = f"./experiments/results/hrnr_comparison_{args.dataset}.json"
    ensure_dir(os.path.dirname(comparison_file))

    with open(comparison_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'seed': args.seed,
            'models': models,
            'results': results
        }, f, indent=2)

    print(f"\nComparison results saved to: {comparison_file}")

    return results


def main():
    """主函数"""
    args = parse_args()

    # 根据模式选择运行方式
    if args.mode == 'single':
        run_single_experiment(args)
    elif args.mode == 'multi_seed':
        run_multi_seed_experiments(args)
    elif args.mode == 'ablation':
        run_ablation_study(args)
    elif args.mode == 'comparison':
        run_comparison_experiment(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
