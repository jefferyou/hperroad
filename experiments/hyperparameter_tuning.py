#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNR_Hyperbolic 超参数优化

支持多种优化算法：
1. Grid Search (网格搜索)
2. Random Search (随机搜索)
3. Bayesian Optimization (贝叶斯优化)
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime
from itertools import product

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VECCITY_ROOT = os.path.join(PROJECT_ROOT, 'VecCity-main')

# 切换到VecCity根目录（VecCity期望从这里运行）
os.chdir(VECCITY_ROOT)

# 添加VecCity路径
sys.path.insert(0, VECCITY_ROOT)

from veccity.pipeline import run_model
from veccity.utils import ensure_dir


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, task, model, dataset, base_config, search_space,
                 metric='auc', mode='max', max_trials=50, method='random'):
        """
        Args:
            task: 任务类型
            model: 模型名称
            dataset: 数据集名称
            base_config: 基础配置字典
            search_space: 超参数搜索空间
            metric: 优化的指标
            mode: 'max' 或 'min'
            max_trials: 最大尝试次数
            method: 搜索方法 ('grid', 'random', 'bayesian')
        """
        self.task = task
        self.model = model
        self.dataset = dataset
        self.base_config = base_config
        self.search_space = search_space
        self.metric = metric
        self.mode = mode
        self.max_trials = max_trials
        self.method = method

        self.trials = []
        self.best_trial = None
        self.best_score = float('-inf') if mode == 'max' else float('inf')

    def _is_better(self, score):
        """判断新分数是否更好"""
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score

    def _generate_grid_configs(self):
        """生成网格搜索配置"""
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]

        configs = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)

        return configs

    def _generate_random_configs(self, n_samples):
        """生成随机搜索配置"""
        configs = []
        for _ in range(n_samples):
            config = {}
            for key, values in self.search_space.items():
                if isinstance(values, list):
                    config[key] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # 假设是范围 (min, max)
                    if isinstance(values[0], int):
                        config[key] = np.random.randint(values[0], values[1] + 1)
                    else:
                        config[key] = np.random.uniform(values[0], values[1])
            configs.append(config)

        return configs

    def _run_trial(self, trial_id, hyperparams):
        """运行单次试验"""
        print(f"\n{'='*80}")
        print(f"Trial {trial_id + 1}/{self.max_trials}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*80}")

        # 合并配置
        config = {**self.base_config, **hyperparams}

        # 生成实验ID
        exp_id = f"tune_{self.model}_{self.dataset}_trial{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 将exp_id添加到配置中
        config['exp_id'] = exp_id

        try:
            # 运行模型（所有参数通过other_args传递）
            result = run_model(
                task=self.task,
                model_name=self.model,
                dataset_name=self.dataset,
                config_file=None,
                saved_model=False,  # 调参时不保存所有模型
                train=True,
                other_args=config
            )

            # 提取目标指标
            if result is not None and self.metric in result:
                score = result[self.metric]
            else:
                score = float('-inf') if self.mode == 'max' else float('inf')
                print(f"Warning: Metric '{self.metric}' not found in result")

            trial_result = {
                'trial_id': trial_id,
                'hyperparams': hyperparams,
                'score': score,
                'full_result': result,
                'exp_id': exp_id,
                'timestamp': datetime.now().isoformat()
            }

            # 更新最佳结果
            if self._is_better(score):
                self.best_score = score
                self.best_trial = trial_result
                print(f"✓ New best score: {score:.4f}")

            self.trials.append(trial_result)

            return trial_result

        except Exception as e:
            print(f"Error in trial {trial_id}: {str(e)}")
            trial_result = {
                'trial_id': trial_id,
                'hyperparams': hyperparams,
                'score': float('-inf') if self.mode == 'max' else float('inf'),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.trials.append(trial_result)
            return trial_result

    def run_grid_search(self):
        """网格搜索"""
        print("=" * 80)
        print("Starting Grid Search")
        print("=" * 80)

        configs = self._generate_grid_configs()
        print(f"Total configurations to try: {len(configs)}")

        # 限制最大试验次数
        configs = configs[:self.max_trials]

        for i, hyperparams in enumerate(configs):
            self._run_trial(i, hyperparams)

        return self._summarize_results()

    def run_random_search(self):
        """随机搜索"""
        print("=" * 80)
        print("Starting Random Search")
        print("=" * 80)
        print(f"Total trials: {self.max_trials}")

        configs = self._generate_random_configs(self.max_trials)

        for i, hyperparams in enumerate(configs):
            self._run_trial(i, hyperparams)

        return self._summarize_results()

    def run_bayesian_optimization(self):
        """贝叶斯优化（简化版，使用随机搜索模拟）"""
        print("=" * 80)
        print("Starting Bayesian Optimization (simplified)")
        print("=" * 80)
        print("Note: Using random search with adaptive sampling")

        # TODO: 实现真正的贝叶斯优化（需要安装额外库如GPyOpt）
        # 这里使用随机搜索作为简化实现
        return self.run_random_search()

    def run(self):
        """运行超参数搜索"""
        if self.method == 'grid':
            return self.run_grid_search()
        elif self.method == 'random':
            return self.run_random_search()
        elif self.method == 'bayesian':
            return self.run_bayesian_optimization()
        else:
            raise ValueError(f"Unknown search method: {self.method}")

    def _summarize_results(self):
        """总结结果"""
        print("\n" + "=" * 80)
        print("Hyperparameter Tuning Summary")
        print("=" * 80)
        print(f"Total trials: {len(self.trials)}")
        print(f"Best {self.metric}: {self.best_score:.4f}")
        print(f"Best hyperparameters:")
        for key, value in self.best_trial['hyperparams'].items():
            print(f"  {key}: {value}")

        # 保存结果（使用绝对路径）
        results_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results')
        result_file = os.path.join(results_dir, f"hypertuning_{self.model}_{self.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        ensure_dir(results_dir)

        summary = {
            'task': self.task,
            'model': self.model,
            'dataset': self.dataset,
            'search_space': self.search_space,
            'method': self.method,
            'metric': self.metric,
            'mode': self.mode,
            'max_trials': self.max_trials,
            'best_score': self.best_score,
            'best_trial': self.best_trial,
            'all_trials': self.trials,
            'timestamp': datetime.now().isoformat()
        }

        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {result_file}")
        print("=" * 80)

        return summary


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for HRNR_Hyperbolic')

    parser.add_argument('--task', type=str, default='road_representation',
                        help='Task type')
    parser.add_argument('--model', type=str, default='HRNR_Hyperbolic',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='xa',
                        help='Dataset name')

    # 搜索设置
    parser.add_argument('--method', type=str, default='random',
                        choices=['grid', 'random', 'bayesian'],
                        help='Search method')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='Maximum number of trials')
    parser.add_argument('--metric', type=str, default='auc',
                        help='Metric to optimize')
    parser.add_argument('--mode', type=str, default='max',
                        choices=['max', 'min'],
                        help='Optimization mode')

    # GPU设置
    parser.add_argument('--gpu', type=bool, default=True,
                        help='Use GPU or not')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # 搜索空间配置文件
    parser.add_argument('--search_space_file', type=str, default=None,
                        help='JSON file defining search space')

    return parser.parse_args()


def get_default_search_space():
    """获取默认搜索空间"""
    return {
        # 双曲空间维度
        'hyperbolic_dim': [128, 224, 256],

        # 损失权重
        'lambda_ce': [0.0, 0.05, 0.1, 0.15, 0.2],
        'lambda_cc': [0.0, 0.05, 0.1, 0.15, 0.2],

        # 对比学习温度
        'temperature': [0.05, 0.07, 0.1],

        # 学习率
        'lp_learning_rate': [5e-5, 1e-4, 2e-4, 5e-4],

        # Dropout
        'dropout': [0.3, 0.5, 0.6, 0.7],

        # 其他
        'alpha': [0.1, 0.2, 0.3],  # LeakyReLU alpha
    }


def main():
    """主函数"""
    args = parse_args()

    # 基础配置
    base_config = {
        'gpu': args.gpu,
        'gpu_id': args.gpu_id,
        'seed': args.seed,
        'max_epoch': 100,
    }

    # 加载搜索空间
    if args.search_space_file is not None:
        with open(args.search_space_file, 'r') as f:
            search_space = json.load(f)
    else:
        search_space = get_default_search_space()

    print("=" * 80)
    print("HRNR_Hyperbolic Hyperparameter Tuning")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Max trials: {args.max_trials}")
    print(f"Metric: {args.metric} ({args.mode})")
    print(f"\nSearch space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    print("=" * 80)

    # 创建调优器
    tuner = HyperparameterTuner(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        base_config=base_config,
        search_space=search_space,
        metric=args.metric,
        mode=args.mode,
        max_trials=args.max_trials,
        method=args.method
    )

    # 运行调优
    results = tuner.run()

    print("\n" + "=" * 80)
    print("Hyperparameter tuning completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
