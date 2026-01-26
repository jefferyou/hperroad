#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的HRNR_Hyperbolic消融实验脚本

支持功能：
1. 完整的消融实验配置（包括HRNR基线）
2. 断点续传（从中断位置继续）
3. GPU加速下游任务
4. 详细的进度保存和结果记录
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VECCITY_ROOT = os.path.join(PROJECT_ROOT, 'VecCity-main')

# 切换到VecCity根目录
os.chdir(VECCITY_ROOT)

# 添加VecCity路径
sys.path.insert(0, VECCITY_ROOT)

from veccity.pipeline import run_model
from veccity.utils import ensure_dir


class AblationExperiment:
    """消融实验管理器"""

    def __init__(self, args):
        self.args = args
        self.results_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'ablation')
        ensure_dir(self.results_dir)

        # 进度文件
        self.progress_file = os.path.join(
            self.results_dir,
            f'ablation_progress_{args.dataset}_seed{args.seed}.json'
        )

        # 结果文件
        self.results_file = os.path.join(
            self.results_dir,
            f'ablation_results_{args.dataset}_seed{args.seed}.json'
        )

        # 定义消融实验配置
        self.ablation_configs = self._define_ablation_configs()

        # 加载进度
        self.progress = self._load_progress()

    def _define_ablation_configs(self):
        """定义所有消融实验配置"""
        configs = [
            # 1. Baseline: 原始HRNR（欧氏空间）
            {
                'id': 'baseline_hrnr',
                'name': 'Baseline HRNR',
                'model': 'HRNR',
                'description': 'Original HRNR with Euclidean space (no hyperbolic)',
                'params': {}
            },

            # 2. Full Model: 完整的HRNR_Hyperbolic
            {
                'id': 'full_model',
                'name': 'Full HRNR_Hyperbolic',
                'model': 'HRNR_Hyperbolic',
                'description': 'Full model with hyperbolic space + entailment + contrastive',
                'params': {
                    'lambda_ce': self.args.lambda_ce,
                    'lambda_cc': self.args.lambda_cc,
                }
            },

            # 3. No Entailment Loss
            {
                'id': 'no_entailment',
                'name': 'HRNR_Hyp w/o Entailment',
                'model': 'HRNR_Hyperbolic',
                'description': 'Hyperbolic space + contrastive (no entailment loss)',
                'params': {
                    'lambda_ce': 0.0,
                    'lambda_cc': self.args.lambda_cc,
                }
            },

            # 4. No Contrastive Loss
            {
                'id': 'no_contrastive',
                'name': 'HRNR_Hyp w/o Contrastive',
                'model': 'HRNR_Hyperbolic',
                'description': 'Hyperbolic space + entailment (no contrastive loss)',
                'params': {
                    'lambda_ce': self.args.lambda_ce,
                    'lambda_cc': 0.0,
                }
            },

            # 5. No Auxiliary Losses (仅双曲空间)
            {
                'id': 'no_auxiliary',
                'name': 'HRNR_Hyp w/o Auxiliary',
                'model': 'HRNR_Hyperbolic',
                'description': 'Only hyperbolic space (no entailment & contrastive)',
                'params': {
                    'lambda_ce': 0.0,
                    'lambda_cc': 0.0,
                }
            },
        ]

        return configs

    def _load_progress(self):
        """加载实验进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"✓ Loaded progress from: {self.progress_file}")
                print(f"  Completed: {len(progress.get('completed', []))} experiments")
                return progress
            except Exception as e:
                print(f"⚠ Failed to load progress: {e}")
                return {'completed': [], 'failed': [], 'current': None}
        else:
            return {'completed': [], 'failed': [], 'current': None}

    def _save_progress(self):
        """保存实验进度"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _is_completed(self, config_id):
        """检查实验是否已完成"""
        return config_id in self.progress.get('completed', [])

    def _mark_completed(self, config_id, result):
        """标记实验为已完成"""
        if 'completed' not in self.progress:
            self.progress['completed'] = []
        self.progress['completed'].append(config_id)
        self.progress['current'] = None
        self._save_progress()

        # 保存结果
        self._save_result(config_id, result)

    def _mark_failed(self, config_id, error):
        """标记实验为失败"""
        if 'failed' not in self.progress:
            self.progress['failed'] = []
        self.progress['failed'].append({
            'config_id': config_id,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        self.progress['current'] = None
        self._save_progress()

    def _save_result(self, config_id, result):
        """保存单个实验结果"""
        # 加载已有结果
        results = {}
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)

        # 添加新结果
        results[config_id] = result

        # 保存
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def run_single_config(self, config):
        """运行单个消融配置"""
        config_id = config['id']

        print("\n" + "=" * 80)
        print(f"Running: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Model: {config['model']}")
        print(f"Parameters: {config['params']}")
        print("=" * 80)

        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"ablation_{config_id}_{self.args.dataset}_s{self.args.seed}_{timestamp}"

        # 构建参数
        other_args = {
            'seed': self.args.seed,
            'gpu': self.args.gpu,
            'gpu_id': self.args.gpu_id,
            'exp_id': exp_id,
        }

        # 如果是HRNR_Hyperbolic，添加双曲参数
        if config['model'] == 'HRNR_Hyperbolic':
            other_args.update({
                'hyperbolic_dim': self.args.hyperbolic_dim,
                'temperature': self.args.temperature,
                'lp_learning_rate': self.args.learning_rate,
                'max_epoch': self.args.max_epoch,
            })
            # 添加消融参数
            other_args.update(config['params'])

        # 运行模型
        try:
            result = run_model(
                task=self.args.task,
                model_name=config['model'],
                dataset_name=self.args.dataset,
                config_file=self.args.config_file,
                saved_model=self.args.saved_model,
                train=self.args.train,
                other_args=other_args
            )

            return {
                'config': config,
                'exp_id': exp_id,
                'result': result,
                'timestamp': timestamp,
                'status': 'success'
            }

        except Exception as e:
            print(f"\n❌ ERROR in {config['name']}: {e}")
            import traceback
            traceback.print_exc()

            return {
                'config': config,
                'exp_id': exp_id,
                'error': str(e),
                'timestamp': timestamp,
                'status': 'failed'
            }

    def run_all(self):
        """运行所有消融实验"""
        print("=" * 80)
        print("COMPLETE ABLATION STUDY")
        print("=" * 80)
        print(f"Dataset: {self.args.dataset}")
        print(f"Seed: {self.args.seed}")
        print(f"GPU: {self.args.gpu} (ID: {self.args.gpu_id})")
        print(f"Total experiments: {len(self.ablation_configs)}")
        print("=" * 80)

        results = {}

        for i, config in enumerate(self.ablation_configs):
            config_id = config['id']

            print(f"\n{'='*80}")
            print(f"Experiment {i+1}/{len(self.ablation_configs)}: {config_id}")
            print(f"{'='*80}")

            # 检查是否已完成
            if self._is_completed(config_id):
                print(f"✓ Already completed, skipping...")
                continue

            # 标记为当前正在运行
            self.progress['current'] = config_id
            self._save_progress()

            # 运行实验
            result = self.run_single_config(config)

            # 保存结果
            if result['status'] == 'success':
                self._mark_completed(config_id, result)
                print(f"✓ Completed: {config_id}")
            else:
                self._mark_failed(config_id, result.get('error', 'Unknown error'))
                print(f"❌ Failed: {config_id}")

            results[config_id] = result

        # 生成总结报告
        self._generate_summary_report(results)

        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {self.results_file}")
        print(f"Completed: {len(self.progress.get('completed', []))}/{len(self.ablation_configs)}")
        print(f"Failed: {len(self.progress.get('failed', []))}")
        print("=" * 80)

        return results

    def _generate_summary_report(self, results):
        """生成总结报告"""
        summary_file = os.path.join(
            self.results_dir,
            f'ablation_summary_{self.args.dataset}_seed{self.args.seed}.txt'
        )

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {self.args.dataset}\n")
            f.write(f"Seed: {self.args.seed}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for config_id, result in results.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"Experiment: {result['config']['name']}\n")
                f.write(f"ID: {config_id}\n")
                f.write(f"Model: {result['config']['model']}\n")
                f.write(f"Description: {result['config']['description']}\n")
                f.write(f"Status: {result['status']}\n")

                if result['status'] == 'success':
                    f.write(f"Result: {result.get('result', 'N/A')}\n")
                else:
                    f.write(f"Error: {result.get('error', 'Unknown')}\n")

                f.write(f"{'-'*80}\n")

        print(f"✓ Summary report saved to: {summary_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run complete ablation study for HRNR_Hyperbolic')

    # 基础参数
    parser.add_argument('--task', type=str, default='segment',
                        help='Task type (segment/parcel/poi)')
    parser.add_argument('--dataset', type=str, default='xa',
                        help='Dataset name')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path')

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

    # 双曲空间超参数
    parser.add_argument('--hyperbolic_dim', type=int, default=224,
                        help='Hyperbolic space dimension')
    parser.add_argument('--lambda_ce', type=float, default=0.1,
                        help='Entailment loss weight (for full model)')
    parser.add_argument('--lambda_cc', type=float, default=0.1,
                        help='Contrastive loss weight (for full model)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Maximum training epochs')

    # 消融实验特定参数
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline HRNR (only run hyperbolic variants)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous progress')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建消融实验管理器
    ablation = AblationExperiment(args)

    # 如果resume标志开启，显示当前进度
    if args.resume:
        print("=" * 80)
        print("RESUMING FROM PREVIOUS PROGRESS")
        print("=" * 80)
        completed = ablation.progress.get('completed', [])
        failed = ablation.progress.get('failed', [])
        print(f"Completed experiments: {len(completed)}")
        print(f"Failed experiments: {len(failed)}")
        if completed:
            print(f"Completed IDs: {', '.join(completed)}")
        if failed:
            print(f"Failed IDs: {[f['config_id'] for f in failed]}")
        print("=" * 80)

    # 运行所有消融实验
    results = ablation.run_all()

    return results


if __name__ == '__main__':
    main()
