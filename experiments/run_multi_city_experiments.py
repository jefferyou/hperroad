"""
多城市多GPU完整实验流程
支持：Beijing, Chengdu, Xi'an, San Francisco
自动化：预训练 + 下游任务评估（TSI/TTE/STS）
多GPU加速优化
"""

import os
import sys
import json
import argparse
from datetime import datetime
import subprocess

# 城市-数据集映射（根据VecCity命名规范）
DATASET_MAP = {
    'beijing': 'bj',
    'chengdu': 'cd',
    'xian': 'xa',
    'sanfrancisco': 'sf'
}

# HRNR默认下游任务轮数（从表格推断）
DEFAULT_TASK_EPOCHS = {
    'tsi': 1,      # Ridge regression，不需要训练epoch
    'tte': 100,    # HRNR原始论文设置
    'sts': 50      # HRNR原始论文设置
}

class MultiCityExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.veccity_path = os.path.join(self.project_root, 'VecCity-main')
        self.results = {}

    def apply_optimizations(self):
        """应用性能优化补丁"""
        print("\n" + "="*80)
        print("STEP 0: Applying Performance Optimizations")
        print("="*80)

        opt_script = os.path.join(self.script_dir, 'apply_downstream_optimizations.py')
        if os.path.exists(opt_script):
            print("Running optimization patch...")
            result = subprocess.run(['python', opt_script], cwd=self.script_dir)
            if result.returncode != 0:
                print("⚠ Warning: Optimization patch failed, continuing anyway...")
        else:
            print("⚠ Optimization script not found, skipping...")

    def run_training(self, city, dataset_code, seed=0):
        """运行预训练阶段"""
        print("\n" + "="*80)
        print(f"STEP 1: Training HRNR_Hyperbolic on {city} (dataset={dataset_code})")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"hrnr_hyp_{dataset_code}_s{seed}_{timestamp}"

        # 构建训练命令
        cmd = [
            'python', 'run_training_only.py',
            '--dataset', dataset_code,
            '--model', 'HRNR_Hyperbolic',
            '--task', 'segment',
            '--seed', str(seed),
            '--max_epoch', str(self.args.max_epoch),
            '--gpu', 'True',
            '--gpu_id', str(self.args.train_gpu)
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.args.train_gpu)

        print(f"Command: {' '.join(cmd)}")
        print(f"Experiment ID: {exp_id}")
        print(f"Using GPU: {self.args.train_gpu}")

        result = subprocess.run(cmd, cwd=self.script_dir, env=env)

        if result.returncode != 0:
            print(f"✗ Training failed for {city}")
            return None

        print(f"✓ Training completed for {city}")
        return exp_id

    def run_downstream_tasks(self, city, dataset_code, exp_id):
        """运行下游任务评估（使用多GPU加速）"""
        print("\n" + "="*80)
        print(f"STEP 2: Evaluating Downstream Tasks on {city}")
        print("="*80)

        gpu_ids = ','.join(map(str, self.args.eval_gpus))
        primary_gpu = self.args.eval_gpus[0]

        # 使用优化的评估脚本
        cmd = [
            'python', 'run_evaluation_only.py',
            '--exp_id', exp_id,
            '--task', 'all',  # 运行所有任务
            '--task_epoch', str(self.args.task_epoch),
            '--gpu_id', str(primary_gpu),
            '--dataset', dataset_code
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_ids

        print(f"Command: {' '.join(cmd)}")
        print(f"Using GPUs: {gpu_ids}")
        print(f"Task epochs: TSI=1, TTE={self.args.task_epoch}, STS={self.args.task_epoch}")

        result = subprocess.run(cmd, cwd=self.script_dir, env=env)

        if result.returncode != 0:
            print(f"✗ Downstream evaluation failed for {city}")
            return None

        print(f"✓ Downstream evaluation completed for {city}")
        return True

    def run_single_city(self, city):
        """运行单个城市的完整实验"""
        dataset_code = DATASET_MAP.get(city.lower())
        if not dataset_code:
            print(f"✗ Unknown city: {city}")
            return False

        print("\n" + "="*80)
        print(f"RUNNING COMPLETE EXPERIMENT FOR {city.upper()}")
        print("="*80)
        print(f"Dataset code: {dataset_code}")
        print(f"Training GPU: {self.args.train_gpu}")
        print(f"Evaluation GPUs: {self.args.eval_gpus}")
        print("="*80)

        start_time = datetime.now()

        # Step 1: 预训练
        exp_id = self.run_training(city, dataset_code, self.args.seed)
        if not exp_id:
            self.results[city] = {'status': 'FAILED', 'stage': 'training'}
            return False

        # Step 2: 下游任务评估
        success = self.run_downstream_tasks(city, dataset_code, exp_id)
        if not success:
            self.results[city] = {'status': 'FAILED', 'stage': 'evaluation', 'exp_id': exp_id}
            return False

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600  # hours

        self.results[city] = {
            'status': 'SUCCESS',
            'exp_id': exp_id,
            'dataset': dataset_code,
            'duration_hours': round(duration, 2),
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✓ {city} completed successfully in {duration:.2f} hours")
        return True

    def run_all_cities(self):
        """运行所有城市的实验"""
        cities = self.args.cities

        print("\n" + "="*80)
        print("MULTI-CITY EXPERIMENT RUNNER")
        print("="*80)
        print(f"Cities to process: {', '.join(cities)}")
        print(f"Total cities: {len(cities)}")
        print("="*80)

        # 先应用优化
        if not self.args.skip_optimization:
            self.apply_optimizations()

        # 依次运行每个城市
        overall_start = datetime.now()

        for i, city in enumerate(cities, 1):
            print(f"\n{'#'*80}")
            print(f"# CITY {i}/{len(cities)}: {city.upper()}")
            print(f"{'#'*80}")

            self.run_single_city(city)

        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds() / 3600

        # 保存结果摘要
        self.save_summary(total_duration)

    def save_summary(self, total_duration):
        """保存实验结果摘要"""
        summary = {
            'total_duration_hours': round(total_duration, 2),
            'cities_completed': sum(1 for r in self.results.values() if r['status'] == 'SUCCESS'),
            'cities_failed': sum(1 for r in self.results.values() if r['status'] == 'FAILED'),
            'results': self.results,
            'config': {
                'max_epoch': self.args.max_epoch,
                'task_epoch': self.args.task_epoch,
                'train_gpu': self.args.train_gpu,
                'eval_gpus': self.args.eval_gpus,
                'seed': self.args.seed
            }
        }

        # 保存JSON
        summary_file = os.path.join(self.script_dir, 'results',
                                    f'multi_city_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total duration: {total_duration:.2f} hours")
        print(f"Cities completed: {summary['cities_completed']}/{len(self.results)}")
        print(f"Cities failed: {summary['cities_failed']}/{len(self.results)}")
        print("\nResults by city:")
        for city, result in self.results.items():
            status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
            print(f"  {status_symbol} {city.upper()}: {result['status']}")
            if result['status'] == 'SUCCESS':
                print(f"      Exp ID: {result['exp_id']}")
                print(f"      Duration: {result['duration_hours']:.2f}h")
        print(f"\nSummary saved to: {summary_file}")
        print("="*80)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-city HRNR_Hyperbolic experiments with multi-GPU acceleration')

    # 城市选择
    parser.add_argument('--cities', type=str, nargs='+',
                        default=['xian', 'beijing', 'chengdu', 'sanfrancisco'],
                        choices=['beijing', 'chengdu', 'xian', 'sanfrancisco'],
                        help='Cities to run experiments on')

    # 训练参数
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Pretraining epochs')
    parser.add_argument('--task_epoch', type=int, default=100,
                        help='Downstream task epochs (TTE/STS)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # GPU配置
    parser.add_argument('--train_gpu', type=int, default=3,
                        help='GPU ID for pretraining')
    parser.add_argument('--eval_gpus', type=int, nargs='+', default=[3, 4, 5, 6, 7],
                        help='GPU IDs for downstream evaluation (multi-GPU)')

    # 优化选项
    parser.add_argument('--skip_optimization', action='store_true',
                        help='Skip applying optimization patches')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    runner = MultiCityExperimentRunner(args)
    runner.run_all_cities()
