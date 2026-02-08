"""
HRNR_Hyperbolic模型多城市下游任务实验脚本

实验设置:
- 5个城市数据集: prt, cd, bj, xa, df
- 每个数据集运行5次 (不同seeds: 31, 42, 53, 64, 75)
- Embedding维度: 128
- 报告均值和标准差

使用方法:
    python run_hrnr_experiments.py --task segment --device gpu
    python run_hrnr_experiments.py --task segment --device gpu --datasets bj cd prt
"""

import os
import sys
import argparse
import subprocess
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run HRNR_Hyperbolic experiments on multiple cities')

    parser.add_argument('--task', type=str, default='segment',
                        help='Task name (default: segment)')
    parser.add_argument('--model', type=str, default='HRNR_Hyperbolic',
                        help='Model name (default: HRNR_Hyperbolic)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['prt', 'cd', 'bj', 'xa', 'df'],
                        help='Datasets to evaluate (default: prt cd bj xa df)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[31, 42, 53, 64, 75],
                        help='Random seeds for multiple runs (default: 31 42 53 64 75)')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use (cpu, gpu, default: gpu)')
    parser.add_argument('--exp_id', type=str, default='1',
                        help='Experiment ID (default: 1)')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                        help='Output directory for results (default: ./experiment_results)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments if results already exist')

    return parser.parse_args()


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def run_single_experiment(task, model, dataset, seed, output_dim, device, exp_id):
    """
    运行单次实验

    Args:
        task: 任务名称
        model: 模型名称
        dataset: 数据集名称
        seed: 随机种子
        output_dim: 输出维度
        device: 设备
        exp_id: 实验ID

    Returns:
        result_dict: 结果字典
    """
    print(f"\n{'='*80}")
    print(f"Running: dataset={dataset}, seed={seed}")
    print(f"{'='*80}\n")

    # 构建命令
    cmd = [
        'python', 'run_downstream_only.py',
        '--task', task,
        '--model', model,
        '--dataset', dataset,
        '--seed', str(seed),
        '--output_dim', str(output_dim),
        '--device', device,
        '--exp_id', str(exp_id)
    ]

    try:
        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )

        # 解析输出获取评估结果
        output = result.stdout
        stderr = result.stderr

        # 提取评估指标（需要根据实际输出格式调整）
        metrics = parse_evaluation_results(output)

        return {
            'dataset': dataset,
            'seed': seed,
            'status': 'success' if result.returncode == 0 else 'failed',
            'metrics': metrics,
            'stdout': output,
            'stderr': stderr
        }

    except subprocess.TimeoutExpired:
        print(f"WARNING: Experiment timed out for dataset={dataset}, seed={seed}")
        return {
            'dataset': dataset,
            'seed': seed,
            'status': 'timeout',
            'metrics': {},
            'stdout': '',
            'stderr': 'Timeout after 2 hours'
        }
    except Exception as e:
        print(f"ERROR: Experiment failed for dataset={dataset}, seed={seed}: {str(e)}")
        return {
            'dataset': dataset,
            'seed': seed,
            'status': 'error',
            'metrics': {},
            'stdout': '',
            'stderr': str(e)
        }


def parse_evaluation_results(output):
    """
    从输出中解析评估结果

    Args:
        output: 命令行输出

    Returns:
        metrics: 指标字典
    """
    metrics = {}

    # 解析速度推断指标
    if 'Speed Inference Results:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'MAE:' in line:
                try:
                    mae = float(line.split('MAE:')[1].strip())
                    metrics['speed_mae'] = mae
                except:
                    pass
            if 'RMSE:' in line:
                try:
                    rmse = float(line.split('RMSE:')[1].strip())
                    metrics['speed_rmse'] = rmse
                except:
                    pass
            if 'MAPE:' in line:
                try:
                    mape = float(line.split('MAPE:')[1].strip().rstrip('%'))
                    metrics['speed_mape'] = mape
                except:
                    pass

    # 解析旅行时间估计指标
    if 'Travel Time Estimation Results:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'MAE:' in line and 'travel' in output[max(0, output.find(line)-200):output.find(line)].lower():
                try:
                    mae = float(line.split('MAE:')[1].strip())
                    metrics['travel_time_mae'] = mae
                except:
                    pass
            if 'RMSE:' in line and 'travel' in output[max(0, output.find(line)-200):output.find(line)].lower():
                try:
                    rmse = float(line.split('RMSE:')[1].strip())
                    metrics['travel_time_rmse'] = rmse
                except:
                    pass
            if 'MAPE:' in line and 'travel' in output[max(0, output.find(line)-200):output.find(line)].lower():
                try:
                    mape = float(line.split('MAPE:')[1].strip().rstrip('%'))
                    metrics['travel_time_mape'] = mape
                except:
                    pass

    # 解析相似性搜索指标
    if 'Similarity Search Results:' in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'Precision@10:' in line:
                try:
                    p10 = float(line.split('Precision@10:')[1].strip())
                    metrics['similarity_p10'] = p10
                except:
                    pass
            if 'Recall@10:' in line:
                try:
                    r10 = float(line.split('Recall@10:')[1].strip())
                    metrics['similarity_r10'] = r10
                except:
                    pass

    return metrics


def aggregate_results(results_list):
    """
    聚合多次运行的结果，计算均值和标准差

    Args:
        results_list: 结果列表

    Returns:
        aggregated: 聚合后的结果
    """
    # 按数据集分组
    dataset_results = {}
    for result in results_list:
        dataset = result['dataset']
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        if result['status'] == 'success' and result['metrics']:
            dataset_results[dataset].append(result['metrics'])

    # 计算每个数据集的统计信息
    aggregated = {}
    for dataset, metrics_list in dataset_results.items():
        if not metrics_list:
            continue

        # 提取所有指标名称
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())

        # 计算每个指标的均值和标准差
        agg_metrics = {}
        for metric_name in all_metrics:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                agg_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

        aggregated[dataset] = {
            'num_runs': len(metrics_list),
            'metrics': agg_metrics
        }

    return aggregated


def save_results(results_list, aggregated, output_dir, args):
    """
    保存实验结果

    Args:
        results_list: 所有实验结果列表
        aggregated: 聚合后的结果
        output_dir: 输出目录
        args: 命令行参数
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存原始结果（JSON格式）
    raw_results_file = os.path.join(output_dir, f'raw_results_{timestamp}.json')
    with open(raw_results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results_list,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"\n✓ Raw results saved to: {raw_results_file}")

    # 保存聚合结果（JSON格式）
    agg_results_file = os.path.join(output_dir, f'aggregated_results_{timestamp}.json')
    with open(agg_results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'aggregated': aggregated,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"✓ Aggregated results saved to: {agg_results_file}")

    # 保存为易读的表格格式（CSV）
    csv_file = os.path.join(output_dir, f'results_table_{timestamp}.csv')
    save_results_as_table(aggregated, csv_file, args)
    print(f"✓ Results table saved to: {csv_file}")

    # 生成并保存报告
    report_file = os.path.join(output_dir, f'report_{timestamp}.txt')
    generate_report(aggregated, report_file, args)
    print(f"✓ Report saved to: {report_file}")


def save_results_as_table(aggregated, csv_file, args):
    """
    将结果保存为CSV表格

    Args:
        aggregated: 聚合后的结果
        csv_file: CSV文件路径
        args: 命令行参数
    """
    rows = []

    for dataset, data in aggregated.items():
        row = {
            'Dataset': dataset,
            'Model': args.model,
            'Task': args.task,
            'Num_Runs': data['num_runs']
        }

        # 添加每个指标的均值和标准差
        for metric_name, metric_data in data['metrics'].items():
            row[f'{metric_name}_mean'] = metric_data['mean']
            row[f'{metric_name}_std'] = metric_data['std']

        rows.append(row)

    # 创建DataFrame并保存
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, float_format='%.4f')


def generate_report(aggregated, report_file, args):
    """
    生成实验报告

    Args:
        aggregated: 聚合后的结果
        report_file: 报告文件路径
        args: 命令行参数
    """
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"HRNR_Hyperbolic Experimental Results Report\n")
        f.write("="*80 + "\n\n")

        f.write(f"Experiment Configuration:\n")
        f.write(f"  Model: {args.model}\n")
        f.write(f"  Task: {args.task}\n")
        f.write(f"  Datasets: {', '.join(args.datasets)}\n")
        f.write(f"  Seeds: {', '.join(map(str, args.seeds))}\n")
        f.write(f"  Output Dimension: {args.output_dim}\n")
        f.write(f"  Device: {args.device}\n")
        f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("Results Summary\n")
        f.write("="*80 + "\n\n")

        for dataset, data in sorted(aggregated.items()):
            f.write(f"\n{'-'*80}\n")
            f.write(f"Dataset: {dataset.upper()}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Number of successful runs: {data['num_runs']}/{len(args.seeds)}\n\n")

            if data['metrics']:
                # 按类别组织指标
                speed_metrics = {k: v for k, v in data['metrics'].items() if 'speed' in k}
                travel_metrics = {k: v for k, v in data['metrics'].items() if 'travel' in k}
                similarity_metrics = {k: v for k, v in data['metrics'].items() if 'similarity' in k}

                if speed_metrics:
                    f.write("Speed Inference:\n")
                    for metric_name, metric_data in speed_metrics.items():
                        f.write(f"  {metric_name:20s}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}\n")
                    f.write("\n")

                if travel_metrics:
                    f.write("Travel Time Estimation:\n")
                    for metric_name, metric_data in travel_metrics.items():
                        f.write(f"  {metric_name:20s}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}\n")
                    f.write("\n")

                if similarity_metrics:
                    f.write("Similarity Search:\n")
                    for metric_name, metric_data in similarity_metrics.items():
                        f.write(f"  {metric_name:20s}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}\n")
                    f.write("\n")
            else:
                f.write("  No metrics available\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")


def main():
    """主函数"""
    args = parse_args()

    print("="*80)
    print("HRNR_Hyperbolic Multi-City Experiments")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Task: {args.task}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Seeds: {', '.join(map(str, args.seeds))}")
    print(f"  Output Dim: {args.output_dim}")
    print(f"  Device: {args.device}")
    print(f"  Exp ID: {args.exp_id}")
    print(f"  Output Dir: {args.output_dir}")
    print()

    # 确保输出目录存在
    ensure_dir(args.output_dir)

    # 运行所有实验
    results_list = []
    total_experiments = len(args.datasets) * len(args.seeds)
    current_experiment = 0

    for dataset in args.datasets:
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Running experiment:")
            print(f"  Dataset: {dataset}, Seed: {seed}")

            result = run_single_experiment(
                task=args.task,
                model=args.model,
                dataset=dataset,
                seed=seed,
                output_dim=args.output_dim,
                device=args.device,
                exp_id=args.exp_id
            )

            results_list.append(result)

            # 打印结果摘要
            if result['status'] == 'success':
                print(f"  ✓ Status: SUCCESS")
                if result['metrics']:
                    print(f"  ✓ Metrics collected: {len(result['metrics'])} metrics")
            else:
                print(f"  ✗ Status: {result['status'].upper()}")

    # 聚合结果
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80)
    aggregated = aggregate_results(results_list)

    # 保存结果
    save_results(results_list, aggregated, args.output_dir, args)

    # 打印总结
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    successful = sum(1 for r in results_list if r['status'] == 'success')
    failed = sum(1 for r in results_list if r['status'] != 'success')
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
