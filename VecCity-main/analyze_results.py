"""
结果分析脚本
用于分析HRNR实验的输出日志，提取指标并计算统计信息

使用方法:
    python analyze_results.py ./experiment_results/20260203_120000
    python analyze_results.py ./experiment_results/20260203_120000 --output summary.csv
"""

import os
import sys
import argparse
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Analyze HRNR experiment results')

    parser.add_argument('result_dir', type=str,
                        help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (default: auto-generated)')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'markdown', 'latex'],
                        default='csv',
                        help='Output format (default: csv)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')

    return parser.parse_args()


def extract_metrics_from_log(log_file):
    """
    从日志文件中提取评估指标

    Args:
        log_file: 日志文件路径

    Returns:
        metrics: 指标字典
    """
    if not os.path.exists(log_file):
        return None

    metrics = {}

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 提取速度推断指标
        patterns = {
            'speed_mae': r'Speed.*?MAE[:\s]+(\d+\.?\d*)',
            'speed_rmse': r'Speed.*?RMSE[:\s]+(\d+\.?\d*)',
            'speed_mape': r'Speed.*?MAPE[:\s]+(\d+\.?\d*)%?',
            'speed_r2': r'Speed.*?R2[:\s]+(\d+\.?\d*)',

            # 旅行时间估计
            'travel_time_mae': r'Travel Time.*?MAE[:\s]+(\d+\.?\d*)',
            'travel_time_rmse': r'Travel Time.*?RMSE[:\s]+(\d+\.?\d*)',
            'travel_time_mape': r'Travel Time.*?MAPE[:\s]+(\d+\.?\d*)%?',
            'travel_time_r2': r'Travel Time.*?R2[:\s]+(\d+\.?\d*)',

            # 相似性搜索
            'similarity_p5': r'Precision@5[:\s]+(\d+\.?\d*)',
            'similarity_p10': r'Precision@10[:\s]+(\d+\.?\d*)',
            'similarity_r5': r'Recall@5[:\s]+(\d+\.?\d*)',
            'similarity_r10': r'Recall@10[:\s]+(\d+\.?\d*)',
            'similarity_map': r'MAP[:\s]+(\d+\.?\d*)',
        }

        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    # 取最后一个匹配（通常是最终结果）
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    pass

    except Exception as e:
        print(f"Warning: Error parsing {log_file}: {str(e)}")
        return None

    return metrics if metrics else None


def parse_filename(filename):
    """
    从文件名中提取数据集和种子信息

    Args:
        filename: 文件名，如 result_bj_seed31.log

    Returns:
        dataset, seed: 数据集名称和种子
    """
    match = re.search(r'result_(\w+)_seed(\d+)', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def analyze_results(result_dir, verbose=False):
    """
    分析实验结果目录

    Args:
        result_dir: 结果目录
        verbose: 是否打印详细信息

    Returns:
        all_results: 所有结果列表
        aggregated: 聚合后的统计信息
    """
    result_files = list(Path(result_dir).glob('result_*.log'))

    if not result_files:
        print(f"No result files found in {result_dir}")
        return [], {}

    print(f"Found {len(result_files)} result files")

    all_results = []
    dataset_metrics = defaultdict(lambda: defaultdict(list))

    for result_file in result_files:
        dataset, seed = parse_filename(result_file.name)

        if dataset is None:
            if verbose:
                print(f"Skipping {result_file.name}: cannot parse filename")
            continue

        metrics = extract_metrics_from_log(result_file)

        if metrics:
            all_results.append({
                'dataset': dataset,
                'seed': seed,
                'metrics': metrics
            })

            # 按数据集收集指标
            for metric_name, value in metrics.items():
                dataset_metrics[dataset][metric_name].append(value)

            if verbose:
                print(f"✓ {result_file.name}: {len(metrics)} metrics extracted")
        else:
            if verbose:
                print(f"✗ {result_file.name}: no metrics found")

    # 计算统计信息
    aggregated = {}
    for dataset, metrics_dict in dataset_metrics.items():
        aggregated[dataset] = {}
        for metric_name, values in metrics_dict.items():
            aggregated[dataset][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values),
                'values': values
            }

    return all_results, aggregated


def format_results_csv(aggregated):
    """将结果格式化为CSV格式"""
    rows = []

    for dataset in sorted(aggregated.keys()):
        metrics = aggregated[dataset]

        row = {'Dataset': dataset}

        for metric_name in sorted(metrics.keys()):
            stats = metrics[metric_name]
            row[f'{metric_name}_mean'] = stats['mean']
            row[f'{metric_name}_std'] = stats['std']
            row[f'{metric_name}_count'] = stats['count']

        rows.append(row)

    return pd.DataFrame(rows)


def format_results_markdown(aggregated):
    """将结果格式化为Markdown表格"""
    lines = ["# HRNR_Hyperbolic Experimental Results\n"]

    for dataset in sorted(aggregated.keys()):
        lines.append(f"\n## Dataset: {dataset.upper()}\n")

        metrics = aggregated[dataset]

        # 分类指标
        speed_metrics = {k: v for k, v in metrics.items() if 'speed' in k}
        travel_metrics = {k: v for k, v in metrics.items() if 'travel' in k}
        similarity_metrics = {k: v for k, v in metrics.items() if 'similarity' in k}

        if speed_metrics:
            lines.append("\n### Speed Inference\n")
            lines.append("| Metric | Mean | Std | Min | Max | Count |")
            lines.append("|--------|------|-----|-----|-----|-------|")
            for metric_name, stats in sorted(speed_metrics.items()):
                lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |")

        if travel_metrics:
            lines.append("\n### Travel Time Estimation\n")
            lines.append("| Metric | Mean | Std | Min | Max | Count |")
            lines.append("|--------|------|-----|-----|-----|-------|")
            for metric_name, stats in sorted(travel_metrics.items()):
                lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |")

        if similarity_metrics:
            lines.append("\n### Similarity Search\n")
            lines.append("| Metric | Mean | Std | Min | Max | Count |")
            lines.append("|--------|------|-----|-----|-----|-------|")
            for metric_name, stats in sorted(similarity_metrics.items()):
                lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |")

    return '\n'.join(lines)


def format_results_latex(aggregated):
    """将结果格式化为LaTeX表格"""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{HRNR\\_Hyperbolic Experimental Results}",
        "\\label{tab:hrnr_results}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Dataset & Metric & Mean & Std & Count \\\\",
        "\\hline"
    ]

    for dataset in sorted(aggregated.keys()):
        metrics = aggregated[dataset]

        for i, (metric_name, stats) in enumerate(sorted(metrics.items())):
            if i == 0:
                lines.append(f"{dataset} & {metric_name} & {stats['mean']:.4f} & "
                           f"{stats['std']:.4f} & {stats['count']} \\\\")
            else:
                lines.append(f" & {metric_name} & {stats['mean']:.4f} & "
                           f"{stats['std']:.4f} & {stats['count']} \\\\")

        lines.append("\\hline")

    lines.extend([
        "\\end{tabular}",
        "\\end{table}"
    ])

    return '\n'.join(lines)


def print_summary(aggregated):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("HRNR_Hyperbolic Experimental Results Summary")
    print("="*80 + "\n")

    for dataset in sorted(aggregated.keys()):
        print(f"\n{'-'*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'-'*80}")

        metrics = aggregated[dataset]

        # 分类显示
        speed_metrics = {k: v for k, v in metrics.items() if 'speed' in k}
        travel_metrics = {k: v for k, v in metrics.items() if 'travel' in k}
        similarity_metrics = {k: v for k, v in metrics.items() if 'similarity' in k}

        if speed_metrics:
            print("\nSpeed Inference:")
            for metric_name, stats in sorted(speed_metrics.items()):
                print(f"  {metric_name:25s}: {stats['mean']:8.4f} ± {stats['std']:8.4f} "
                      f"(n={stats['count']})")

        if travel_metrics:
            print("\nTravel Time Estimation:")
            for metric_name, stats in sorted(travel_metrics.items()):
                print(f"  {metric_name:25s}: {stats['mean']:8.4f} ± {stats['std']:8.4f} "
                      f"(n={stats['count']})")

        if similarity_metrics:
            print("\nSimilarity Search:")
            for metric_name, stats in sorted(similarity_metrics.items()):
                print(f"  {metric_name:25s}: {stats['mean']:8.4f} ± {stats['std']:8.4f} "
                      f"(n={stats['count']})")

    print("\n" + "="*80 + "\n")


def main():
    """主函数"""
    args = parse_args()

    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    print(f"Analyzing results in: {args.result_dir}")

    # 分析结果
    all_results, aggregated = analyze_results(args.result_dir, verbose=args.verbose)

    if not aggregated:
        print("No results to analyze")
        sys.exit(1)

    # 打印摘要
    print_summary(aggregated)

    # 生成输出文件
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.result_dir, f'summary.{args.format}')

    if args.format == 'csv':
        df = format_results_csv(aggregated)
        df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"✓ CSV summary saved to: {output_file}")

    elif args.format == 'json':
        with open(output_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'aggregated': {
                    dataset: {
                        metric: {k: v for k, v in stats.items() if k != 'values'}
                        for metric, stats in metrics.items()
                    }
                    for dataset, metrics in aggregated.items()
                }
            }, f, indent=2)
        print(f"✓ JSON summary saved to: {output_file}")

    elif args.format == 'markdown':
        markdown = format_results_markdown(aggregated)
        with open(output_file, 'w') as f:
            f.write(markdown)
        print(f"✓ Markdown summary saved to: {output_file}")

    elif args.format == 'latex':
        latex = format_results_latex(aggregated)
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"✓ LaTeX summary saved to: {output_file}")


if __name__ == '__main__':
    main()
