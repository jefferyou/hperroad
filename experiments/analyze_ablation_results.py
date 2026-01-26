#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验结果分析和可视化

功能：
1. 读取消融实验结果
2. 生成对比表格
3. 绘制性能对比图
4. 分析各组件的贡献
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 尝试导入matplotlib，如果没有则跳过绘图
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


class AblationAnalyzer:
    """消融实验结果分析器"""

    def __init__(self, results_file):
        self.results_file = results_file
        self.results = self._load_results()
        self.output_dir = os.path.dirname(results_file)

    def _load_results(self):
        """加载结果文件"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'r') as f:
            results = json.load(f)

        print(f"✓ Loaded {len(results)} experiments from {self.results_file}")
        return results

    def extract_metrics(self):
        """提取所有实验的评估指标"""
        metrics_data = []

        for config_id, result in self.results.items():
            if result['status'] != 'success':
                print(f"⚠ Skipping failed experiment: {config_id}")
                continue

            config = result['config']
            exp_result = result.get('result', {})

            # 提取下游任务的性能指标
            # 注意：具体的指标名称需要根据VecCity的返回格式调整
            row = {
                'ID': config_id,
                'Name': config['name'],
                'Model': config['model'],
                'Description': config['description'],
            }

            # 添加参数
            if 'params' in config:
                for k, v in config['params'].items():
                    row[k] = v

            # 添加结果（需要根据实际返回格式调整）
            if isinstance(exp_result, dict):
                row.update(exp_result)

            metrics_data.append(row)

        return pd.DataFrame(metrics_data)

    def generate_comparison_table(self):
        """生成对比表格"""
        df = self.extract_metrics()

        if df.empty:
            print("⚠ No successful experiments to compare")
            return None

        # 保存为CSV
        csv_file = os.path.join(self.output_dir, 'ablation_comparison.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ Comparison table saved to: {csv_file}")

        # 生成Markdown表格
        md_file = os.path.join(self.output_dir, 'ablation_comparison.md')
        with open(md_file, 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Performance Comparison\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Configuration Details\n\n")

            for config_id, result in self.results.items():
                if result['status'] == 'success':
                    config = result['config']
                    f.write(f"### {config['name']}\n\n")
                    f.write(f"- **ID**: {config_id}\n")
                    f.write(f"- **Model**: {config['model']}\n")
                    f.write(f"- **Description**: {config['description']}\n")
                    if 'params' in config:
                        f.write(f"- **Parameters**: {config['params']}\n")
                    f.write("\n")

        print(f"✓ Markdown report saved to: {md_file}")

        return df

    def analyze_component_contribution(self):
        """分析各组件的贡献"""
        df = self.extract_metrics()

        if df.empty:
            print("⚠ No data to analyze")
            return

        analysis_file = os.path.join(self.output_dir, 'component_analysis.txt')

        with open(analysis_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPONENT CONTRIBUTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # 假设有指标列（需要根据实际情况调整）
            metric_cols = [col for col in df.columns
                          if col not in ['ID', 'Name', 'Model', 'Description', 'lambda_ce', 'lambda_cc']]

            if not metric_cols:
                f.write("No metric columns found in results.\n")
                return

            for metric in metric_cols:
                f.write(f"\n{'-'*80}\n")
                f.write(f"Metric: {metric}\n")
                f.write(f"{'-'*80}\n\n")

                # 找到baseline和full model
                baseline_row = df[df['ID'] == 'baseline_hrnr']
                full_row = df[df['ID'] == 'full_model']

                if not baseline_row.empty and not full_row.empty and metric in baseline_row.columns:
                    baseline_val = baseline_row[metric].values[0]
                    full_val = full_row[metric].values[0]

                    try:
                        improvement = ((full_val - baseline_val) / baseline_val) * 100
                        f.write(f"Baseline (HRNR): {baseline_val}\n")
                        f.write(f"Full Model (HRNR_Hyperbolic): {full_val}\n")
                        f.write(f"Improvement: {improvement:.2f}%\n\n")
                    except:
                        pass

                # 分析各个消融版本
                f.write("Ablation variants:\n")
                for _, row in df.iterrows():
                    if row['ID'] not in ['baseline_hrnr', 'full_model'] and metric in row:
                        f.write(f"  - {row['Name']}: {row[metric]}\n")

                f.write("\n")

        print(f"✓ Component analysis saved to: {analysis_file}")

    def plot_results(self):
        """绘制结果对比图"""
        if not HAS_MATPLOTLIB:
            print("⚠ Matplotlib not available, skipping plots")
            return

        df = self.extract_metrics()

        if df.empty:
            print("⚠ No data to plot")
            return

        # 获取指标列
        metric_cols = [col for col in df.columns
                      if col not in ['ID', 'Name', 'Model', 'Description', 'lambda_ce', 'lambda_cc']]

        if not metric_cols:
            print("⚠ No metrics to plot")
            return

        # 为每个指标创建柱状图
        for metric in metric_cols:
            if metric not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            # 提取数据
            names = df['Name'].tolist()
            values = df[metric].tolist()

            # 创建颜色映射
            colors = []
            for name in names:
                if 'Baseline' in name:
                    colors.append('#e74c3c')  # 红色 - baseline
                elif 'Full' in name:
                    colors.append('#2ecc71')  # 绿色 - full model
                else:
                    colors.append('#3498db')  # 蓝色 - 消融版本

            # 绘制柱状图
            bars = ax.bar(range(len(names)), values, color=colors, alpha=0.7)

            # 设置标签
            ax.set_xlabel('Configuration', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'Ablation Study: {metric}', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9)

            # 添加网格
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()

            # 保存图片
            plot_file = os.path.join(self.output_dir, f'ablation_{metric}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ Plot saved to: {plot_file}")

    def generate_report(self):
        """生成完整的分析报告"""
        print("\n" + "=" * 80)
        print("GENERATING ABLATION ANALYSIS REPORT")
        print("=" * 80)

        # 1. 对比表格
        print("\n1. Generating comparison table...")
        df = self.generate_comparison_table()

        # 2. 组件贡献分析
        print("\n2. Analyzing component contributions...")
        self.analyze_component_contribution()

        # 3. 绘图
        print("\n3. Generating plots...")
        self.plot_results()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Analyze ablation study results')

    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to the ablation results JSON file')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建分析器
    analyzer = AblationAnalyzer(args.results_file)

    # 生成报告
    analyzer.generate_report()


if __name__ == '__main__':
    main()
