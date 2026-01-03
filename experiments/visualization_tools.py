#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNR_Hyperbolic 可视化和分析工具

提供功能：
1. 训练曲线可视化
2. 超参数调优结果分析
3. 双曲空间嵌入可视化
4. 消融实验结果对比
5. 多模型性能对比
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
sns.set_palette('husl')

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


class ExperimentVisualizer:
    """实验结果可视化工具"""

    def __init__(self, results_dir=None, output_dir=None):
        # 使用绝对路径
        if results_dir is None:
            results_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results')
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'experiments', 'figures')

        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_curves(self, log_file, save_path=None):
        """
        绘制训练曲线

        Args:
            log_file: 日志文件路径
            save_path: 保存路径
        """
        # 解析日志文件
        losses = {'train': [], 'struct': [], 'ce': [], 'cc': []}
        metrics = {'auc': [], 'f1': [], 'precision': [], 'recall': []}
        epochs = []

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # 解析损失
                    if 'loss:' in line:
                        parts = line.split('loss:')
                        if len(parts) > 1:
                            loss_str = parts[1].split(',')[0].strip()
                            try:
                                losses['train'].append(float(loss_str))
                            except:
                                pass

                    # 解析AUC
                    if 'auc:' in line:
                        parts = line.split('auc:')
                        if len(parts) > 1:
                            auc_str = parts[1].split()[0].strip()
                            try:
                                metrics['auc'].append(float(auc_str))
                            except:
                                pass

            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 损失曲线
            if losses['train']:
                axes[0, 0].plot(losses['train'], label='Total Loss', linewidth=2)
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss Curve')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # AUC曲线
            if metrics['auc']:
                axes[0, 1].plot(metrics['auc'], label='AUC', color='green', linewidth=2)
                axes[0, 1].set_xlabel('Evaluation Step')
                axes[0, 1].set_ylabel('AUC')
                axes[0, 1].set_title('AUC During Training')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path is None:
                save_path = os.path.join(self.output_dir, 'training_curves.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
            plt.close()

        except Exception as e:
            print(f"Error plotting training curves: {e}")

    def plot_hyperparameter_importance(self, tuning_result_file, save_path=None):
        """
        绘制超参数重要性分析

        Args:
            tuning_result_file: 超参数调优结果文件
            save_path: 保存路径
        """
        with open(tuning_result_file, 'r') as f:
            results = json.load(f)

        trials = results['all_trials']
        metric = results['metric']

        # 提取超参数和分数
        param_names = list(results['search_space'].keys())
        param_scores = {name: [] for name in param_names}

        for trial in trials:
            if 'error' not in trial:
                score = trial['score']
                for param in param_names:
                    if param in trial['hyperparams']:
                        value = trial['hyperparams'][param]
                        param_scores[param].append((value, score))

        # 计算每个超参数的影响
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, param in enumerate(param_names):
            if i >= len(axes):
                break

            if not param_scores[param]:
                continue

            values, scores = zip(*param_scores[param])

            # 根据值类型选择绘图方式
            if isinstance(values[0], (int, float)):
                # 散点图
                axes[i].scatter(values, scores, alpha=0.6)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{param} vs {metric}')
                axes[i].grid(True, alpha=0.3)
            else:
                # 箱线图
                unique_values = list(set(values))
                grouped_scores = {v: [] for v in unique_values}
                for v, s in zip(values, scores):
                    grouped_scores[v].append(s)

                axes[i].boxplot([grouped_scores[v] for v in unique_values],
                               labels=[str(v) for v in unique_values])
                axes[i].set_xlabel(param)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{param} Distribution')
                axes[i].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(param_names), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'hyperparameter_importance.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter importance plot saved to {save_path}")
        plt.close()

    def plot_ablation_study(self, ablation_result_file, save_path=None):
        """
        绘制消融实验结果

        Args:
            ablation_result_file: 消融实验结果文件
            save_path: 保存路径
        """
        with open(ablation_result_file, 'r') as f:
            results = json.load(f)

        # 提取配置和结果
        configs = []
        aucs = []
        f1s = []

        for item in results['results']:
            configs.append(item['config']['name'])
            result = item['result']
            aucs.append(result.get('auc', 0))
            f1s.append(result.get('f1', 0))

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(configs))
        width = 0.35

        # AUC对比
        bars1 = axes[0].bar(x, aucs, width, label='AUC', color='skyblue')
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('AUC')
        axes[0].set_title('Ablation Study - AUC Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')

        # F1对比
        bars2 = axes[1].bar(x, f1s, width, label='F1', color='lightcoral')
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('F1')
        axes[1].set_title('Ablation Study - F1 Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(configs, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'ablation_study.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation study plot saved to {save_path}")
        plt.close()

    def plot_model_comparison(self, comparison_result_file, save_path=None):
        """
        绘制模型对比结果

        Args:
            comparison_result_file: 模型对比结果文件
            save_path: 保存路径
        """
        with open(comparison_result_file, 'r') as f:
            results = json.load(f)

        models = results['models']
        model_results = results['results']

        # 提取指标
        metrics = ['auc', 'f1', 'precision', 'recall']
        metric_values = {metric: [] for metric in metrics}

        for model in models:
            result = model_results.get(model, {})
            for metric in metrics:
                metric_values[metric].append(result.get(metric, 0))

        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            x = np.arange(len(models))
            bars = axes[i].bar(x, metric_values[metric], color=['steelblue', 'coral'][:len(models)])

            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(models, rotation=15)
            axes[i].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'model_comparison.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
        plt.close()

    def plot_embedding_pca(self, embedding_file, labels=None, save_path=None):
        """
        使用PCA可视化双曲嵌入

        Args:
            embedding_file: 嵌入文件路径（.npy）
            labels: 节点标签
            save_path: 保存路径
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("sklearn not installed, skipping PCA visualization")
            return

        # 加载嵌入
        embeddings = np.load(embedding_file)

        # 只取空间部分（忽略时间分量）
        if embeddings.shape[1] > 2:
            spatial_embeddings = embeddings[:, 1:]  # 排除第一个时间分量
        else:
            spatial_embeddings = embeddings

        # PCA降维到2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(spatial_embeddings)

        # 绘图
        plt.figure(figsize=(10, 8))

        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       alpha=0.6, s=10, color='steelblue')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Hyperbolic Embeddings Visualization (PCA)')
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'embedding_pca.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding PCA plot saved to {save_path}")
        plt.close()

    def generate_experiment_report(self, experiment_results, save_path=None):
        """
        生成实验报告（Markdown格式）

        Args:
            experiment_results: 实验结果字典
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'experiment_report.md')

        report = []
        report.append("# HRNR_Hyperbolic 实验报告\n")
        report.append(f"生成时间: {experiment_results.get('timestamp', 'N/A')}\n")
        report.append("\n## 实验配置\n")

        config = experiment_results.get('config', {})
        report.append("| 参数 | 值 |\n")
        report.append("|------|----|\n")
        for key, value in config.items():
            report.append(f"| {key} | {value} |\n")

        report.append("\n## 实验结果\n")

        results = experiment_results.get('results', {})
        report.append("| 指标 | 值 |\n")
        report.append("|------|----|\n")
        for key, value in results.items():
            report.append(f"| {key} | {value:.4f} |\n")

        report.append("\n## 总结\n")
        report.append(f"- 最佳AUC: {results.get('auc', 0):.4f}\n")
        report.append(f"- 最佳F1: {results.get('f1', 0):.4f}\n")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)

        print(f"Experiment report saved to {save_path}")


def main():
    """示例使用"""
    visualizer = ExperimentVisualizer()

    # 示例：可视化训练曲线
    # visualizer.plot_training_curves('path/to/log/file.log')

    # 示例：可视化超参数重要性
    # visualizer.plot_hyperparameter_importance('path/to/tuning/results.json')

    # 示例：可视化消融实验
    # visualizer.plot_ablation_study('path/to/ablation/results.json')

    # 示例：可视化模型对比
    # visualizer.plot_model_comparison('path/to/comparison/results.json')

    # 示例：可视化嵌入
    # visualizer.plot_embedding_pca('path/to/embeddings.npy')

    print("Visualization tools ready!")
    print("Use the ExperimentVisualizer class to create plots.")


if __name__ == '__main__':
    main()
