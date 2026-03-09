#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HHRoad (HRNR_Hyperbolic) 论文可视化工具集
============================================

为HHRoad论文生成高质量的学术可视化图表，包括：

1. 主实验结果对比表（Table）  - 多城市、多基线、多下游任务
2. 消融实验柱状图             - 各组件贡献分析
3. 双曲嵌入空间可视化         - Poincaré disk投影 + 层次结构
4. 超参数敏感性分析           - lambda_ce, lambda_cc, temperature, dim
5. 训练收敛曲线               - Loss分解 + 下游指标变化
6. 层次结构可视化             - Segment-Locality-Region 蕴含关系
7. 嵌入质量分析               - 距离分布、范数分布对比
8. 雷达图                     - 多维度性能对比

使用方式：
    python visualize_hhroad_results.py --mode all
    python visualize_hhroad_results.py --mode table
    python visualize_hhroad_results.py --mode ablation
    ...

所有图表以300dpi PDF/PNG保存，适合直接插入论文。
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局样式设置 - 学术论文标准
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 配色方案
COLORS = {
    'hhroad': '#E63946',       # 红色 - 我们的方法
    'hrnr': '#457B9D',         # 蓝色 - 基线HRNR
    'hyperroad': '#2A9D8F',    # 绿色
    'deepwalk': '#E9C46A',     # 黄色
    'node2vec': '#F4A261',     # 橙色
    'line': '#264653',         # 深蓝
    'gat': '#A8DADC',          # 浅蓝
    'chebconv': '#6A4C93',     # 紫色
    'geomgcn': '#1982C4',      # 蓝色
    'sarn': '#8AC926',         # 绿色
    'toast': '#FF595E',        # 红
    'start': '#FFCA3A',        # 黄
    'jclrnt': '#6A994E',       # 深绿
    'srn2vec': '#BC4749',      # 暗红
    'trajrne': '#A06CD5',      # 淡紫
}

MODEL_DISPLAY_NAMES = {
    'HRNR_Hyperbolic': 'HHRoad (Ours)',
    'HRNR': 'HRNR',
    'HyperRoad': 'HyperRoad',
    'DeepWalk': 'DeepWalk',
    'Node2Vec': 'Node2Vec',
    'LINE': 'LINE',
    'GAT': 'GAT',
    'ChebConv': 'ChebConv',
    'GeomGCN': 'GeomGCN',
    'SARN': 'SARN',
    'Toast': 'Toast',
    'START': 'START',
    'JCLRNT': 'JCLRNT',
    'SRN2Vec': 'SRN2Vec',
    'TrajRNE': 'TrajRNE',
}

CITY_NAMES = {
    'xa': "Xi'an",
    'bj': 'Beijing',
    'cd': 'Chengdu',
    'sz': 'Shenzhen',
    'sh': 'Shanghai',
}

TASK_DISPLAY = {
    'tsi': 'Speed Inference',
    'tte': 'Travel Time Est.',
    'sts': 'Similarity Search',
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 主实验结果对比表 (LaTeX Table)
# ============================================================
def generate_main_results_table(results_data=None, output_path=None):
    """
    生成论文主表：多城市 x 多模型 x 多下游任务。

    results_data: dict，格式如下：
    {
        "xa": {
            "HRNR_Hyperbolic": {"tsi_mae": 1.23, "tsi_rmse": 2.34, "tte_mae": ..., "sts_hr10": ..., "sts_hr50": ...},
            "HRNR": {...},
            ...
        },
        "bj": {...},
        ...
    }
    如果results_data为None，使用示例数据。
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'main_results_table.tex')

    if results_data is None:
        results_data = _get_example_main_results()

    cities = list(results_data.keys())
    models = list(next(iter(results_data.values())).keys())

    # 下游任务指标
    metrics = [
        ('tsi_mae', 'MAE$\\downarrow$'),
        ('tsi_rmse', 'RMSE$\\downarrow$'),
        ('tte_mae', 'MAE$\\downarrow$'),
        ('tte_rmse', 'RMSE$\\downarrow$'),
        ('tte_mape', 'MAPE$\\downarrow$'),
        ('sts_hr10', 'HR@10$\\uparrow$'),
        ('sts_hr50', 'HR@50$\\uparrow$'),
    ]

    # 找出每列最优值
    def find_best(city_data, metric_key):
        is_lower_better = 'mae' in metric_key or 'rmse' in metric_key or 'mape' in metric_key
        vals = {m: city_data[m].get(metric_key, float('inf') if is_lower_better else 0)
                for m in city_data}
        if is_lower_better:
            return min(vals, key=vals.get)
        else:
            return max(vals, key=vals.get)

    # 生成LaTeX
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Performance comparison on three downstream tasks across multiple cities. "
                  "\\textbf{Bold} indicates the best result. $\\downarrow$ means lower is better, $\\uparrow$ means higher is better.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\resizebox{\\textwidth}{!}{")

    ncols = 1 + len(metrics)  # Model + metrics
    col_spec = "l" + "|" + "cc" + "|" + "ccc" + "|" + "cc"
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    # Header row 1: task groups
    lines.append("& \\multicolumn{2}{c|}{\\textbf{Speed Inference}} "
                  "& \\multicolumn{3}{c|}{\\textbf{Travel Time Est.}} "
                  "& \\multicolumn{2}{c}{\\textbf{Similarity Search}} \\\\")

    # Header row 2: metric names
    header2 = "\\textbf{Model}"
    for _, display in metrics:
        header2 += f" & {display}"
    header2 += " \\\\"
    lines.append(header2)

    for city in cities:
        city_data = results_data[city]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{CITY_NAMES.get(city, city)}}}}} \\\\")
        lines.append("\\midrule")

        for model in models:
            if model not in city_data:
                continue
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            if model == 'HRNR_Hyperbolic':
                display_name = '\\textbf{' + display_name + '}'

            row = display_name
            for metric_key, _ in metrics:
                val = city_data[model].get(metric_key, None)
                best_model = find_best(city_data, metric_key)
                if val is not None:
                    val_str = f"{val:.4f}" if val < 1 else f"{val:.2f}"
                    if model == best_model:
                        val_str = '\\textbf{' + val_str + '}'
                    row += f" & {val_str}"
                else:
                    row += " & -"
            row += " \\\\"
            lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[Table] Main results table saved to: {output_path}")

    # Also generate a visual heatmap version
    _plot_results_heatmap(results_data, metrics)


def _plot_results_heatmap(results_data, metrics):
    """生成性能对比热力图"""
    cities = list(results_data.keys())

    for city in cities:
        city_data = results_data[city]
        models = list(city_data.keys())
        metric_keys = [m[0] for m in metrics]
        metric_labels = [m[1].replace('$\\downarrow$', '↓').replace('$\\uparrow$', '↑')
                         for m in metrics]

        data_matrix = []
        for model in models:
            row = [city_data[model].get(mk, 0) for mk in metric_keys]
            data_matrix.append(row)

        df = pd.DataFrame(data_matrix,
                           index=[MODEL_DISPLAY_NAMES.get(m, m) for m in models],
                           columns=metric_labels)

        # Normalize each column for color mapping (rank-based)
        df_rank = df.rank(axis=0)
        # For lower-is-better metrics, invert ranks
        for i, mk in enumerate(metric_keys):
            if 'mae' in mk or 'rmse' in mk or 'mape' in mk:
                df_rank.iloc[:, i] = df_rank.iloc[:, i].max() + 1 - df_rank.iloc[:, i]

        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.45)))
        sns.heatmap(df_rank, annot=df.values, fmt='.3f', cmap='RdYlGn',
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'Rank (higher = better)'},
                    annot_kws={'size': 8})
        ax.set_title(f"Performance Comparison — {CITY_NAMES.get(city, city)}", fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        save_path = os.path.join(OUTPUT_DIR, f'heatmap_{city}.pdf')
        plt.savefig(save_path)
        plt.close()
        print(f"[Heatmap] {city} saved to: {save_path}")


# ============================================================
# 2. 消融实验可视化
# ============================================================
def plot_ablation_study(ablation_data=None, output_path=None):
    """
    消融实验柱状图。

    ablation_data: dict 格式：
    {
        "Full Model": {"tsi_mae": ..., "tte_mae": ..., "sts_hr10": ...},
        "w/o Entailment": {...},
        "w/o Contrastive": {...},
        "w/o Both Aux.": {...},
        "Euclidean (HRNR)": {...},
    }
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'ablation_study.pdf')

    if ablation_data is None:
        ablation_data = _get_example_ablation_data()

    configs = list(ablation_data.keys())
    # Select representative metrics
    metrics_to_plot = [
        ('tsi_mae', 'TSI MAE ↓', True),
        ('tte_mae', 'TTE MAE ↓', True),
        ('sts_hr10', 'STS HR@10 ↑', False),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    colors_ablation = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#6A4C93']
    hatches = ['', '//', '\\\\', 'xx', '..']

    for idx, (metric_key, metric_label, lower_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [ablation_data[c].get(metric_key, 0) for c in configs]
        x = np.arange(len(configs))

        bars = ax.bar(x, values, color=colors_ablation[:len(configs)],
                      edgecolor='black', linewidth=0.5, alpha=0.85,
                      hatch=[hatches[i % len(hatches)] for i in range(len(configs))])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=35, ha='right', fontsize=8)
        ax.set_title(metric_label, fontweight='bold', fontsize=11)

        # Highlight best
        if lower_better:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('#E63946')
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Ablation] Saved to: {output_path}")


# ============================================================
# 3. 双曲嵌入空间可视化 (Poincaré Disk)
# ============================================================
def plot_hyperbolic_embedding(embedding_file=None, output_path=None,
                               labels=None, hierarchy_info=None):
    """
    将Lorentz模型嵌入可视化到Poincaré圆盘上。
    支持：
    - 按标签着色
    - 显示层次结构（Segment-Locality-Region）
    - 距离原点的径向分布

    embedding_file: .npy文件路径 (N x d+1 Lorentz嵌入)
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'hyperbolic_embedding.pdf')

    if embedding_file is not None and os.path.exists(embedding_file):
        embeddings = np.load(embedding_file)
        print(f"Loaded embeddings: shape={embeddings.shape}")
    else:
        # Generate synthetic demo data
        print("No embedding file found, generating synthetic demo data...")
        embeddings = _generate_synthetic_lorentz_embeddings(n=2000, dim=32)

    # Convert Lorentz to Poincaré ball (2D projection)
    poincare_2d = _lorentz_to_poincare_2d(embeddings)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

    # --- Panel (a): Poincaré disk embedding ---
    ax1 = fig.add_subplot(gs[0])
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=1.5, linestyle='-')
    ax1.add_patch(circle)

    # Color by distance from origin (hierarchy level proxy)
    norms = np.sqrt(poincare_2d[:, 0] ** 2 + poincare_2d[:, 1] ** 2)
    scatter = ax1.scatter(poincare_2d[:, 0], poincare_2d[:, 1],
                           c=norms, cmap='viridis', s=3, alpha=0.6, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.7, label='Distance from origin')
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('(a) Poincaré Disk Projection', fontweight='bold')

    # Add origin marker
    ax1.plot(0, 0, 'r*', markersize=12, markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax1.annotate('Origin\n(Region level)', xy=(0, 0), xytext=(0.3, 0.8),
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                  fontsize=8, color='red', fontweight='bold')

    # --- Panel (b): Radial distribution ---
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(norms, bins=50, color='#457B9D', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(np.median(norms), color='#E63946', linestyle='--', linewidth=2,
                label=f'Median = {np.median(norms):.3f}')
    ax2.set_xlabel('Poincaré Norm $\\|\\mathbf{x}\\|$')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) Radial Distribution', fontweight='bold')
    ax2.legend()

    # --- Panel (c): Lorentz time component distribution ---
    ax3 = fig.add_subplot(gs[2])
    t_component = embeddings[:, 0]  # Time component
    ax3.hist(t_component, bins=50, color='#2A9D8F', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax3.axvline(np.median(t_component), color='#E63946', linestyle='--', linewidth=2,
                label=f'Median = {np.median(t_component):.2f}')
    ax3.set_xlabel('Time Component $x_0$')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Lorentz Time Component', fontweight='bold')
    ax3.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"[Embedding] Saved to: {output_path}")


def _lorentz_to_poincare_2d(lorentz_emb):
    """
    将Lorentz嵌入 (N, d+1) 转换为Poincaré球坐标，
    然后PCA降至2D。

    Lorentz -> Poincaré: x_P = x_{1:d} / (x_0 + 1)
    """
    from sklearn.decomposition import PCA

    t = lorentz_emb[:, 0]  # time component
    x = lorentz_emb[:, 1:]  # spatial components

    # Lorentz to Poincaré
    denom = t + 1.0
    denom = np.clip(denom, a_min=1e-8, a_max=None)
    poincare = x / denom[:, np.newaxis]

    # Clip to unit ball
    norms = np.linalg.norm(poincare, axis=1, keepdims=True)
    too_large = norms > 0.99
    poincare = np.where(too_large, poincare * 0.99 / norms, poincare)

    # PCA to 2D
    if poincare.shape[1] > 2:
        pca = PCA(n_components=2)
        poincare_2d = pca.fit_transform(poincare)
        # Rescale to stay within unit disk
        max_norm = np.max(np.linalg.norm(poincare_2d, axis=1))
        if max_norm > 0:
            poincare_2d = poincare_2d / max_norm * 0.95
    else:
        poincare_2d = poincare[:, :2]

    return poincare_2d


def _generate_synthetic_lorentz_embeddings(n=2000, dim=32):
    """Generate synthetic Lorentz embeddings with hierarchical structure."""
    np.random.seed(42)
    spatial = np.random.randn(n, dim) * 0.5

    # Create hierarchy: some points closer to origin (region), some farther (segment)
    # Exponential distribution of norms to mimic hierarchy
    scales = np.random.exponential(scale=1.0, size=n)
    spatial = spatial * scales[:, np.newaxis]

    # Compute time component: t = sqrt(1 + ||x||^2)
    spatial_norm_sq = np.sum(spatial ** 2, axis=1, keepdims=True)
    t = np.sqrt(1.0 + spatial_norm_sq)

    lorentz = np.concatenate([t, spatial], axis=1)
    return lorentz


# ============================================================
# 4. 超参数敏感性分析
# ============================================================
def plot_hyperparameter_sensitivity(sensitivity_data=None, output_path=None):
    """
    绘制超参数敏感性分析图。

    sensitivity_data: dict 格式：
    {
        "lambda_ce": {"values": [0.01, 0.05, 0.1, 0.15, 0.2], "tsi_mae": [...], "tte_mae": [...]},
        "lambda_cc": {...},
        "temperature": {...},
        "hyperbolic_dim": {...},
    }
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'hyperparameter_sensitivity.pdf')

    if sensitivity_data is None:
        sensitivity_data = _get_example_sensitivity_data()

    params = list(sensitivity_data.keys())
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3.5))
    if n_params == 1:
        axes = [axes]

    param_display = {
        'lambda_ce': '$\\lambda_{ce}$',
        'lambda_cc': '$\\lambda_{cc}$',
        'temperature': '$\\tau$',
        'hyperbolic_dim': '$d_h$',
    }

    for idx, param in enumerate(params):
        ax = axes[idx]
        data = sensitivity_data[param]
        x_vals = data['values']

        # Plot multiple metrics
        for metric_key, metric_label, color, marker in [
            ('tsi_mae', 'TSI MAE', '#E63946', 'o'),
            ('tte_mae', 'TTE MAE', '#457B9D', 's'),
        ]:
            if metric_key in data:
                ax.plot(x_vals, data[metric_key], marker=marker, color=color,
                        label=metric_label, linewidth=2, markersize=6,
                        markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xlabel(param_display.get(param, param), fontweight='bold', fontsize=12)
        if idx == 0:
            ax.set_ylabel('MAE ↓', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_title(f'Sensitivity to {param_display.get(param, param)}',
                      fontweight='bold', fontsize=10)

        # Mark optimal
        if 'tsi_mae' in data:
            best_idx = np.argmin(data['tsi_mae'])
            ax.axvline(x_vals[best_idx], color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Sensitivity] Saved to: {output_path}")


# ============================================================
# 5. 训练收敛曲线
# ============================================================
def plot_training_curves(training_log=None, output_path=None):
    """
    绘制训练过程中的Loss分解和下游指标变化。

    training_log: dict 格式：
    {
        "steps": [0, 20, 40, ...],
        "loss_total": [...],
        "loss_struct": [...],
        "loss_ce": [...],
        "loss_cc": [...],
        "auc": [...],
        "f1": [...],
    }
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'training_curves.pdf')

    if training_log is None:
        training_log = _get_example_training_log()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    steps = training_log['steps']

    # Panel (a): Total loss
    ax = axes[0]
    ax.plot(steps, training_log['loss_total'], color='#E63946', linewidth=2, label='Total Loss')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Total Loss', fontweight='bold')
    ax.legend()

    # Panel (b): Loss decomposition
    ax = axes[1]
    ax.plot(steps, training_log['loss_struct'], color='#457B9D', linewidth=1.5,
            label='$\\mathcal{L}_{struct}$', alpha=0.8)
    ax.plot(steps, training_log['loss_ce'], color='#2A9D8F', linewidth=1.5,
            label='$\\lambda_{ce} \\cdot \\mathcal{L}_{ent}$', alpha=0.8)
    ax.plot(steps, training_log['loss_cc'], color='#E9C46A', linewidth=1.5,
            label='$\\lambda_{cc} \\cdot \\mathcal{L}_{con}$', alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Component')
    ax.set_title('(b) Loss Decomposition', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel (c): AUC and F1
    ax = axes[2]
    ax.plot(steps, training_log['auc'], color='#E63946', linewidth=2,
            label='AUC', marker='o', markersize=3)
    ax.plot(steps, training_log['f1'], color='#457B9D', linewidth=2,
            label='F1', marker='s', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Score')
    ax.set_title('(c) Evaluation Metrics', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Training] Saved to: {output_path}")


# ============================================================
# 6. 层次结构蕴含关系可视化
# ============================================================
def plot_hierarchy_visualization(output_path=None):
    """
    可视化Segment-Locality-Region三层次结构和蕴含锥。
    生成示意图展示：
    - 双曲空间中层次嵌入的相对位置
    - 蕴含锥的打开角度
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'hierarchy_visualization.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): Poincaré disk with hierarchy ---
    ax = axes[0]
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Region nodes (near center)
    np.random.seed(42)
    n_region = 5
    n_locality = 20
    n_segment = 100

    region_pts = np.random.randn(n_region, 2) * 0.08
    locality_pts = np.random.randn(n_locality, 2) * 0.3
    segment_pts = np.random.randn(n_segment, 2) * 0.6

    # Clip to disk
    for pts in [region_pts, locality_pts, segment_pts]:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        too_large = norms > 0.95
        pts[:] = np.where(too_large, pts * 0.95 / norms, pts)

    ax.scatter(segment_pts[:, 0], segment_pts[:, 1], s=8, c='#A8DADC',
               alpha=0.5, label=f'Segment (n={n_segment})', zorder=2)
    ax.scatter(locality_pts[:, 0], locality_pts[:, 1], s=40, c='#457B9D',
               alpha=0.8, label=f'Locality (n={n_locality})', zorder=3,
               edgecolors='black', linewidth=0.5)
    ax.scatter(region_pts[:, 0], region_pts[:, 1], s=120, c='#E63946',
               alpha=0.9, label=f'Region (n={n_region})', zorder=4,
               edgecolors='black', linewidth=1, marker='D')

    # Draw entailment cones for a few region points
    for i in range(min(3, n_region)):
        _draw_cone(ax, region_pts[i], angle=40, length=0.5, color='#E63946')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('(a) Hierarchical Embeddings\nin Poincaré Disk', fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # --- Panel (b): Entailment cone schematic ---
    ax = axes[1]
    circle2 = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax.add_patch(circle2)

    # Draw a single entailment cone example
    parent = np.array([0.1, 0.15])
    children_in = np.array([[0.4, 0.5], [0.35, 0.45], [0.5, 0.55]])
    child_out = np.array([[-0.3, 0.6]])

    # Parent point
    ax.plot(parent[0], parent[1], 'D', color='#E63946', markersize=12,
            markeredgecolor='black', markeredgewidth=1, zorder=5)
    ax.annotate('Parent\n(Region)', xy=parent, xytext=(-0.4, -0.3),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                fontsize=9, color='#E63946', fontweight='bold')

    # Entailment cone
    _draw_cone(ax, parent, angle=35, length=0.8, color='#E63946', alpha=0.15, fill=True)

    # Children inside cone
    ax.scatter(children_in[:, 0], children_in[:, 1], s=60, c='#2A9D8F',
               edgecolors='black', linewidth=0.5, zorder=4, marker='o')
    for ci in children_in:
        ax.annotate('', xy=ci, xytext=parent,
                      arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=0.8, alpha=0.5))

    # Child outside cone
    ax.scatter(child_out[:, 0], child_out[:, 1], s=60, c='#E9C46A',
               edgecolors='black', linewidth=0.5, zorder=4, marker='X')
    ax.annotate('Outside cone\n(not entailed)', xy=child_out[0], xytext=(-0.6, 0.85),
                arrowprops=dict(arrowstyle='->', color='#E9C46A', lw=1.5),
                fontsize=8, color='#E9C46A', fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#E63946',
               markersize=10, label='Parent (wider cone)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2A9D8F',
               markersize=8, label='Child (in cone, entailed)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#E9C46A',
               markersize=8, label='Node (outside cone)'),
        mpatches.Patch(color='#E63946', alpha=0.15, label='Entailment cone'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_title('(b) Entailment Cone\nin Hyperbolic Space', fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Hierarchy] Saved to: {output_path}")


def _draw_cone(ax, apex, angle=30, length=0.5, color='red', alpha=0.2, fill=False):
    """Draw an entailment cone on the Poincaré disk."""
    angle_rad = np.radians(angle)
    # Direction from origin through apex
    apex_angle = np.arctan2(apex[1], apex[0])

    theta1 = apex_angle - angle_rad
    theta2 = apex_angle + angle_rad

    # Draw cone edges
    for theta in [theta1, theta2]:
        end = apex + length * np.array([np.cos(theta), np.sin(theta)])
        ax.plot([apex[0], end[0]], [apex[1], end[1]], color=color, linewidth=1, alpha=0.6)

    if fill:
        # Fill the cone area
        thetas = np.linspace(theta1, theta2, 30)
        pts = [apex]
        for t in thetas:
            pts.append(apex + length * np.array([np.cos(t), np.sin(t)]))
        pts.append(apex)
        pts = np.array(pts)
        from matplotlib.patches import Polygon
        cone = Polygon(pts, closed=True, color=color, alpha=alpha)
        ax.add_patch(cone)


# ============================================================
# 7. 嵌入质量分析 - Euclidean vs Hyperbolic
# ============================================================
def plot_embedding_quality_comparison(hyp_emb_file=None, euc_emb_file=None, output_path=None):
    """
    对比欧氏嵌入和双曲嵌入的质量。
    - 距离分布对比
    - 范数分布对比
    - 邻域保持度
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'embedding_quality.pdf')

    # Load or generate data
    if hyp_emb_file and os.path.exists(hyp_emb_file):
        hyp_emb = np.load(hyp_emb_file)
    else:
        hyp_emb = _generate_synthetic_lorentz_embeddings(n=1000, dim=32)

    if euc_emb_file and os.path.exists(euc_emb_file):
        euc_emb = np.load(euc_emb_file)
    else:
        np.random.seed(123)
        euc_emb = np.random.randn(hyp_emb.shape[0], hyp_emb.shape[1] - 1) * 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel (a): Distance distribution
    ax = axes[0]
    n_sample = min(500, hyp_emb.shape[0])
    idx = np.random.choice(hyp_emb.shape[0], n_sample, replace=False)

    # Euclidean pairwise distances (sample)
    euc_sample = euc_emb[idx[:100]]
    from sklearn.metrics import pairwise_distances
    euc_dists = pairwise_distances(euc_sample).flatten()
    euc_dists = euc_dists[euc_dists > 0]

    # Hyperbolic pairwise distances (sample spatial part)
    hyp_spatial = hyp_emb[idx[:100], 1:]
    hyp_dists = pairwise_distances(hyp_spatial).flatten()
    hyp_dists = hyp_dists[hyp_dists > 0]

    ax.hist(euc_dists, bins=50, alpha=0.6, color='#457B9D', label='Euclidean', density=True)
    ax.hist(hyp_dists, bins=50, alpha=0.6, color='#E63946', label='Hyperbolic', density=True)
    ax.set_xlabel('Pairwise Distance')
    ax.set_ylabel('Density')
    ax.set_title('(a) Distance Distribution', fontweight='bold')
    ax.legend()

    # Panel (b): Norm distribution
    ax = axes[1]
    euc_norms = np.linalg.norm(euc_emb, axis=1)
    hyp_spatial_norms = np.linalg.norm(hyp_emb[:, 1:], axis=1)
    ax.hist(euc_norms, bins=50, alpha=0.6, color='#457B9D', label='Euclidean', density=True)
    ax.hist(hyp_spatial_norms, bins=50, alpha=0.6, color='#E63946', label='Hyperbolic', density=True)
    ax.set_xlabel('Embedding Norm $\\|\\mathbf{x}\\|$')
    ax.set_ylabel('Density')
    ax.set_title('(b) Norm Distribution', fontweight='bold')
    ax.legend()

    # Panel (c): Variance explained by top-k PCA components
    ax = axes[2]
    from sklearn.decomposition import PCA
    max_k = min(20, euc_emb.shape[1], hyp_emb.shape[1] - 1)

    pca_euc = PCA(n_components=max_k).fit(euc_emb[:, :max_k] if euc_emb.shape[1] > max_k else euc_emb)
    pca_hyp = PCA(n_components=max_k).fit(hyp_emb[:, 1:max_k+1] if hyp_emb.shape[1] - 1 > max_k else hyp_emb[:, 1:])

    ax.plot(range(1, max_k + 1), np.cumsum(pca_euc.explained_variance_ratio_),
            'o-', color='#457B9D', label='Euclidean', linewidth=2, markersize=4)
    ax.plot(range(1, max_k + 1), np.cumsum(pca_hyp.explained_variance_ratio_),
            's-', color='#E63946', label='Hyperbolic', linewidth=2, markersize=4)
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_title('(c) Information Concentration', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Quality] Saved to: {output_path}")


# ============================================================
# 8. 雷达图 - 多维度性能对比
# ============================================================
def plot_radar_chart(radar_data=None, output_path=None):
    """
    雷达图展示HHRoad vs 基线在多个维度上的对比。

    radar_data: dict 格式：
    {
        "HHRoad": {"TSI": 0.9, "TTE": 0.85, "STS": 0.92, "Hierarchy": 0.88, "Scalability": 0.8},
        "HRNR": {"TSI": 0.82, ...},
        "HyperRoad": {...},
    }
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'radar_chart.pdf')

    if radar_data is None:
        radar_data = _get_example_radar_data()

    categories = list(next(iter(radar_data.values())).keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors_radar = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#6A4C93']
    linestyles = ['-', '--', '-.', ':', '-']

    for idx, (model, values) in enumerate(radar_data.items()):
        vals = [values[cat] for cat in categories]
        vals += vals[:1]
        color = colors_radar[idx % len(colors_radar)]
        ls = linestyles[idx % len(linestyles)]

        ax.plot(angles, vals, color=color, linewidth=2 if idx == 0 else 1.5,
                linestyle=ls, label=model)
        ax.fill(angles, vals, color=color, alpha=0.08 if idx == 0 else 0.03)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax.set_title('Multi-dimensional Performance Comparison', fontweight='bold',
                  pad=20, fontsize=12)

    plt.savefig(output_path)
    plt.close()
    print(f"[Radar] Saved to: {output_path}")


# ============================================================
# 9. 多城市性能对比条形图
# ============================================================
def plot_multi_city_comparison(results_data=None, output_path=None):
    """
    分组柱状图：多个城市上的核心指标对比。
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'multi_city_comparison.pdf')

    if results_data is None:
        results_data = _get_example_main_results()

    cities = list(results_data.keys())
    # Select key models for visual clarity
    key_models = ['HRNR_Hyperbolic', 'HRNR', 'HyperRoad', 'START', 'JCLRNT', 'Toast']
    available_models = [m for m in key_models if m in next(iter(results_data.values()))]

    metrics = [
        ('tsi_mae', 'Speed Inference\nMAE ↓'),
        ('tte_mae', 'Travel Time Est.\nMAE ↓'),
        ('sts_hr10', 'Similarity Search\nHR@10 ↑'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for m_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[m_idx]
        x = np.arange(len(cities))
        width = 0.8 / len(available_models)

        for i, model in enumerate(available_models):
            vals = [results_data[city].get(model, {}).get(metric_key, 0) for city in cities]
            offset = (i - len(available_models) / 2 + 0.5) * width
            color = COLORS.get(model.lower(), f'C{i}')
            display_name = MODEL_DISPLAY_NAMES.get(model, model)

            bars = ax.bar(x + offset, vals, width * 0.9, label=display_name if m_idx == 0 else '',
                           color=color, edgecolor='black', linewidth=0.3, alpha=0.85)

        ax.set_xlabel('City', fontweight='bold')
        ax.set_ylabel(metric_label.split('\n')[0], fontweight='bold')
        ax.set_title(metric_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CITY_NAMES.get(c, c) for c in cities])

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(6, len(available_models)),
               bbox_to_anchor=(0.5, 1.08), fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[MultiCity] Saved to: {output_path}")


# ============================================================
# 10. 模型架构示意图 (文字版)
# ============================================================
def plot_model_architecture(output_path=None):
    """
    绘制HHRoad模型架构概述图。
    展示：特征输入 -> 欧氏嵌入 -> 双曲映射 -> 层次图卷积 -> 损失函数
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'model_architecture.pdf')

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define boxes
    boxes = [
        (0.5, 2.5, 2.5, 1.5, 'Road\nFeatures\n(lane, type,\nlength, node)', '#A8DADC'),
        (3.5, 2.5, 2.0, 1.5, 'Euclidean\nEmbedding\n$\\mathbb{R}^d$', '#457B9D'),
        (6.0, 2.5, 2.5, 1.5, 'Hyperbolic\nMapping\n$\\mathbb{H}^d$ (Lorentz)', '#E63946'),
        (9.0, 2.5, 2.5, 1.5, 'Hierarchical\nGraph Conv.\n(S→L→R→L→S)', '#2A9D8F'),
        (12.0, 2.5, 2.0, 1.5, 'Road\nRepresentation\n$\\mathbf{h} \\in \\mathbb{R}^d$', '#E9C46A'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                         facecolor=color, edgecolor='black',
                                         linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white' if color in ['#457B9D', '#E63946', '#2A9D8F'] else 'black')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    connections = [(3.0, 3.25, 3.5, 3.25),
                   (5.5, 3.25, 6.0, 3.25),
                   (8.5, 3.25, 9.0, 3.25),
                   (11.5, 3.25, 12.0, 3.25)]

    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)

    # Loss functions (below)
    loss_boxes = [
        (6.5, 0.5, 2.0, 1.2, '$\\mathcal{L}_{struct}$\nCross-Entropy', '#457B9D'),
        (9.0, 0.5, 2.0, 1.2, '$\\mathcal{L}_{ent}$\nEntailment Cone', '#E63946'),
        (11.5, 0.5, 2.0, 1.2, '$\\mathcal{L}_{con}$\nContrastive', '#2A9D8F'),
    ]

    for x, y, w, h, text, color in loss_boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                         facecolor=color, edgecolor='black',
                                         linewidth=1, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # Arrows from main flow to losses
    for lx, ly, lw, lh, _, _ in loss_boxes:
        ax.annotate('', xy=(lx + lw / 2, ly + lh),
                     xytext=(10.25, 2.5),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='--'))

    # Total loss
    ax.text(10.25, 0.1, '$\\mathcal{L} = \\mathcal{L}_{struct} + \\lambda_{ce}\\mathcal{L}_{ent} + \\lambda_{cc}\\mathcal{L}_{con}$',
            ha='center', fontsize=11, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Title
    ax.text(8, 5.5, 'HHRoad: Hierarchical Hyperbolic Road Network Representation',
            ha='center', fontsize=14, fontweight='bold')

    plt.savefig(output_path)
    plt.close()
    print(f"[Architecture] Saved to: {output_path}")


# ============================================================
# 示例数据生成函数
# ============================================================
def _get_example_main_results():
    """生成示例主实验结果数据（请替换为真实实验数据）"""
    np.random.seed(42)
    models = ['DeepWalk', 'Node2Vec', 'LINE', 'ChebConv', 'GAT', 'GeomGCN',
              'JCLRNT', 'SRN2Vec', 'HyperRoad', 'SARN', 'Toast', 'START',
              'TrajRNE', 'HRNR', 'HRNR_Hyperbolic']

    results = {}
    for city in ['xa', 'bj', 'cd']:
        city_data = {}
        for model in models:
            # Base performance (random but consistent)
            base = hash(model + city) % 100 / 100.0
            improvement = 0.15 if model == 'HRNR_Hyperbolic' else (0.1 if model == 'HRNR' else 0)

            city_data[model] = {
                'tsi_mae': max(0.5, 3.5 - base * 2 - improvement * 3 + np.random.randn() * 0.1),
                'tsi_rmse': max(1.0, 5.0 - base * 2.5 - improvement * 4 + np.random.randn() * 0.15),
                'tte_mae': max(10, 80 - base * 50 - improvement * 20 + np.random.randn() * 3),
                'tte_rmse': max(20, 120 - base * 60 - improvement * 25 + np.random.randn() * 5),
                'tte_mape': max(5, 35 - base * 20 - improvement * 10 + np.random.randn() * 2),
                'sts_hr10': min(0.99, 0.3 + base * 0.4 + improvement * 0.2 + np.random.randn() * 0.03),
                'sts_hr50': min(0.99, 0.5 + base * 0.3 + improvement * 0.15 + np.random.randn() * 0.02),
            }
        results[city] = city_data
    return results


def _get_example_ablation_data():
    """生成示例消融实验数据（请替换为真实实验数据）"""
    return {
        'Full Model\n(HHRoad)': {'tsi_mae': 1.05, 'tte_mae': 28.5, 'sts_hr10': 0.78},
        'w/o Entailment\n($\\lambda_{ce}$=0)': {'tsi_mae': 1.18, 'tte_mae': 31.2, 'sts_hr10': 0.72},
        'w/o Contrastive\n($\\lambda_{cc}$=0)': {'tsi_mae': 1.15, 'tte_mae': 30.8, 'sts_hr10': 0.74},
        'w/o Both\nAux. Losses': {'tsi_mae': 1.25, 'tte_mae': 33.1, 'sts_hr10': 0.68},
        'Euclidean\n(HRNR)': {'tsi_mae': 1.32, 'tte_mae': 35.4, 'sts_hr10': 0.65},
    }


def _get_example_sensitivity_data():
    """生成示例超参数敏感性数据（请替换为真实实验数据）"""
    return {
        'lambda_ce': {
            'values': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'tsi_mae': [1.25, 1.12, 1.05, 1.08, 1.15, 1.28],
            'tte_mae': [33.0, 30.1, 28.5, 29.2, 31.0, 34.5],
        },
        'lambda_cc': {
            'values': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'tsi_mae': [1.22, 1.10, 1.05, 1.07, 1.12, 1.20],
            'tte_mae': [32.5, 29.8, 28.5, 29.0, 30.5, 33.0],
        },
        'temperature': {
            'values': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
            'tsi_mae': [1.30, 1.15, 1.08, 1.05, 1.10, 1.22],
            'tte_mae': [35.0, 31.0, 29.5, 28.5, 30.0, 33.5],
        },
        'hyperbolic_dim': {
            'values': [64, 128, 192, 224, 256, 320],
            'tsi_mae': [1.35, 1.18, 1.10, 1.05, 1.06, 1.08],
            'tte_mae': [36.0, 31.5, 29.5, 28.5, 28.8, 29.2],
        },
    }


def _get_example_training_log():
    """生成示例训练日志数据（请替换为真实训练数据）"""
    n_steps = 100
    steps = list(range(0, n_steps * 20, 20))

    # Simulated loss curves
    loss_struct = [2.0 * np.exp(-i / 30) + 0.3 + np.random.randn() * 0.05 for i in range(n_steps)]
    loss_ce = [0.5 * np.exp(-i / 40) + 0.05 + np.random.randn() * 0.02 for i in range(n_steps)]
    loss_cc = [0.8 * np.exp(-i / 35) + 0.1 + np.random.randn() * 0.03 for i in range(n_steps)]
    loss_total = [ls + 0.1 * lce + 0.1 * lcc for ls, lce, lcc in zip(loss_struct, loss_ce, loss_cc)]

    # Simulated metrics
    auc = [min(0.95, 0.5 + 0.4 * (1 - np.exp(-i / 25)) + np.random.randn() * 0.02) for i in range(n_steps)]
    f1 = [min(0.9, 0.3 + 0.5 * (1 - np.exp(-i / 30)) + np.random.randn() * 0.03) for i in range(n_steps)]

    return {
        'steps': steps,
        'loss_total': loss_total,
        'loss_struct': loss_struct,
        'loss_ce': loss_ce,
        'loss_cc': loss_cc,
        'auc': auc,
        'f1': f1,
    }


def _get_example_radar_data():
    """生成示例雷达图数据（请替换为真实数据）"""
    return {
        'HHRoad (Ours)': {
            'Speed Inf.': 0.92, 'Travel Time': 0.88, 'Similarity': 0.85,
            'Hierarchy': 0.90, 'Scalability': 0.82, 'Convergence': 0.87
        },
        'HRNR': {
            'Speed Inf.': 0.80, 'Travel Time': 0.78, 'Similarity': 0.72,
            'Hierarchy': 0.65, 'Scalability': 0.85, 'Convergence': 0.80
        },
        'HyperRoad': {
            'Speed Inf.': 0.78, 'Travel Time': 0.75, 'Similarity': 0.70,
            'Hierarchy': 0.60, 'Scalability': 0.80, 'Convergence': 0.75
        },
    }


# ============================================================
# 实际数据加载函数
# ============================================================
def load_results_from_cache(exp_id, model, dataset, output_dim=128):
    """
    从VecCity缓存中加载实验结果。

    Args:
        exp_id: 实验ID
        model: 模型名称
        dataset: 数据集名称
        output_dim: 输出维度
    Returns:
        dict: 实验结果
    """
    result_path = os.path.join(
        PROJECT_ROOT, 'VecCity-main', 'veccity', 'cache',
        str(exp_id), 'evaluate_cache',
        f'{exp_id}_evaluate_{model}_{dataset}_{output_dim}.csv'
    )

    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        return df.iloc[0].to_dict()

    # Try alternative path
    alt_path = os.path.join(
        PROJECT_ROOT, 'VecCity-main', 'raw_data', 'new', 'evaluate_cache',
        f'{exp_id}_evaluate_{exp_id}_{model}_{dataset}_{output_dim}.csv'
    )
    if os.path.exists(alt_path):
        df = pd.read_csv(alt_path)
        return df.iloc[0].to_dict()

    return None


def find_embedding_files():
    """查找所有已保存的嵌入文件"""
    cache_dir = os.path.join(PROJECT_ROOT, 'VecCity-main', 'veccity', 'cache')
    embedding_files = []

    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                if f.startswith('road_embedding') and f.endswith('.npy'):
                    embedding_files.append(os.path.join(root, f))

    return embedding_files


def load_training_log_from_file(log_file):
    """
    从VecCity日志文件中解析训练数据。

    Args:
        log_file: 日志文件路径
    Returns:
        dict: 训练日志数据
    """
    data = {
        'steps': [], 'loss_total': [], 'loss_struct': [],
        'loss_ce': [], 'loss_cc': [], 'auc': [], 'f1': []
    }

    if not os.path.exists(log_file):
        return None

    step = 0
    with open(log_file, 'r') as f:
        for line in f:
            # Parse loss lines
            if 'loss:' in line and 'struct:' in line:
                try:
                    parts = line.split('loss:')[1]
                    loss_total = float(parts.split(',')[0].strip())
                    loss_struct = float(parts.split('struct:')[1].split(',')[0].strip())
                    loss_ce = float(parts.split('ce:')[1].split(',')[0].strip())
                    loss_cc = float(parts.split('cc:')[1].strip())

                    data['steps'].append(step)
                    data['loss_total'].append(loss_total)
                    data['loss_struct'].append(loss_struct)
                    data['loss_ce'].append(loss_ce)
                    data['loss_cc'].append(loss_cc)
                    step += 1
                except (ValueError, IndexError):
                    pass

            # Parse AUC
            if 'auc:' in line and 'max_auc' not in line:
                try:
                    auc = float(line.split('auc:')[1].strip().split()[0])
                    data['auc'].append(auc)
                except (ValueError, IndexError):
                    pass

            # Parse F1
            if 'p/r/f:' in line or '@p/r/f:' in line:
                try:
                    parts = line.split('/')
                    if len(parts) >= 3:
                        f1 = float(parts[-1].strip())
                        data['f1'].append(f1)
                except (ValueError, IndexError):
                    pass

    return data if data['steps'] else None


# ============================================================
# 主函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='HHRoad Paper Visualization Tools')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'table', 'ablation', 'embedding', 'sensitivity',
                                 'training', 'hierarchy', 'quality', 'radar', 'city', 'arch'],
                        help='Visualization mode')
    parser.add_argument('--embedding_file', type=str, default=None,
                        help='Path to hyperbolic embedding .npy file')
    parser.add_argument('--euc_embedding_file', type=str, default=None,
                        help='Path to euclidean embedding .npy file')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to results JSON file')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    return parser.parse_args()


def main():
    args = parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HHRoad Paper Visualization Tools")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mode: {args.mode}")
    print()

    # Try to find actual embedding files
    emb_files = find_embedding_files()
    if emb_files:
        print(f"Found {len(emb_files)} embedding files:")
        for f in emb_files:
            print(f"  - {f}")
    print()

    # Load results if file provided
    results_data = None
    if args.results_file and os.path.exists(args.results_file):
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)

    # Select embedding file
    emb_file = args.embedding_file
    if emb_file is None and emb_files:
        emb_file = emb_files[0]

    # Load training log
    training_log = None
    if args.log_file:
        training_log = load_training_log_from_file(args.log_file)

    # Generate visualizations
    if args.mode in ['all', 'table']:
        print("--- Generating main results table ---")
        generate_main_results_table(results_data)

    if args.mode in ['all', 'ablation']:
        print("--- Generating ablation study ---")
        plot_ablation_study()

    if args.mode in ['all', 'embedding']:
        print("--- Generating embedding visualization ---")
        plot_hyperbolic_embedding(emb_file)

    if args.mode in ['all', 'sensitivity']:
        print("--- Generating hyperparameter sensitivity ---")
        plot_hyperparameter_sensitivity()

    if args.mode in ['all', 'training']:
        print("--- Generating training curves ---")
        plot_training_curves(training_log)

    if args.mode in ['all', 'hierarchy']:
        print("--- Generating hierarchy visualization ---")
        plot_hierarchy_visualization()

    if args.mode in ['all', 'quality']:
        print("--- Generating embedding quality comparison ---")
        plot_embedding_quality_comparison(emb_file, args.euc_embedding_file)

    if args.mode in ['all', 'radar']:
        print("--- Generating radar chart ---")
        plot_radar_chart()

    if args.mode in ['all', 'city']:
        print("--- Generating multi-city comparison ---")
        plot_multi_city_comparison(results_data)

    if args.mode in ['all', 'arch']:
        print("--- Generating model architecture diagram ---")
        plot_model_architecture()

    print()
    print("=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print()
    print("To use with real data, provide:")
    print("  --embedding_file <path_to_embedding.npy>")
    print("  --results_file <path_to_results.json>")
    print("  --log_file <path_to_training.log>")
    print()
    print("Or modify the _get_example_*() functions with your real experimental data.")


if __name__ == '__main__':
    main()
