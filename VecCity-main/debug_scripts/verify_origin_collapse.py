#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证所有嵌入是否都坍缩到了Lorentz原点 [1, 0, 0, ..., 0]
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_collapse():
    """验证嵌入坍缩假设"""
    print("="*80)
    print("验证假设：所有嵌入都坍缩到Lorentz原点 [1, 0, 0, ..., 0]")
    print("="*80)

    from veccity.config import ConfigParser
    from veccity.data import get_dataset
    from veccity.utils import get_model

    config = ConfigParser(
        task='segment',
        model='HRNR_Hyperbolic',
        dataset='cd',
        config_file=None,
        saved_model=False,
        train=True,
        other_args={'gpu': True, 'gpu_id': 0}
    )

    dataset = get_dataset(config)
    model = get_model(config, dataset.get_data_feature())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("\n生成嵌入...")
    model.eval()

    with torch.no_grad():
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature, model.lane_feature, model.adj
        )

    segment_emb = model.graph_enc.segment_hyp_emb.cpu().numpy()

    print("\n" + "="*80)
    print("检查空间分量（维度1-224）是否全为0")
    print("="*80)

    # 空间分量（跳过时间分量）
    spatial_components = segment_emb[:, 1:]

    print(f"\n空间分量形状: {spatial_components.shape}")
    print(f"空间分量范围: [{spatial_components.min():.10f}, {spatial_components.max():.10f}]")
    print(f"空间分量均值: {spatial_components.mean():.10f}")
    print(f"空间分量标准差: {spatial_components.std():.10f}")
    print(f"空间分量平方和的均值: {np.mean(np.sum(spatial_components**2, axis=1)):.10f}")

    # 检查每个点的空间分量范数
    spatial_norms = np.sqrt(np.sum(spatial_components**2, axis=1))
    print(f"\n每个点的空间分量范数:")
    print(f"  最小值: {spatial_norms.min():.10f}")
    print(f"  最大值: {spatial_norms.max():.10f}")
    print(f"  均值: {spatial_norms.mean():.10f}")

    # 显示前10个点
    print(f"\n前10个点:")
    print(f"{'点编号':<8} {'时间分量':<15} {'空间范数':<15} {'是否≈原点':<10}")
    print("-"*60)
    for i in range(min(10, segment_emb.shape[0])):
        time_comp = segment_emb[i, 0]
        spatial_norm = spatial_norms[i]
        is_origin = "是" if spatial_norm < 1e-6 else "否"
        print(f"{i:<8} {time_comp:<15.10f} {spatial_norm:<15.10f} {is_origin:<10}")

    # 统计
    threshold = 1e-6
    at_origin_count = np.sum(spatial_norms < threshold)
    print(f"\n统计:")
    print(f"  阈值: {threshold}")
    print(f"  在原点的点数: {at_origin_count} / {segment_emb.shape[0]}")
    print(f"  在原点的比例: {100.0 * at_origin_count / segment_emb.shape[0]:.2f}%")

    print("\n" + "="*80)
    print("结论:")
    print("="*80)

    if at_origin_count > segment_emb.shape[0] * 0.99:
        print("\n❌❌❌ 确认：99%以上的嵌入都在Lorentz原点！")
        print("\n所有嵌入的形式都是: [1, 0, 0, 0, ..., 0]")
        print("\n这就是为什么：")
        print("  1. 所有点之间的Minkowski内积 <x,y> = -1")
        print("  2. 所有 cos_angle = -1")
        print("  3. 所有角度 = π (180度)")
        print("  4. 梯度在clamp边界消失")
        print("  5. STS完全失败 (HR@3 = 2%)")
        print("\n需要修复的地方：")
        print("  → 检查图编码器的初始化")
        print("  → 检查exp_map投影到Lorentz流形的实现")
        print("  → 检查是否有数值下溢导致空间分量归零")
    else:
        print(f"\n只有 {100.0 * at_origin_count / segment_emb.shape[0]:.2f}% 的点在原点")
        print("问题可能在其他地方")

    # 保存一些样本点供分析
    print(f"\n保存前100个点到 sample_embeddings.txt 用于详细分析")
    with open('sample_embeddings.txt', 'w') as f:
        f.write("前100个Segment嵌入点\n")
        f.write("="*80 + "\n\n")
        for i in range(min(100, segment_emb.shape[0])):
            f.write(f"点 {i}:\n")
            f.write(f"  时间分量: {segment_emb[i, 0]:.10f}\n")
            f.write(f"  空间范数: {spatial_norms[i]:.10f}\n")
            f.write(f"  完整向量: {segment_emb[i]}\n\n")

    print("✓ 完成")

if __name__ == '__main__':
    verify_collapse()
