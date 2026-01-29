#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深入诊断：追踪空间分量坍缩的根本原因
检查从欧氏特征到双曲嵌入的整个流程
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_spatial_collapse():
    """追踪空间分量从欧氏空间到双曲空间的变化"""
    print("="*80)
    print("诊断：空间分量坍缩的根本原因")
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
    model.eval()

    print("\n" + "="*80)
    print("步骤1: 检查原始欧氏特征")
    print("="*80)

    # 获取原始特征嵌入
    node_emb = model.graph_enc.node_emb_layer(model.node_feature).detach()
    type_emb = model.graph_enc.type_emb_layer(model.type_feature).detach()
    length_emb = model.graph_enc.length_emb_layer(model.length_feature).detach()
    lane_emb = model.graph_enc.lane_emb_layer(model.lane_feature).detach()

    # 拼接欧氏特征
    euclidean_feat = torch.cat([node_emb, type_emb, length_emb, lane_emb], dim=-1)
    euclidean_feat_np = euclidean_feat.cpu().numpy()

    print(f"\n欧氏特征形状: {euclidean_feat_np.shape}")
    print(f"欧氏特征范围: [{euclidean_feat_np.min():.6f}, {euclidean_feat_np.max():.6f}]")
    print(f"欧氏特征均值: {euclidean_feat_np.mean():.6f}")
    print(f"欧氏特征标准差: {euclidean_feat_np.std():.6f}")
    print(f"欧氏特征L2范数统计:")
    norms = np.linalg.norm(euclidean_feat_np, axis=1)
    print(f"  最小值: {norms.min():.6f}")
    print(f"  最大值: {norms.max():.6f}")
    print(f"  均值: {norms.mean():.6f}")

    print("\n" + "="*80)
    print("步骤2: 检查HyperbolicEmbedding的线性变换")
    print("="*80)

    # 获取线性变换后的向量（投影前）
    hyp_emb_layer = model.graph_enc.hyp_embedding
    with torch.no_grad():
        transformed = hyp_emb_layer.linear(euclidean_feat)

    transformed_np = transformed.cpu().numpy()
    print(f"\n线性变换后的向量形状: {transformed_np.shape}")
    print(f"线性变换后范围: [{transformed_np.min():.10f}, {transformed_np.max():.10f}]")
    print(f"线性变换后均值: {transformed_np.mean():.10f}")
    print(f"线性变换后标准差: {transformed_np.std():.10f}")
    print(f"线性变换后L2范数统计:")
    transformed_norms = np.linalg.norm(transformed_np, axis=1)
    print(f"  最小值: {transformed_norms.min():.10f}")
    print(f"  最大值: {transformed_norms.max():.10f}")
    print(f"  均值: {transformed_norms.mean():.10f}")

    # 检查线性层的权重
    weight = hyp_emb_layer.linear.weight.detach().cpu().numpy()
    bias = hyp_emb_layer.linear.bias.detach().cpu().numpy()
    print(f"\n线性层权重统计:")
    print(f"  形状: {weight.shape}")
    print(f"  范围: [{weight.min():.6f}, {weight.max():.6f}]")
    print(f"  均值: {weight.mean():.6f}")
    print(f"  标准差: {weight.std():.6f}")
    print(f"线性层偏置统计:")
    print(f"  范围: [{bias.min():.6f}, {bias.max():.6f}]")
    print(f"  均值: {bias.mean():.6f}")

    print("\n" + "="*80)
    print("步骤3: 检查投影到Lorentz空间")
    print("="*80)

    # 投影到Lorentz空间
    with torch.no_grad():
        hyp_feat = hyp_emb_layer(euclidean_feat)

    hyp_feat_np = hyp_feat.cpu().numpy()
    time_comp = hyp_feat_np[:, 0]
    spatial_comp = hyp_feat_np[:, 1:]

    print(f"\n投影后Lorentz嵌入形状: {hyp_feat_np.shape}")
    print(f"时间分量统计:")
    print(f"  范围: [{time_comp.min():.10f}, {time_comp.max():.10f}]")
    print(f"  均值: {time_comp.mean():.10f}")
    print(f"  标准差: {time_comp.std():.10f}")
    print(f"\n空间分量统计:")
    print(f"  范围: [{spatial_comp.min():.10f}, {spatial_comp.max():.10f}]")
    print(f"  均值: {spatial_comp.mean():.10f}")
    print(f"  标准差: {spatial_comp.std():.10f}")

    spatial_norms = np.linalg.norm(spatial_comp, axis=1)
    print(f"空间分量L2范数统计:")
    print(f"  最小值: {spatial_norms.min():.10f}")
    print(f"  最大值: {spatial_norms.max():.10f}")
    print(f"  均值: {spatial_norms.mean():.10f}")

    # 验证Lorentz约束
    lorentz_constraint = -time_comp**2 + np.sum(spatial_comp**2, axis=1)
    print(f"\nLorentz约束 <x,x> = -1 检查:")
    print(f"  实际值范围: [{lorentz_constraint.min():.10f}, {lorentz_constraint.max():.10f}]")
    print(f"  与-1的偏差均值: {np.abs(lorentz_constraint + 1.0).mean():.10f}")
    print(f"  与-1的偏差最大值: {np.abs(lorentz_constraint + 1.0).max():.10f}")

    print("\n" + "="*80)
    print("步骤4: 手工验证project_to_lorentz公式")
    print("="*80)

    # 手工计算投影
    x_norm_sq = np.sum(transformed_np**2, axis=1, keepdims=True)
    x_0_manual = np.sqrt(1.0 + x_norm_sq)

    print(f"\n手工计算:")
    print(f"||x||^2 范围: [{x_norm_sq.min():.10f}, {x_norm_sq.max():.10f}]")
    print(f"||x||^2 均值: {x_norm_sq.mean():.10f}")
    print(f"x_0 = sqrt(1 + ||x||^2) 范围: [{x_0_manual.min():.10f}, {x_0_manual.max():.10f}]")
    print(f"x_0 均值: {x_0_manual.mean():.10f}")

    print("\n" + "="*80)
    print("结论分析")
    print("="*80)

    if transformed_norms.mean() < 1e-3:
        print("\n🔍 发现问题：线性变换后的向量范数太小！")
        print(f"   平均范数: {transformed_norms.mean():.10f}")
        print(f"\n根据公式: h = [sqrt(1 + ||x||^2), x_1, x_2, ..., x_d]")
        print(f"   当 ||x|| ≈ {transformed_norms.mean():.2e} 时:")
        print(f"   时间分量 ≈ sqrt(1 + {transformed_norms.mean()**2:.2e}) ≈ 1.0")
        print(f"   空间分量 ≈ x ≈ {transformed_norms.mean():.2e}")
        print(f"\n所以所有点都聚集在 [1, 0, 0, ..., 0] 附近！")

        print("\n可能的根本原因:")
        if norms.mean() < 0.1:
            print("  1. ❌ 原始欧氏特征本身就很小")
            print(f"     → 检查embedding层的初始化")
        if weight.std() < 0.01:
            print("  2. ❌ 线性层权重太小")
            print(f"     → 当前权重标准差: {weight.std():.6f}")
            print(f"     → 检查线性层的初始化")

        print("\n建议修复:")
        print("  1. 增加线性层权重的初始化scale")
        print("  2. 考虑在project_to_lorentz前添加缩放因子")
        print("  3. 或使用exp_map从原点出发，而不是project_to_lorentz")
    else:
        print("\n线性变换看起来正常，问题可能在其他地方")

    # 保存详细数据
    print("\n保存详细数据到 spatial_collapse_diagnosis.npz")
    np.savez('spatial_collapse_diagnosis.npz',
             euclidean_feat=euclidean_feat_np,
             transformed=transformed_np,
             hyperbolic_feat=hyp_feat_np,
             weight=weight,
             bias=bias)
    print("✓ 完成")

if __name__ == '__main__':
    diagnose_spatial_collapse()
