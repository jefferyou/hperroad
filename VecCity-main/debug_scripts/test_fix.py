#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的空间分量是否恢复正常
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fix():
    """测试修复后的效果"""
    print("="*80)
    print("测试修复后的图编码器")
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

    print("\n生成嵌入...")
    with torch.no_grad():
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature, model.lane_feature, model.adj
        )

    segment_emb = model.graph_enc.segment_hyp_emb.cpu().numpy()

    print("\n" + "="*80)
    print("修复后的segment_hyp_emb统计")
    print("="*80)

    time_comp = segment_emb[:, 0]
    spatial_comp = segment_emb[:, 1:]
    spatial_norms = np.linalg.norm(spatial_comp, axis=1)

    print(f"\n形状: {segment_emb.shape}")
    print(f"\n时间分量:")
    print(f"  范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
    print(f"  均值: {time_comp.mean():.6f}")

    print(f"\n空间分量:")
    print(f"  范围: [{spatial_comp.min():.10f}, {spatial_comp.max():.10f}]")
    print(f"  均值: {spatial_comp.mean():.10f}")
    print(f"  标准差: {spatial_comp.std():.10f}")

    print(f"\n空间范数:")
    print(f"  最小值: {spatial_norms.min():.10f}")
    print(f"  最大值: {spatial_norms.max():.10f}")
    print(f"  均值: {spatial_norms.mean():.10f}")

    # Lorentz约束
    lorentz_constraint = -time_comp**2 + np.sum(spatial_comp**2, axis=1)
    print(f"\nLorentz约束 <x,x> = -1:")
    print(f"  偏差均值: {np.abs(lorentz_constraint + 1.0).mean():.10f}")
    print(f"  偏差最大值: {np.abs(lorentz_constraint + 1.0).max():.10f}")

    # 统计接近原点的点
    near_origin_count = np.sum(spatial_norms < 1e-3)
    print(f"\n接近原点 (空间范数<0.001):")
    print(f"  数量: {near_origin_count} / {segment_emb.shape[0]}")
    print(f"  比例: {100.0 * near_origin_count / segment_emb.shape[0]:.1f}%")

    print("\n" + "="*80)
    print("对比修复前后")
    print("="*80)

    print("\n修复前:")
    print("  空间范数均值: ~0.0002")
    print("  时间分量: 全部接近1.0")
    print("  接近原点比例: 100%")

    print("\n修复后:")
    print(f"  空间范数均值: {spatial_norms.mean():.6f}")
    print(f"  时间分量范围: [{time_comp.min():.2f}, {time_comp.max():.2f}]")
    print(f"  接近原点比例: {100.0 * near_origin_count / segment_emb.shape[0]:.1f}%")

    print("\n" + "="*80)
    print("判断")
    print("="*80)

    if spatial_norms.mean() > 0.1:
        print("\n✅ 成功！空间分量恢复正常")
        print("   空间范数均值 > 0.1，点不再全部坍缩到原点")
    elif spatial_norms.mean() > 0.01:
        print("\n⚠️  部分改善，但仍需优化")
        print(f"   空间范数均值 = {spatial_norms.mean():.6f}")
        print("   可能需要进一步调整")
    else:
        print("\n❌ 修复似乎无效")
        print(f"   空间范数均值仍然很小: {spatial_norms.mean():.10f}")
        print("   可能需要检查其他问题")

    # 检查唯一性
    unique_rows = np.unique(segment_emb, axis=0).shape[0]
    print(f"\n唯一嵌入点数: {unique_rows} / {segment_emb.shape[0]}")
    if unique_rows > segment_emb.shape[0] * 0.5:
        print("✅ 嵌入具有良好的多样性")
    else:
        print("⚠️  嵌入仍缺乏多样性")

    print("\n完成！")

if __name__ == '__main__':
    test_fix()
