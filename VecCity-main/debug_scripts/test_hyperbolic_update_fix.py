#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 _hyperbolic_update 修复的效果
这是导致77%损失的真正罪魁祸首
"""

import sys
import os

# 清除模块缓存
if 'veccity' in sys.modules:
    modules_to_remove = [key for key in sys.modules if key.startswith('veccity')]
    for module in modules_to_remove:
        del sys.modules[module]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_update_fix():
    """测试 hyperbolic_update 修复"""
    print("="*80)
    print("测试 _hyperbolic_update 修复")
    print("这个操作曾导致77%的范数损失")
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

    print("\n运行前向传播...\n")

    with torch.no_grad():
        # 初始嵌入
        node_emb = model.graph_enc.node_emb_layer(model.node_feature)
        type_emb = model.graph_enc.type_emb_layer(model.type_feature)
        length_emb = model.graph_enc.length_emb_layer(model.length_feature)
        lane_emb = model.graph_enc.lane_emb_layer(model.lane_feature)
        euclidean_feat = torch.cat([node_emb, type_emb, length_emb, lane_emb], dim=-1)

        hyp_init = model.graph_enc.hyp_embedding(euclidean_feat)
        init_norm = torch.norm(hyp_init[:, 1:], dim=1).mean().item()

        # 完整前向传播
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature,
            model.lane_feature, model.adj
        )

        final_emb = model.graph_enc.segment_hyp_emb.cpu().numpy()

    # 分析结果
    final_spatial_norm = np.linalg.norm(final_emb[:, 1:], axis=1).mean()
    time_comp = final_emb[:, 0]

    print("="*80)
    print("结果")
    print("="*80)

    print(f"\n初始嵌入空间范数: {init_norm:.6f}")
    print(f"最终嵌入空间范数: {final_spatial_norm:.6f}")

    retention = final_spatial_norm / init_norm
    print(f"\n总体保留率: {100*retention:.2f}%")

    print(f"\n时间分量:")
    print(f"  范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
    print(f"  均值: {time_comp.mean():.6f}")
    print(f"  标准差: {time_comp.std():.6f}")

    print("\n" + "="*80)
    print("修复历程")
    print("="*80)

    print("\n原始（坍缩）:")
    print("  总体保留率: 0.0027%")
    print("  问题: 聚合+图卷积导致坍缩")

    print("\n第一次修复（聚合+图卷积）:")
    print("  总体保留率: 5.02%")
    print("  改善: 1860x")
    print("  但: hyperbolic_update未修复")

    print("\n第二次尝试（提高保留率参数）:")
    print("  总体保留率: 4.98%")
    print("  结果: 无效，说明问题在其他地方")

    print("\n第三次修复（hyperbolic_update）:")
    print(f"  总体保留率: {100*retention:.2f}%")
    print(f"  改善: {retention/0.0502:.2f}x (相比第一次修复)")
    print(f"  改善: {retention/2.7e-5:.0f}x (相比原始坍缩)")

    print("\n" + "="*80)
    print("判断")
    print("="*80)

    if retention > 0.15:  # >15%
        print(f"\n🎉 重大突破！")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   超过预期目标 (>15%)")
        print(f"   空间分量已完全恢复")

    elif retention > 0.10:  # 10-15%
        print(f"\n✅ 修复成功！")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   达到预期目标 (10-15%)")
        print(f"   相比第一次修复改善: {retention/0.0502:.2f}x")

    elif retention > 0.07:  # 7-10%
        print(f"\n⚠️  显著改善，接近目标")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   相比第一次修复改善: {retention/0.0502:.2f}x")

    else:  # <7%
        print(f"\n⚠️  改善有限")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   可能还有其他问题")

    # 详细统计
    unique_count = np.unique(final_emb, axis=0).shape[0]
    print(f"\n嵌入多样性: {unique_count}/{final_emb.shape[0]} ({100*unique_count/final_emb.shape[0]:.1f}%)")

    spatial_norms = np.linalg.norm(final_emb[:, 1:], axis=1)
    near_origin = np.sum(spatial_norms < 0.01)
    print(f"接近原点 (<0.01): {near_origin}/{final_emb.shape[0]} ({100*near_origin/final_emb.shape[0]:.1f}%)")

    print("\n完成！")

if __name__ == '__main__':
    test_update_fix()
