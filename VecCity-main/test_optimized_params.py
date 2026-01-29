#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化后的参数效果
聚合: 70%→80%, 分发: 85%→90%, 图卷积: 80%→90%
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

def test_optimized():
    """测试优化后的效果"""
    print("="*80)
    print("测试优化后的范数保留参数")
    print("聚合: 70%→80%, 分发: 85%→90%, 图卷积: 80%→90%")
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
    spatial_comp = final_emb[:, 1:]

    print("="*80)
    print("结果对比")
    print("="*80)

    print(f"\n初始嵌入空间范数: {init_norm:.6f}")
    print(f"最终嵌入空间范数: {final_spatial_norm:.6f}")

    retention = final_spatial_norm / init_norm
    print(f"\n总体保留率: {100*retention:.2f}%")
    print(f"衰减比例: {retention:.6e}")

    print(f"\n时间分量:")
    print(f"  范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
    print(f"  均值: {time_comp.mean():.6f}")
    print(f"  标准差: {time_comp.std():.6f}")

    print(f"\n空间分量:")
    print(f"  范围: [{spatial_comp.min():.6f}, {spatial_comp.max():.6f}]")
    print(f"  均值: {spatial_comp.mean():.6f}")
    print(f"  标准差: {spatial_comp.std():.6f}")

    # 检查Lorentz约束
    lorentz_constraint = -time_comp**2 + np.sum(spatial_comp**2, axis=1)
    print(f"\nLorentz约束 <x,x> = -1:")
    print(f"  偏差均值: {np.abs(lorentz_constraint + 1.0).mean():.6e}")
    print(f"  偏差最大: {np.abs(lorentz_constraint + 1.0).max():.6e}")

    print("\n" + "="*80)
    print("对比修复历史")
    print("="*80)

    print("\n修复前（坍缩）:")
    print("  总体保留率: 0.0027%")
    print("  最终空间范数: 0.0002")
    print("  时间分量: [1.0, 1.0]")

    print("\n第一次修复（聚合70%, 分发85%, 图卷积80%）:")
    print("  总体保留率: 5.02%")
    print("  最终空间范数: 0.428")
    print("  时间分量: [1.058, 1.134]")

    print("\n优化后（聚合80%, 分发90%, 图卷积90%）:")
    print(f"  总体保留率: {100*retention:.2f}%")
    print(f"  最终空间范数: {final_spatial_norm:.3f}")
    print(f"  时间分量: [{time_comp.min():.3f}, {time_comp.max():.3f}]")

    print("\n" + "="*80)
    print("判断")
    print("="*80)

    # 计算改善
    improvement_over_collapse = retention / 2.7e-5  # 相比坍缩的改善
    improvement_over_first_fix = retention / 0.0502  # 相比第一次修复的改善

    if retention > 0.10:  # >10%
        print(f"\n🎉 优化成功！")
        print(f"   总体保留率: {100*retention:.2f}% (目标: >10%)")
        print(f"   相比坍缩改善: {improvement_over_collapse:.0f}x")
        print(f"   相比第一次修复改善: {improvement_over_first_fix:.2f}x")

        if retention > 0.15:
            print(f"   ✅ 优秀！保留率超过15%")
        else:
            print(f"   ✅ 良好！达到预期目标")

    elif retention > 0.07:  # 7-10%
        print(f"\n⚠️  有改善，接近目标")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   相比第一次修复改善: {improvement_over_first_fix:.2f}x")
        print(f"   建议: 可以进一步提高到85%/92%/92%")

    else:  # <7%
        print(f"\n⚠️  改善有限")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   相比第一次修复改善: {improvement_over_first_fix:.2f}x")
        print(f"   可能需要检查其他因素")

    # 检查唯一性
    unique_count = np.unique(final_emb, axis=0).shape[0]
    diversity = 100 * unique_count / final_emb.shape[0]
    print(f"\n嵌入多样性: {unique_count}/{final_emb.shape[0]} ({diversity:.1f}%)")

    if diversity > 95:
        print("   ✅ 优秀的多样性")
    elif diversity > 90:
        print("   ✅ 良好的多样性")
    else:
        print(f"   ⚠️  多样性可以更好")

    # 检查接近原点的点
    spatial_norms = np.linalg.norm(spatial_comp, axis=1)
    near_origin = np.sum(spatial_norms < 0.01)
    print(f"\n接近原点的点 (<0.01): {near_origin}/{final_emb.shape[0]} ({100*near_origin/final_emb.shape[0]:.1f}%)")

    if near_origin == 0:
        print("   ✅ 完全没有点坍缩到原点")
    elif near_origin < final_emb.shape[0] * 0.01:
        print("   ✅ 极少点接近原点")
    else:
        print(f"   ⚠️  仍有较多点接近原点")

    print("\n完成！")

if __name__ == '__main__':
    test_optimized()
