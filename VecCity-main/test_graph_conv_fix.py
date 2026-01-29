#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试图卷积修复的效果
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

def test_graph_conv_fix():
    """测试图卷积修复"""
    print("="*80)
    print("测试图卷积修复（权重初始化 + 范数保持）")
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

    print("\n检查图卷积层的权重初始化...")
    weight_norm = torch.norm(model.graph_enc.tl_layer_1.fnc_gcn.weight).item()
    print(f"fnc_gcn权重范数: {weight_norm:.6f}")

    if weight_norm > 1.0:
        print("✅ 权重初始化正常 (gain=1.0)")
    else:
        print("⚠️  权重可能仍然很小")

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
    print(f"衰减比例: {retention:.6e}")

    print(f"\n时间分量:")
    print(f"  范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
    print(f"  均值: {time_comp.mean():.6f}")
    print(f"  标准差: {time_comp.std():.6f}")

    print("\n" + "="*80)
    print("判断")
    print("="*80)

    if retention > 0.2:  # 保留>20%
        print(f"\n✅ 修复成功！总体保留率 {100*retention:.2f}% > 20%")
        print(f"   最终空间范数: {final_spatial_norm:.6f}")

        # 检查时间分量多样性
        if time_comp.std() > 0.01:
            print(f"   ✅ 时间分量有多样性 (std={time_comp.std():.6f})")
        else:
            print(f"   ⚠️  时间分量缺乏多样性 (std={time_comp.std():.6f})")

    elif retention > 0.05:  # 5% - 20%
        print(f"\n⚠️  显著改善，但可以更好")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   建议：进一步调整范数保持参数（从80%提高到90%）")

    elif retention > 0.01:  # 1% - 5%
        print(f"\n⚠️  有改善，但仍不够")
        print(f"   总体保留率: {100*retention:.2f}%")
        print(f"   可能还有其他操作导致坍缩")

    else:  # <1%
        print(f"\n❌ 修复无效或未生效")
        print(f"   总体保留率: {100*retention:.4f}%")
        print(f"   请检查：")
        print(f"   1. 代码是否被正确加载（重启Python）")
        print(f"   2. 权重初始化是否更新")

    # 检查唯一性
    unique_count = np.unique(final_emb, axis=0).shape[0]
    print(f"\n唯一embedding数: {unique_count}/{final_emb.shape[0]} ({100*unique_count/final_emb.shape[0]:.1f}%)")

    # 检查接近原点的点
    spatial_norms = np.linalg.norm(final_emb[:, 1:], axis=1)
    near_origin = np.sum(spatial_norms < 0.001)
    print(f"接近原点的点 (<0.001): {near_origin}/{final_emb.shape[0]} ({100*near_origin/final_emb.shape[0]:.1f}%)")

    print("\n完成！")

if __name__ == '__main__':
    test_graph_conv_fix()
