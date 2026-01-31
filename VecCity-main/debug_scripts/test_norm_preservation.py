#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试范数保持修复的效果
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

def test_norm_preservation():
    """测试范数保持修复"""
    print("="*80)
    print("测试范数保持修复（norm preservation fix）")
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

    # 添加调试钩子
    aggregation_stats = []

    original_aggregate = model.graph_enc.tl_layer_1._aggregate_to_cluster

    def debug_aggregate(embeddings, assignment_matrix):
        input_norm = torch.norm(embeddings[:, 1:], dim=1).mean().item()
        result = original_aggregate(embeddings, assignment_matrix)
        output_norm = torch.norm(result[:, 1:], dim=1).mean().item()

        retention = output_norm / input_norm if input_norm > 0 else 0
        aggregation_stats.append({
            'input_norm': input_norm,
            'output_norm': output_norm,
            'retention': retention
        })

        print(f"\n[聚合] 输入范数: {input_norm:.6f} → 输出范数: {output_norm:.6f} (保留{100*retention:.1f}%)")

        return result

    model.graph_enc.tl_layer_1._aggregate_to_cluster = debug_aggregate

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

    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)

    print(f"\n初始嵌入空间范数: {init_norm:.6f}")
    print(f"最终嵌入空间范数: {final_spatial_norm:.10f}")

    total_retention = final_spatial_norm / init_norm
    print(f"\n总体保留率: {100*total_retention:.2f}%")
    print(f"衰减比例: {total_retention:.6e}")

    print(f"\n聚合操作统计 ({len(aggregation_stats)} 次调用):")
    for i, stat in enumerate(aggregation_stats, 1):
        print(f"  第{i}次: {stat['input_norm']:.4f} → {stat['output_norm']:.4f} (保留{100*stat['retention']:.1f}%)")

    print("\n" + "="*80)
    print("判断")
    print("="*80)

    if total_retention > 0.05:  # 保留>5%
        print(f"\n✅ 修复有效！总体保留率 {100*total_retention:.2f}% > 5%")
        print(f"   最终空间范数: {final_spatial_norm:.6f}")

        # 检查聚合保留率
        avg_agg_retention = np.mean([s['retention'] for s in aggregation_stats])
        print(f"   聚合平均保留率: {100*avg_agg_retention:.1f}%")

        if avg_agg_retention > 0.6:
            print(f"   ✅ 聚合操作保持良好 (>{60}%)")
        else:
            print(f"   ⚠️  聚合仍有较大衰减 (<{60}%)")

    elif total_retention > 0.001:  # 0.1% - 5%
        print(f"\n⚠️  部分改善，但仍需优化")
        print(f"   总体保留率: {100*total_retention:.2f}%")
        print(f"   可能需要增加保留比例（从70%/85%提高到90%/95%）")

    else:  # <0.1%
        print(f"\n❌ 修复无效，衰减仍然严重")
        print(f"   总体保留率: {100*total_retention:.4f}%")
        print(f"   问题可能在其他操作（图卷积、门控等）")

    # 检查唯一性
    unique_count = np.unique(final_emb, axis=0).shape[0]
    print(f"\n唯一embedding数: {unique_count}/{final_emb.shape[0]}")

    if unique_count > final_emb.shape[0] * 0.9:
        print("✅ 嵌入具有良好的多样性")
    else:
        print(f"⚠️  嵌入多样性: {100*unique_count/final_emb.shape[0]:.1f}%")

    print("\n完成！")

if __name__ == '__main__':
    test_norm_preservation()
