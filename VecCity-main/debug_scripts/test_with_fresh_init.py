#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用全新初始化测试修复效果
强制重新加载模块
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

def test_with_fresh_init():
    """使用全新初始化测试"""
    print("="*80)
    print("使用全新初始化测试修复效果")
    print("="*80)

    from veccity.config import ConfigParser
    from veccity.data import get_dataset
    from veccity.utils import get_model

    # 强制不加载保存的模型
    config = ConfigParser(
        task='segment',
        model='HRNR_Hyperbolic',
        dataset='cd',
        config_file=None,
        saved_model=False,  # 关键：不加载已保存的模型
        train=True,
        other_args={'gpu': True, 'gpu_id': 0}
    )

    dataset = get_dataset(config)
    model = get_model(config, dataset.get_data_feature())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 检查代码版本
    import inspect
    agg_source = inspect.getsource(model.graph_enc.tl_layer_1._aggregate_to_cluster)
    if "log_map" in agg_source and "exp_map" in agg_source:
        print("\n✅ 代码已更新（包含log_map和exp_map）")
    else:
        print("\n❌ 代码仍是旧版本")
        print("请重启Python解释器或清除__pycache__")
        return

    # 在eval模式下测试（不训练）
    model.eval()

    print("\n第1次前向传播（初始化权重）:")
    with torch.no_grad():
        # 获取初始HyperbolicEmbedding输出
        node_emb = model.graph_enc.node_emb_layer(model.node_feature)
        type_emb = model.graph_enc.type_emb_layer(model.type_feature)
        length_emb = model.graph_enc.length_emb_layer(model.length_feature)
        lane_emb = model.graph_enc.lane_emb_layer(model.lane_feature)
        euclidean_feat = torch.cat([node_emb, type_emb, length_emb, lane_emb], dim=-1)

        hyp_init = model.graph_enc.hyp_embedding(euclidean_feat)
        init_spatial_norm = torch.norm(hyp_init[:, 1:], dim=1).mean().item()

        print(f"  初始双曲嵌入空间范数: {init_spatial_norm:.6f}")

        # 完整前向传播
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature,
            model.lane_feature, model.adj
        )

        segment_emb = model.graph_enc.segment_hyp_emb.cpu().numpy()

    # 分析结果
    time_comp = segment_emb[:, 0]
    spatial_comp = segment_emb[:, 1:]
    spatial_norms = np.linalg.norm(spatial_comp, axis=1)

    print(f"\n最终segment_hyp_emb:")
    print(f"  形状: {segment_emb.shape}")
    print(f"  时间分量范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
    print(f"  空间范数均值: {spatial_norms.mean():.10f}")
    print(f"  空间范数最大值: {spatial_norms.max():.10f}")

    # 计算衰减比例
    decay_ratio = spatial_norms.mean() / init_spatial_norm
    print(f"\n空间范数衰减: {init_spatial_norm:.6f} → {spatial_norms.mean():.10f}")
    print(f"衰减比例: {decay_ratio:.6e} (保留了 {100*decay_ratio:.4f}%)")

    if decay_ratio > 0.01:
        print("\n✅ 修复有效！空间范数保留>1%")
    elif decay_ratio > 0.001:
        print("\n⚠️  部分改善，但仍有较大衰减")
    else:
        print("\n❌ 修复似乎无效，衰减仍然严重")

    # 检查唯一性
    unique_count = np.unique(segment_emb, axis=0).shape[0]
    print(f"\n唯一embedding数: {unique_count}/{segment_emb.shape[0]}")

    # 如果修复无效，检查第一层的中间输出
    if decay_ratio < 0.01:
        print("\n" + "="*80)
        print("详细调试第1层图编码器")
        print("="*80)

        with torch.no_grad():
            # 重新计算
            euclidean_feat = torch.cat([
                model.graph_enc.node_emb_layer(model.node_feature),
                model.graph_enc.type_emb_layer(model.type_feature),
                model.graph_enc.length_emb_layer(model.length_feature),
                model.graph_enc.lane_emb_layer(model.lane_feature)
            ], dim=-1)

            hyp_feat = model.graph_enc.hyp_embedding(euclidean_feat)

            print(f"\n输入到第1层:")
            print(f"  空间范数均值: {torch.norm(hyp_feat[:, 1:], dim=1).mean().item():.6f}")

            # 调用第一层
            hyp_feat_1 = model.graph_enc.tl_layer_1(
                model.graph_enc.struct_adj, hyp_feat, model.adj
            )

            print(f"\n第1层输出:")
            print(f"  空间范数均值: {torch.norm(hyp_feat_1[:, 1:], dim=1).mean().item():.10f}")

            layer1_decay = torch.norm(hyp_feat_1[:, 1:], dim=1).mean().item() / torch.norm(hyp_feat[:, 1:], dim=1).mean().item()
            print(f"  第1层衰减比例: {layer1_decay:.6e}")

if __name__ == '__main__':
    test_with_fresh_init()
