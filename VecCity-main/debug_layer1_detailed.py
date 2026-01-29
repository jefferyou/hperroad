#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细追踪第1层每个操作的范数变化
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
import torch.nn.functional as F

def debug_layer1_detailed():
    """详细追踪第1层每个操作"""
    print("="*80)
    print("详细追踪第1层每个操作的范数变化")
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

    layer = model.graph_enc.tl_layer_1

    def get_spatial_norm(x):
        """计算空间分量范数"""
        if x.dim() == 1:
            return torch.norm(x[1:]).item()
        return torch.norm(x[:, 1:], dim=1).mean().item()

    print("\n准备输入数据...")
    with torch.no_grad():
        # 准备输入
        node_emb = model.graph_enc.node_emb_layer(model.node_feature)
        type_emb = model.graph_enc.type_emb_layer(model.type_feature)
        length_emb = model.graph_enc.length_emb_layer(model.length_feature)
        lane_emb = model.graph_enc.lane_emb_layer(model.lane_feature)
        euclidean_feat = torch.cat([node_emb, type_emb, length_emb, lane_emb], dim=-1)

        hyp_feat = model.graph_enc.hyp_embedding(euclidean_feat)

        print(f"初始输入空间范数: {get_spatial_norm(hyp_feat):.6f}")

        # 手动执行第1层的每一步
        struct_assign_norm = layer.struct_assign / (F.relu(torch.sum(layer.struct_assign, 0) - 1.0) + 1.0)
        fnc_assign_norm = layer.fnc_assign / (F.relu(torch.sum(layer.fnc_assign, 0) - 1.0) + 1.0)

        print("\n" + "="*80)
        print("步骤1: Segment -> Locality 聚合")
        print("="*80)
        struct_emb = layer._aggregate_to_cluster(hyp_feat, struct_assign_norm)
        print(f"聚合后: {get_spatial_norm(struct_emb):.6f}")

        print("\n" + "="*80)
        print("步骤2: Locality -> Region 聚合")
        print("="*80)
        fnc_emb = layer._aggregate_to_cluster(struct_emb, fnc_assign_norm)
        print(f"聚合后: {get_spatial_norm(fnc_emb):.6f}")

        print("\n" + "="*80)
        print("步骤3: F2F 图卷积 (Region内部)")
        print("="*80)
        fnc_adj = layer._compute_hyperbolic_affinity(fnc_emb)
        fnc_adj = fnc_adj + torch.eye(fnc_adj.shape[0]).to(device) * 1.0
        print(f"图卷积前: {get_spatial_norm(fnc_emb):.6f}")
        fnc_emb = layer.fnc_gcn(fnc_emb, fnc_adj)
        print(f"图卷积后: {get_spatial_norm(fnc_emb):.6f}")
        ratio_3 = get_spatial_norm(fnc_emb) / 4.19  # 聚合后的值
        print(f"保留率: {100*ratio_3:.1f}%")

        print("\n" + "="*80)
        print("步骤4: F2C 分发 (Region -> Locality)")
        print("="*80)
        fnc_message = layer._distribute_from_cluster(fnc_emb, layer.fnc_assign, fnc_assign_norm)
        print(f"分发后: {get_spatial_norm(fnc_message):.6f}")

        print("\n" + "="*80)
        print("步骤5: 门控 + Hyperbolic Update (Locality)")
        print("="*80)
        print(f"更新前 struct_emb: {get_spatial_norm(struct_emb):.6f}")
        print(f"fnc_message: {get_spatial_norm(fnc_message):.6f}")
        struct_emb = layer._hyperbolic_update(struct_emb, fnc_message, weight=0.15)
        print(f"更新后 struct_emb: {get_spatial_norm(struct_emb):.6f}")

        print("\n" + "="*80)
        print("步骤6: C2C 图卷积 (Locality内部)")
        print("="*80)
        struct_adj_processed = F.relu(model.graph_enc.struct_adj - torch.eye(model.graph_enc.struct_adj.shape[1]).to(device) * 10000.0) + \
                              torch.eye(model.graph_enc.struct_adj.shape[1]).to(device) * 1.0
        print(f"图卷积前: {get_spatial_norm(struct_emb):.6f}")
        struct_emb = layer.struct_gcn(struct_emb, struct_adj_processed)
        print(f"图卷积后: {get_spatial_norm(struct_emb):.6f}")

        print("\n" + "="*80)
        print("步骤7: C2N 分发 (Locality -> Segment)")
        print("="*80)
        struct_message = layer._distribute_from_cluster(struct_emb, layer.struct_assign, struct_assign_norm)
        print(f"分发后: {get_spatial_norm(struct_message):.6f}")

        print("\n" + "="*80)
        print("步骤8: 门控 + Hyperbolic Update (Segment)")
        print("="*80)
        print(f"更新前 hyp_feat: {get_spatial_norm(hyp_feat):.6f}")
        print(f"struct_message: {get_spatial_norm(struct_message):.6f}")
        hyp_feat = layer._hyperbolic_update(hyp_feat, struct_message, weight=0.5)
        print(f"更新后 hyp_feat: {get_spatial_norm(hyp_feat):.6f}")

        print("\n" + "="*80)
        print("步骤9: N2N 图卷积 (Segment内部)")
        print("="*80)
        print(f"图卷积前: {get_spatial_norm(hyp_feat):.6f}")
        hyp_feat = layer.node_gcn(hyp_feat, model.adj)
        print(f"图卷积后: {get_spatial_norm(hyp_feat):.6f}")

        print("\n" + "="*80)
        print("总结")
        print("="*80)
        print(f"最终输出: {get_spatial_norm(hyp_feat):.10f}")

if __name__ == '__main__':
    debug_layer1_detailed()
