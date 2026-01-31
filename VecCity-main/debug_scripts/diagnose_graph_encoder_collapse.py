#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断图编码器中的空间分量坍缩
追踪每一层的输出，找出坍塌发生在哪一层
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_graph_encoder():
    """检查图编码器每一层的输出"""
    print("="*80)
    print("诊断：图编码器中空间分量在哪一层坍缩")
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

    def analyze_embedding(emb, name):
        """分析一个embedding的统计信息"""
        emb_np = emb.detach().cpu().numpy()
        time_comp = emb_np[:, 0]
        spatial_comp = emb_np[:, 1:]
        spatial_norms = np.linalg.norm(spatial_comp, axis=1)

        print(f"\n{name}:")
        print(f"  形状: {emb_np.shape}")
        print(f"  时间分量: [{time_comp.min():.6f}, {time_comp.max():.6f}], 均值={time_comp.mean():.6f}")
        print(f"  空间分量范围: [{spatial_comp.min():.10f}, {spatial_comp.max():.10f}]")
        print(f"  空间范数: [{spatial_norms.min():.10f}, {spatial_norms.max():.10f}], 均值={spatial_norms.mean():.10f}")

        # 检查Lorentz约束
        lorentz_constraint = -time_comp**2 + np.sum(spatial_comp**2, axis=1)
        print(f"  Lorentz <x,x>: 偏差均值={np.abs(lorentz_constraint + 1.0).mean():.10f}")

        # 检查是否在原点附近
        near_origin_count = np.sum(spatial_norms < 1e-3)
        print(f"  接近原点 (空间范数<0.001): {near_origin_count}/{emb_np.shape[0]} ({100*near_origin_count/emb_np.shape[0]:.1f}%)")

        return emb_np, spatial_norms

    print("\n" + "="*80)
    print("步骤1: 初始双曲嵌入（HyperbolicEmbedding层输出）")
    print("="*80)

    with torch.no_grad():
        # 获取欧氏特征
        node_emb = model.graph_enc.node_emb_layer(model.node_feature)
        type_emb = model.graph_enc.type_emb_layer(model.type_feature)
        length_emb = model.graph_enc.length_emb_layer(model.length_feature)
        lane_emb = model.graph_enc.lane_emb_layer(model.lane_feature)
        euclidean_feat = torch.cat([node_emb, type_emb, length_emb, lane_emb], dim=-1)

        # 初始双曲嵌入
        hyp_feat_init = model.graph_enc.hyp_embedding(euclidean_feat)

    emb0, norms0 = analyze_embedding(hyp_feat_init, "初始双曲嵌入")

    print("\n" + "="*80)
    print("步骤2: 第1层图编码器输出")
    print("="*80)

    with torch.no_grad():
        hyp_feat_1 = model.graph_enc.tl_layer_1(
            model.graph_enc.struct_adj, hyp_feat_init, model.adj
        )

    emb1, norms1 = analyze_embedding(hyp_feat_1, "第1层输出")

    # 检查变化
    norm_ratio_1 = norms1.mean() / norms0.mean()
    print(f"\n  空间范数变化: {norms0.mean():.6f} → {norms1.mean():.6f} (比例: {norm_ratio_1:.6f})")
    if norm_ratio_1 < 0.1:
        print(f"  ⚠️  第1层导致空间范数大幅减小 ({norm_ratio_1:.2%})!")

    print("\n" + "="*80)
    print("步骤3: 第2层图编码器输出")
    print("="*80)

    with torch.no_grad():
        hyp_feat_2 = model.graph_enc.tl_layer_2(
            model.graph_enc.struct_adj, hyp_feat_1, model.adj
        )

    emb2, norms2 = analyze_embedding(hyp_feat_2, "第2层输出")

    norm_ratio_2 = norms2.mean() / norms1.mean()
    print(f"\n  空间范数变化: {norms1.mean():.6f} → {norms2.mean():.6f} (比例: {norm_ratio_2:.6f})")
    if norm_ratio_2 < 0.1:
        print(f"  ⚠️  第2层导致空间范数大幅减小 ({norm_ratio_2:.2%})!")

    print("\n" + "="*80)
    print("步骤4: 第3层图编码器输出")
    print("="*80)

    with torch.no_grad():
        hyp_feat_3 = model.graph_enc.tl_layer_3(
            model.graph_enc.struct_adj, hyp_feat_2, model.adj
        )

    emb3, norms3 = analyze_embedding(hyp_feat_3, "第3层输出")

    norm_ratio_3 = norms3.mean() / norms2.mean()
    print(f"\n  空间范数变化: {norms2.mean():.6f} → {norms3.mean():.6f} (比例: {norm_ratio_3:.6f})")
    if norm_ratio_3 < 0.1:
        print(f"  ⚠️  第3层导致空间范数大幅减小 ({norm_ratio_3:.2%})!")

    print("\n" + "="*80)
    print("步骤5: 最终segment_hyp_emb")
    print("="*80)

    # 运行完整的forward
    with torch.no_grad():
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature, model.lane_feature, model.adj
        )
        segment_emb_final = model.graph_enc.segment_hyp_emb

    emb_final, norms_final = analyze_embedding(segment_emb_final, "最终segment_hyp_emb")

    print("\n" + "="*80)
    print("汇总：空间范数变化轨迹")
    print("="*80)

    print(f"\n初始嵌入:     {norms0.mean():.10f}")
    print(f"第1层后:       {norms1.mean():.10f}  (×{norm_ratio_1:.6f})")
    print(f"第2层后:       {norms2.mean():.10f}  (×{norm_ratio_2:.6f})")
    print(f"第3层后:       {norms3.mean():.10f}  (×{norm_ratio_3:.6f})")
    print(f"最终输出:      {norms_final.mean():.10f}  (×{norms_final.mean()/norms3.mean():.6f})")

    total_ratio = norms_final.mean() / norms0.mean()
    print(f"\n总体变化: {norms0.mean():.6f} → {norms_final.mean():.10f}  (×{total_ratio:.2e})")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    # 找出导致最大衰减的层
    ratios = {
        "第1层": norm_ratio_1,
        "第2层": norm_ratio_2,
        "第3层": norm_ratio_3,
    }

    min_layer = min(ratios, key=ratios.get)
    min_ratio = ratios[min_layer]

    if total_ratio < 1e-3:
        print(f"\n❌ 确认：空间分量在图编码器中严重坍缩")
        print(f"\n罪魁祸首: {min_layer} (空间范数衰减到 {min_ratio:.2%})")
        print(f"\n需要检查 {min_layer} 的:")
        print(f"  1. HyperbolicGraphConv 的实现")
        print(f"  2. 聚合操作 (_aggregate_to_cluster)")
        print(f"  3. 门控机制")
        print(f"  4. 是否有数值下溢")
    else:
        print(f"\n空间分量没有严重坍缩，问题可能在其他地方")

    print("\n保存详细数据到 graph_encoder_diagnosis.npz")
    np.savez('graph_encoder_diagnosis.npz',
             emb0=emb0, norms0=norms0,
             emb1=emb1, norms1=norms1,
             emb2=emb2, norms2=norms2,
             emb3=emb3, norms3=norms3,
             emb_final=emb_final, norms_final=norms_final)
    print("✓ 完成")

if __name__ == '__main__':
    diagnose_graph_encoder()
