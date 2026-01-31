#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查生成的嵌入是否正常
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_embeddings():
    """检查嵌入质量"""
    print("="*80)
    print("检查cd数据集HRNR_Hyperbolic生成的嵌入质量")
    print("="*80)

    from veccity.config import ConfigParser
    from veccity.data import get_dataset
    from veccity.utils import get_model

    # 配置
    print("\n加载配置和数据...")
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

    print(f"✓ 模型已移至设备: {device}")

    # 生成嵌入
    print("\n生成嵌入...")
    model.eval()

    with torch.no_grad():
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature, model.lane_feature, model.adj
        )

    # 检查各层嵌入
    segment_emb = model.graph_enc.segment_hyp_emb.cpu().numpy()
    locality_emb = model.graph_enc.locality_hyp_emb.cpu().numpy()
    region_emb = model.graph_enc.region_hyp_emb.cpu().numpy()

    print("\n" + "="*80)
    print("嵌入统计")
    print("="*80)

    for name, emb in [("Segment", segment_emb), ("Locality", locality_emb), ("Region", region_emb)]:
        print(f"\n{name} 嵌入:")
        print(f"  形状: {emb.shape}")
        print(f"  数值范围: [{emb.min():.6f}, {emb.max():.6f}]")
        print(f"  均值: {emb.mean():.6f}")
        print(f"  标准差: {emb.std():.6f}")

        # 检查时间分量（第一维）
        time_comp = emb[:, 0]
        print(f"  时间分量（第一维）:")
        print(f"    范围: [{time_comp.min():.6f}, {time_comp.max():.6f}]")
        print(f"    均值: {time_comp.mean():.6f}")

        # 检查是否所有点都相同
        unique_rows = np.unique(emb, axis=0).shape[0]
        print(f"  唯一点数量: {unique_rows} / {emb.shape[0]}")
        if unique_rows == 1:
            print(f"  ❌❌❌ 所有嵌入点完全相同！")
        elif unique_rows < emb.shape[0] * 0.1:
            print(f"  ⚠️  嵌入缺乏多样性（只有{unique_rows}个唯一点）")

        # 检查Lorentz约束: <x, x> = -1
        minkowski_dot = -emb[:, 0]**2 + np.sum(emb[:, 1:]**2, axis=1)
        print(f"  Lorentz约束 <x,x>:")
        print(f"    范围: [{minkowski_dot.min():.6f}, {minkowski_dot.max():.6f}]")
        print(f"    均值: {minkowski_dot.mean():.6f}")
        print(f"    与-1的平均偏差: {np.abs(minkowski_dot + 1.0).mean():.6f}")

        # 检查点之间的相似度
        if emb.shape[0] > 1:
            # 计算前10个点之间的Minkowski内积
            n_samples = min(10, emb.shape[0])
            sample_emb = emb[:n_samples]

            print(f"  前{n_samples}个点之间的Minkowski内积:")
            for i in range(min(3, n_samples)):
                for j in range(i+1, min(3, n_samples)):
                    x, y = sample_emb[i], sample_emb[j]
                    xy = -x[0]*y[0] + np.sum(x[1:]*y[1:])
                    xx = -x[0]**2 + np.sum(x[1:]**2)
                    yy = -y[0]**2 + np.sum(y[1:]**2)
                    cos_angle = xy / (np.sqrt(-xx) * np.sqrt(-yy) + 1e-7)
                    print(f"    点{i}-点{j}: <x,y>={xy:.6f}, cos_angle={cos_angle:.6f}")

    print("\n" + "="*80)
    print("结论:")
    print("="*80)

    # 判断
    if unique_rows == 1:
        print("\n❌❌❌ 致命问题：所有嵌入点完全相同！")
        print("这解释了为什么所有angle_between的cos_angle都是-1.0")
        print("嵌入生成过程有严重bug，需要检查：")
        print("  1. 嵌入初始化")
        print("  2. exp_map投影到Lorentz流形")
        print("  3. 图编码器的前向传播")
    elif np.all(np.abs(minkowski_dot + 1.0) > 0.1):
        print("\n❌ Lorentz约束严重违背！")
        print("嵌入不在Lorentz流形上，计算的角度没有意义")
    else:
        print("\n嵌入看起来基本正常")
        print("问题可能在其他地方")

if __name__ == '__main__':
    check_embeddings()
