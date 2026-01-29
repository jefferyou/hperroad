#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试：验证修复后的代码是否被调用
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_fix():
    """检查修复代码是否被调用"""
    print("="*80)
    print("调试：检查修复后的代码是否被调用")
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

    # 检查 _aggregate_to_cluster 的源代码
    import inspect
    source = inspect.getsource(model.graph_enc.tl_layer_1._aggregate_to_cluster)

    print("\n当前 _aggregate_to_cluster 的源代码:")
    print("-" * 80)
    print(source[:500])  # 只打印前500个字符
    print("-" * 80)

    if "log_map" in source and "exp_map" in source:
        print("\n✅ 代码已更新！包含 log_map 和 exp_map")
    else:
        print("\n❌ 代码未更新！仍然是旧的矩阵乘法版本")
        return

    # 添加调试钩子
    original_aggregate = model.graph_enc.tl_layer_1._aggregate_to_cluster

    def debug_aggregate(embeddings, assignment_matrix):
        print(f"\n[DEBUG] _aggregate_to_cluster 被调用")
        print(f"  输入embeddings形状: {embeddings.shape}")
        print(f"  输入空间范数: {torch.norm(embeddings[:, 1:], dim=1).mean().item():.6f}")

        result = original_aggregate(embeddings, assignment_matrix)

        print(f"  输出形状: {result.shape}")
        print(f"  输出空间范数: {torch.norm(result[:, 1:], dim=1).mean().item():.6f}")

        return result

    model.graph_enc.tl_layer_1._aggregate_to_cluster = debug_aggregate

    print("\n运行一次前向传播...")
    with torch.no_grad():
        model.node_emb = model.graph_enc(
            model.node_feature, model.type_feature, model.length_feature, model.lane_feature, model.adj
        )

    print("\n调试完成")

if __name__ == '__main__':
    debug_fix()
