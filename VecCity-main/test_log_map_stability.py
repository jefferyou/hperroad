#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 log_map 在原点附近的数值稳定性
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_log_map_near_origin():
    """测试log_map在原点附近的行为"""
    print("="*80)
    print("测试 log_map 在原点附近的数值稳定性")
    print("="*80)

    from veccity.upstream.road_representation.hyperbolic_utils import LorentzManifold

    manifold = LorentzManifold()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Lorentz原点
    origin = torch.zeros(1, 225, device=device)
    origin[0, 0] = 1.0

    print("\n测试1: log_map(origin, origin)")
    print("-" * 40)
    v = manifold.log_map(origin, origin)
    print(f"结果: {v[0, :5]}")  # 只打印前5维
    print(f"范数: {torch.norm(v).item():.10e}")

    print("\n测试2: log_map(origin, 原点附近的点)")
    print("-" * 40)

    # 创建几个原点附近的点
    for scale in [0.001, 0.01, 0.1, 1.0]:
        # 创建一个离原点距离为scale的点
        spatial = torch.randn(1, 224, device=device) * scale
        point = manifold.project_to_lorentz(spatial)

        spatial_norm = torch.norm(point[:, 1:]).item()

        # log_map
        v = manifold.log_map(origin.expand_as(point), point)
        v_norm = torch.norm(v).item()

        print(f"\n空间分量范数 {spatial_norm:.6f}:")
        print(f"  切向量范数: {v_norm:.10f}")
        print(f"  比例: {v_norm/spatial_norm:.6f}")

    print("\n测试3: 批量log_map")
    print("-" * 40)

    # 创建10个原点附近的点（模拟实际情况）
    batch_size = 10
    spatial_batch = torch.randn(batch_size, 224, device=device) * 0.01
    points_batch = manifold.project_to_lorentz(spatial_batch)

    print(f"输入点空间范数: [{torch.norm(points_batch[:, 1:], dim=1).min().item():.6f}, "
          f"{torch.norm(points_batch[:, 1:], dim=1).max().item():.6f}]")

    # 批量log_map
    v_batch = manifold.log_map(
        origin.expand(batch_size, -1),
        points_batch
    )

    v_norms = torch.norm(v_batch, dim=1)
    print(f"切向量范数: [{v_norms.min().item():.10f}, {v_norms.max().item():.10f}]")
    print(f"切向量范数均值: {v_norms.mean().item():.10f}")

    # 平均后exp_map回去
    avg_v = v_batch.mean(dim=0, keepdim=True)
    print(f"\n平均切向量范数: {torch.norm(avg_v).item():.10f}")

    # exp_map回到流形
    result_point = manifold.exp_map(origin, avg_v)
    result_spatial_norm = torch.norm(result_point[:, 1:]).item()

    print(f"exp_map结果空间范数: {result_spatial_norm:.10f}")
    print(f"原始平均空间范数: {torch.norm(points_batch[:, 1:], dim=1).mean().item():.10f}")

    ratio = result_spatial_norm / torch.norm(points_batch[:, 1:], dim=1).mean().item()
    print(f"范数保留比例: {ratio:.6f}")

    if ratio < 0.1:
        print(f"\n⚠️  警告：log_map → 平均 → exp_map 导致{100*(1-ratio):.1f}%的信息丢失！")

        # 检查是否是eps导致的问题
        print(f"\n调试信息:")
        xy = manifold.minkowski_dot(origin.expand_as(points_batch), points_batch, keepdim=True)
        print(f"  Minkowski内积 <origin, points>: [{xy.min().item():.10f}, {xy.max().item():.10f}]")

        from veccity.upstream.road_representation.hyperbolic_optimizations import Arcosh, sinh, AdaptiveEpsilon
        eps = AdaptiveEpsilon.get_eps(points_batch)
        xy_clamped = torch.clamp(xy, max=-1.0 - eps)
        acosh_input = -xy_clamped
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)
        dist = Arcosh.apply(acosh_input)

        print(f"  距离 dist: [{dist.min().item():.10e}, {dist.max().item():.10e}]")
        print(f"  sinh(dist): [{sinh(dist).min().item():.10e}, {sinh(dist).max().item():.10e}]")
        print(f"  eps: {eps}")

        coef = dist / (sinh(dist) + eps)
        print(f"  系数 coef: [{coef.min().item():.6f}, {coef.max().item():.6f}]")

        # 问题可能在这里：当dist很小时，sinh(dist) ≈ dist，所以coef ≈ 1
        # 但如果加了eps，coef = dist/(dist+eps) < 1，可能导致切向量被缩小

        if eps > dist.mean().item():
            print(f"\n⚠️  发现问题：eps={eps:.2e} > 平均距离={dist.mean().item():.2e}")
            print(f"     这会导致coef = dist/(dist+eps)显著<1，切向量被不当缩小！")

if __name__ == '__main__':
    test_log_map_near_origin()
