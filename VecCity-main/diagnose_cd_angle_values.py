#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断cd数据集训练时angle_between中cos_angle的实际值分布
监控训练过程中是否有超出[-1,1]范围的值
"""

import torch
import numpy as np
import sys
import os

# Monkey patch angle_between to log cos_angle values
original_angle_between = None
cos_angle_log = []

def patched_angle_between(self, x, y):
    """带监控的angle_between"""
    from veccity.upstream.road_representation.hyperbolic_optimizations import AdaptiveEpsilon, Acos

    eps = AdaptiveEpsilon.get_eps(x)
    min_norm = AdaptiveEpsilon.get_min_norm(x)

    # 计算Minkowski内积
    xy = self.manifold.minkowski_dot(x, y, keepdim=False)
    xx = self.manifold.minkowski_dot(x, x, keepdim=False)
    yy = self.manifold.minkowski_dot(y, y, keepdim=False)

    # 确保Lorentz约束（负数）
    xx = torch.clamp(xx, max=-eps)
    yy = torch.clamp(yy, max=-eps)

    x_norm = torch.sqrt(-xx + min_norm)
    y_norm = torch.sqrt(-yy + min_norm)

    # 计算余弦值
    cos_angle = xy / (x_norm * y_norm + eps)

    # *** 记录cos_angle的统计信息 ***
    with torch.no_grad():
        cos_min = cos_angle.min().item()
        cos_max = cos_angle.max().item()
        cos_mean = cos_angle.mean().item()
        out_of_range_count = ((cos_angle < -1.0) | (cos_angle > 1.0)).sum().item()
        total_count = cos_angle.numel()

        cos_angle_log.append({
            'min': cos_min,
            'max': cos_max,
            'mean': cos_mean,
            'out_of_range': out_of_range_count,
            'total': total_count,
            'out_of_range_pct': 100.0 * out_of_range_count / total_count if total_count > 0 else 0
        })

    # 使用自定义Acos（带梯度截断和数值稳定性保护）
    angle = Acos.apply(cos_angle)

    # 确保angle是有限的
    angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

    return angle

def run_diagnostic():
    """运行诊断"""
    print("="*80)
    print("诊断cd数据集训练时angle_between中cos_angle的值分布")
    print("="*80)

    # Import after path setup
    from veccity.utils import get_model, get_executor, get_config
    from veccity.data import get_dataset

    # Patch angle_between
    from veccity.upstream.road_representation import hyperbolic_utils
    global original_angle_between
    original_angle_between = hyperbolic_utils.EntailmentCone.angle_between
    hyperbolic_utils.EntailmentCone.angle_between = patched_angle_between

    print("\n已安装监控patch，将在训练过程中记录cos_angle值")

    # 配置
    config = get_config(
        task='road_representation',
        model='HRNR_Hyperbolic',
        dataset='cd',
        config_file=None,
        saved_model=False,
        train=True,
        other_args={'gpu': True, 'gpu_id': 0, 'max_epoch': 5}  # 只训练5个epoch用于诊断
    )

    config['max_epoch'] = 5  # 限制训练轮数

    # 加载数据和模型
    dataset = get_dataset(config)
    model = get_model(config, dataset.get_data_feature())
    executor = get_executor(config, model, dataset)

    print("\n开始训练（仅5个epoch用于诊断）...")
    print("="*80)

    # 训练
    executor.train(model, dataset)

    # 分析结果
    print("\n" + "="*80)
    print("诊断结果")
    print("="*80)

    if len(cos_angle_log) == 0:
        print("❌ 没有记录到任何angle_between调用")
        print("   这可能说明EntailmentCone没有被使用，或者patch失败")
        return

    print(f"\n总共记录了 {len(cos_angle_log)} 次angle_between调用\n")

    # 统计
    all_mins = [log['min'] for log in cos_angle_log]
    all_maxs = [log['max'] for log in cos_angle_log]
    all_means = [log['mean'] for log in cos_angle_log]
    all_out_pcts = [log['out_of_range_pct'] for log in cos_angle_log]

    print("cos_angle值的总体统计:")
    print(f"  全局最小值: {min(all_mins):.6f}")
    print(f"  全局最大值: {max(all_maxs):.6f}")
    print(f"  平均值范围: [{min(all_means):.6f}, {max(all_means):.6f}]")
    print(f"  超出[-1,1]范围的比例: {np.mean(all_out_pcts):.2f}%")

    # 详细记录（显示前10次和后10次）
    print("\n前10次调用的cos_angle值:")
    print(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}")
    print("-"*80)
    for i, log in enumerate(cos_angle_log[:10]):
        print(f"{i:<6} {log['min']:<12.6f} {log['max']:<12.6f} {log['mean']:<12.6f} {log['out_of_range_pct']:<12.2f}%")

    if len(cos_angle_log) > 10:
        print("\n...")
        print(f"\n后10次调用的cos_angle值:")
        print(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}")
        print("-"*80)
        for i, log in enumerate(cos_angle_log[-10:], len(cos_angle_log)-10):
            print(f"{i:<6} {log['min']:<12.6f} {log['max']:<12.6f} {log['mean']:<12.6f} {log['out_of_range_pct']:<12.2f}%")

    # 判断
    print("\n" + "="*80)
    print("结论:")
    print("="*80)

    if min(all_mins) < -1.0 or max(all_maxs) > 1.0:
        print("✓ 确认问题: cos_angle值超出[-1, 1]范围！")
        print(f"  - 最小值达到 {min(all_mins):.6f} (应该 >= -1)")
        print(f"  - 最大值达到 {max(all_maxs):.6f} (应该 <= 1)")
        print(f"  - 平均有 {np.mean(all_out_pcts):.2f}% 的值超出范围")
        print("\n这说明angle_between的欧几里得角度公式不适用于Lorentz双曲空间！")
        print("需要使用正确的双曲几何公式。")
    else:
        print("✗ cos_angle值在[-1, 1]范围内")
        print("  问题可能不在angle_between的数学公式")

    print("\n保存日志到 cos_angle_diagnostic.txt")
    with open('cos_angle_diagnostic.txt', 'w') as f:
        f.write("angle_between cos_angle值诊断日志\n")
        f.write("="*80 + "\n\n")
        f.write(f"总调用次数: {len(cos_angle_log)}\n")
        f.write(f"全局最小值: {min(all_mins):.6f}\n")
        f.write(f"全局最大值: {max(all_maxs):.6f}\n")
        f.write(f"超出范围比例: {np.mean(all_out_pcts):.2f}%\n\n")
        f.write("详细记录:\n")
        f.write(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}\n")
        f.write("-"*80 + "\n")
        for i, log in enumerate(cos_angle_log):
            f.write(f"{i:<6} {log['min']:<12.6f} {log['max']:<12.6f} {log['mean']:<12.6f} {log['out_of_range_pct']:<12.2f}%\n")

if __name__ == '__main__':
    run_diagnostic()
