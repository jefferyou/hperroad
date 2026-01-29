#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超简化版诊断：直接触发entailment_loss计算来监控angle_between
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 全局变量记录cos_angle值
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

    # 确保Lorentz约束
    xx = torch.clamp(xx, max=-eps)
    yy = torch.clamp(yy, max=-eps)

    x_norm = torch.sqrt(-xx + min_norm)
    y_norm = torch.sqrt(-yy + min_norm)

    # 计算余弦值
    cos_angle = xy / (x_norm * y_norm + eps)

    # *** 记录cos_angle统计 ***
    with torch.no_grad():
        cos_angle_log.append({
            'min': cos_angle.min().item(),
            'max': cos_angle.max().item(),
            'mean': cos_angle.mean().item(),
            'out_of_range': ((cos_angle < -1.0) | (cos_angle > 1.0)).sum().item(),
            'total': cos_angle.numel(),
            'out_of_range_pct': 100.0 * ((cos_angle < -1.0) | (cos_angle > 1.0)).sum().item() / cos_angle.numel()
        })

    # 使用自定义Acos
    angle = Acos.apply(cos_angle)
    angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

    return angle

def run_diagnostic():
    """运行诊断"""
    print("="*80)
    print("快速诊断：直接触发compute_entailment_loss并监控angle_between")
    print("="*80)

    # Import
    from veccity.config import ConfigParser
    from veccity.data import get_dataset
    from veccity.utils import get_model

    # Patch angle_between
    from veccity.upstream.road_representation import hyperbolic_utils
    original_angle_between = hyperbolic_utils.EntailmentCone.angle_between
    hyperbolic_utils.EntailmentCone.angle_between = patched_angle_between

    print("\n✓ 已安装监控patch")

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

    # 直接触发compute_entailment_loss
    print("\n" + "="*80)
    print("触发compute_entailment_loss（会调用angle_between）...")
    print("="*80)

    model.eval()

    # 先做一次forward pass生成嵌入
    print("\n步骤1: 生成嵌入（调用encode）")
    try:
        # 从dataloader获取一个batch
        train_dataloader = dataset.train_dataloader
        train_set, train_label = next(iter(train_dataloader))
        train_set = train_set.clone().detach().to(device)

        # 调用encode生成嵌入
        _ = model.encode(train_set)
        print("✓ 嵌入已生成")

    except Exception as e:
        print(f"✗ 生成嵌入失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 现在调用compute_entailment_loss，这会触发angle_between
    print("\n步骤2: 调用compute_entailment_loss（会触发angle_between）")
    try:
        for i in range(10):  # 调用10次来收集数据
            loss = model.compute_entailment_loss()
            if (i + 1) % 3 == 0:
                print(f"  已调用 {i+1}/10 次，记录了 {len(cos_angle_log)} 次angle_between调用")

        print(f"\n✓ 成功调用compute_entailment_loss 10次")

    except Exception as e:
        print(f"\n✗ compute_entailment_loss失败: {e}")
        import traceback
        traceback.print_exc()

    # 恢复原始方法
    hyperbolic_utils.EntailmentCone.angle_between = original_angle_between

    # 分析结果
    print("\n" + "="*80)
    print("诊断结果")
    print("="*80)

    if len(cos_angle_log) == 0:
        print("❌ 没有记录到任何angle_between调用")
        return

    print(f"\n✓ 总共记录了 {len(cos_angle_log)} 次angle_between调用\n")

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

    # 显示详细记录
    print("\n前10次调用的cos_angle值:")
    print(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}")
    print("-"*60)
    for i, log in enumerate(cos_angle_log[:10]):
        print(f"{i:<6} {log['min']:<12.6f} {log['max']:<12.6f} {log['mean']:<12.6f} {log['out_of_range_pct']:<12.2f}%")

    # 判断
    print("\n" + "="*80)
    print("结论:")
    print("="*80)

    if min(all_mins) < -1.0 or max(all_maxs) > 1.0:
        print("✓✓✓ 确认问题: cos_angle值超出[-1, 1]范围！")
        print(f"  - 最小值达到 {min(all_mins):.6f} (应该 >= -1)")
        print(f"  - 最大值达到 {max(all_maxs):.6f} (应该 <= 1)")
        print(f"  - 平均有 {np.mean(all_out_pcts):.2f}% 的值超出范围")
        print("\n这说明angle_between的欧几里得角度公式不适用于Lorentz双曲空间！")
        print("Minkowski内积可以是任意负数，不受[-1,1]限制。")
        print("\n需要修复angle_between，使用正确的双曲几何公式。")
    else:
        print("✗ cos_angle值在[-1, 1]范围内")
        print("  问题可能在其他地方")

    # 保存日志
    log_file = 'cos_angle_diagnostic.txt'
    print(f"\n保存详细日志到 {log_file}")
    with open(log_file, 'w', encoding='utf-8') as f:
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

    print("✓ 诊断完成")

if __name__ == '__main__':
    run_diagnostic()
