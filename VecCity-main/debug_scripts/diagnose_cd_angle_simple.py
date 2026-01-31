#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版诊断脚本：直接运行训练并监控angle_between的cos_angle值
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

    # 使用自定义Acos
    angle = Acos.apply(cos_angle)
    angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

    return angle

def run_diagnostic():
    """运行诊断"""
    print("="*80)
    print("简化版诊断：监控cd数据集训练时angle_between的cos_angle值")
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
    print("\n加载配置...")
    config = ConfigParser(
        task='segment',
        model='HRNR_Hyperbolic',
        dataset='cd',
        config_file=None,
        saved_model=False,
        train=True,
        other_args={'gpu': True, 'gpu_id': 0}
    )

    # 加载数据和模型
    print("加载数据集...")
    dataset = get_dataset(config)

    print("创建模型...")
    model = get_model(config, dataset.get_data_feature())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"✓ 模型已移至设备: {device}")

    # 直接调用模型的训练方法，只训练几个batch
    print("\n" + "="*80)
    print("开始训练监控（只运行20个batch用于快速诊断）...")
    print("="*80)

    model.train()

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    # 运行几个batch
    batch_count = 0
    max_batches = 20

    try:
        # 从数据集获取数据（这部分需要根据实际的数据加载方式调整）
        # 由于HRNR_Hyperbolic直接使用图数据，我们直接调用forward
        for i in range(max_batches):
            try:
                # 前向传播（这会触发angle_between调用）
                output = model()

                # 计算一个简单的loss（只是为了触发反向传播）
                if hasattr(output, 'mean'):
                    loss = output.mean()
                else:
                    loss = torch.tensor(0.0, device=device)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1

                if batch_count % 5 == 0:
                    print(f"  已完成 {batch_count}/{max_batches} 个batch，记录了 {len(cos_angle_log)} 次angle_between调用")

            except Exception as e:
                print(f"  Batch {i} 出错（继续）: {e}")
                continue

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n训练过程出错: {e}")
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
        print("   这可能说明:")
        print("   1. EntailmentCone在训练中未被使用")
        print("   2. 模型forward过程没有调用angle_between")
        print("   3. 训练过程出现错误")
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

    # 显示部分详细记录
    print("\n前10次调用的cos_angle值:")
    print(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}")
    print("-"*60)
    for i, log in enumerate(cos_angle_log[:10]):
        print(f"{i:<6} {log['min']:<12.6f} {log['max']:<12.6f} {log['mean']:<12.6f} {log['out_of_range_pct']:<12.2f}%")

    if len(cos_angle_log) > 20:
        print("\n...")
        print(f"\n后10次调用的cos_angle值:")
        print(f"{'序号':<6} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'超出比例':<12}")
        print("-"*60)
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
        print("Minkowski内积可以是任意负数，不受[-1,1]限制。")
        print("需要使用正确的双曲几何公式来计算角度。")
    else:
        print("✗ cos_angle值在[-1, 1]范围内")
        print("  问题可能不在angle_between的数学公式")

    # 保存详细日志
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
