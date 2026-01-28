#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同版本的angle_between实现对cd数据集的影响
"""

import torch
import numpy as np

# 模拟Lorentz manifold的minkowski_dot
def minkowski_dot(x, y, keepdim=False):
    """Minkowski inner product: -x[0]*y[0] + sum(x[1:]*y[1:])"""
    result = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    if keepdim:
        result = result.unsqueeze(-1)
    return result

# 原始版本：没有自定义Acos
def angle_between_original(x, y):
    """原始版本（e8357f4之前）"""
    xy = minkowski_dot(x, y, keepdim=False)
    x_norm = torch.sqrt(-minkowski_dot(x, x, keepdim=False))
    y_norm = torch.sqrt(-minkowski_dot(y, y, keepdim=False))

    cos_angle = xy / (x_norm * y_norm + 1e-6)
    cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-5, max=1.0 - 1e-5)

    angle = torch.acos(cos_angle)
    return angle, cos_angle

# 版本1：自定义Acos，严格clamp (commit 1255ef6)
class AcosStrict(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=-0.9999, max=0.9999)
        ctx.save_for_backward(x)
        z = x.double()
        result = torch.acos(z)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        denominator = torch.sqrt(1.0 - x * x + 1e-8)
        grad = -grad_output / denominator
        grad = torch.clamp(grad, min=-10.0, max=10.0)
        return grad

def angle_between_strict(x, y):
    """严格clamp版本 (commit 1255ef6)"""
    xy = minkowski_dot(x, y, keepdim=False)
    xx = minkowski_dot(x, x, keepdim=False)
    yy = minkowski_dot(y, y, keepdim=False)

    xx = torch.clamp(xx, max=-1e-7)
    yy = torch.clamp(yy, max=-1e-7)

    x_norm = torch.sqrt(-xx + 1e-15)
    y_norm = torch.sqrt(-yy + 1e-15)

    cos_angle = xy / (x_norm * y_norm + 1e-7)
    angle = AcosStrict.apply(cos_angle)
    angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

    return angle, cos_angle

# 版本2：自定义Acos，宽松clamp (commit e47e18c)
class AcosRelaxed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=-1.0 + 1e-5, max=1.0 - 1e-5)
        ctx.save_for_backward(x)
        z = x.double()
        result = torch.acos(z)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        denominator = torch.sqrt(1.0 - x * x + 1e-8)
        grad = -grad_output / denominator
        grad = torch.clamp(grad, min=-50.0, max=50.0)
        return grad

def angle_between_relaxed(x, y):
    """宽松clamp版本 (commit e47e18c)"""
    xy = minkowski_dot(x, y, keepdim=False)
    xx = minkowski_dot(x, x, keepdim=False)
    yy = minkowski_dot(y, y, keepdim=False)

    xx = torch.clamp(xx, max=-1e-7)
    yy = torch.clamp(yy, max=-1e-7)

    x_norm = torch.sqrt(-xx + 1e-15)
    y_norm = torch.sqrt(-yy + 1e-15)

    cos_angle = xy / (x_norm * y_norm + 1e-7)
    angle = AcosRelaxed.apply(cos_angle)
    angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

    return angle, cos_angle

def test_versions():
    """测试不同版本在模拟cd数据集特征上的表现"""
    print("="*80)
    print("测试angle_between不同版本的行为")
    print("="*80)

    # 模拟cd数据集的特征：层次结构复杂，cos_angle经常接近±1
    torch.manual_seed(42)

    # 生成一些测试点（Lorentz空间）
    n = 1000
    d = 224  # 空间维度

    # 生成满足Lorentz约束的点: x[0]^2 - sum(x[1:]^2) = 1
    spatial = torch.randn(n, d) * 0.1  # 空间部分
    temporal = torch.sqrt(1.0 + torch.sum(spatial**2, dim=1, keepdim=True))  # 时间部分
    points = torch.cat([temporal, spatial], dim=1)  # [n, d+1]

    # 测试用例1：相近的点（cos_angle接近1）
    x_near = points[:100]
    y_near = points[:100] + torch.randn(100, d+1) * 0.01
    # 重新normalize到Lorentz manifold
    y_near_spatial = y_near[:, 1:]
    y_near_temporal = torch.sqrt(1.0 + torch.sum(y_near_spatial**2, dim=1, keepdim=True))
    y_near = torch.cat([y_near_temporal, y_near_spatial], dim=1)

    # 测试用例2：相反的点（cos_angle接近-1）
    x_far = points[100:200]
    y_far = points[100:200]
    y_far_spatial = -y_far[:, 1:] * 0.9
    y_far_temporal = torch.sqrt(1.0 + torch.sum(y_far_spatial**2, dim=1, keepdim=True))
    y_far = torch.cat([y_far_temporal, y_far_spatial], dim=1)

    # 测试用例3：一般的点
    x_normal = points[200:300]
    y_normal = points[300:400]

    test_cases = [
        ("相近点 (cos≈1)", x_near, y_near),
        ("相反点 (cos≈-1)", x_far, y_far),
        ("一般点", x_normal, y_normal)
    ]

    versions = [
        ("原始版本 (no custom Acos)", angle_between_original),
        ("严格clamp版本 (1255ef6)", angle_between_strict),
        ("宽松clamp版本 (e47e18c)", angle_between_relaxed)
    ]

    for case_name, x, y in test_cases:
        print(f"\n{'='*80}")
        print(f"测试用例: {case_name}")
        print('='*80)

        for version_name, angle_fn in versions:
            try:
                x_req = x.clone().requires_grad_(True)
                y_req = y.clone().requires_grad_(True)

                angle, cos_angle = angle_fn(x_req, y_req)

                # 测试反向传播
                loss = angle.sum()
                loss.backward()

                print(f"\n{version_name}:")
                print(f"  cos_angle范围: [{cos_angle.min().item():.6f}, {cos_angle.max().item():.6f}]")
                print(f"  angle范围: [{angle.min().item():.6f}, {angle.max().item():.6f}]")
                print(f"  x梯度: min={x_req.grad.min().item():.6f}, max={x_req.grad.max().item():.6f}, mean={x_req.grad.mean().item():.6f}")
                print(f"  y梯度: min={y_req.grad.min().item():.6f}, max={y_req.grad.max().item():.6f}, mean={y_req.grad.mean().item():.6f}")

                # 检查是否有NaN或Inf
                if torch.isnan(angle).any() or torch.isinf(angle).any():
                    print(f"  ⚠️  发现NaN或Inf!")
                if torch.isnan(x_req.grad).any() or torch.isinf(x_req.grad).any():
                    print(f"  ⚠️  x梯度有NaN或Inf!")
                if torch.isnan(y_req.grad).any() or torch.isinf(y_req.grad).any():
                    print(f"  ⚠️  y梯度有NaN或Inf!")

            except Exception as e:
                print(f"\n{version_name}:")
                print(f"  ❌ 错误: {e}")

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print("\n分析:")
    print("1. 如果严格clamp版本的梯度明显小于其他版本，说明梯度消失")
    print("2. 如果宽松clamp版本仍然梯度很小，说明问题不在clamp")
    print("3. 如果原始版本出现NaN，说明需要自定义Acos")
    print("4. cos_angle接近±1时，所有版本都可能有数值问题")

if __name__ == '__main__':
    test_versions()
