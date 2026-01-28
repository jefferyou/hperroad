"""
Hyperbolic Computation Optimizations
双曲计算优化工具 - 基于最佳实践的数值稳定性和性能优化
"""

import torch
import torch.nn.functional as F


# ============================================================================
# 1. 数值稳定性优化：包装双曲函数
# ============================================================================

def cosh(x, clamp=15):
    """
    数值稳定的 cosh 函数
    Args:
        x: 输入张量
        clamp: 截断阈值，防止溢出
    """
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    """
    数值稳定的 sinh 函数
    """
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    """
    数值稳定的 tanh 函数
    """
    return x.clamp(-clamp, clamp).tanh()


# ============================================================================
# 2. 自定义反双曲函数（提升精度和稳定性）
# ============================================================================

class Artanh(torch.autograd.Function):
    """
    自定义 artanh 函数，提升数值稳定性
    使用 double 精度计算后转回原精度
    """
    @staticmethod
    def forward(ctx, x):
        # 严格边界保护
        x = x.clamp(-1.0 + 1e-15, 1.0 - 1e-15)
        ctx.save_for_backward(x)

        # 提升精度计算
        z = x.double()
        result = (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 使用稳定的导数计算
        return grad_output / (1 - x * x + 1e-15)


class Arcosh(torch.autograd.Function):
    """
    自定义 acosh 函数，提升数值稳定性
    """
    @staticmethod
    def forward(ctx, x):
        # 严格边界保护：acosh 定义域为 [1, ∞)
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)

        # 提升精度计算
        z = x.double()
        result = torch.log(z + torch.sqrt(z * z - 1.0))
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 使用稳定的导数计算
        return grad_output / torch.sqrt(x * x - 1.0 + 1e-15)


class Acos(torch.autograd.Function):
    """
    自定义 acos 函数，提升数值稳定性
    acos的导数在边界附近趋于无穷，需要特别小心
    """
    @staticmethod
    def forward(ctx, x):
        # 严格边界保护：acos 定义域为 [-1, 1]
        # 使用更大的margin避免导数爆炸
        x = x.clamp(min=-0.9999, max=0.9999)
        ctx.save_for_backward(x)

        # 提升精度计算
        z = x.double()
        result = torch.acos(z)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 导数：-1 / sqrt(1 - x^2)
        # 在边界附近截断导数避免爆炸
        denominator = torch.sqrt(1.0 - x * x + 1e-8)
        # 截断梯度幅度
        grad = -grad_output / denominator
        grad = torch.clamp(grad, min=-10.0, max=10.0)  # 限制梯度幅度
        return grad


# ============================================================================
# 3. 自适应精度管理
# ============================================================================

class AdaptiveEpsilon:
    """
    根据数据类型使用不同的 epsilon 值
    """
    EPS = {
        torch.float16: 1e-4,
        torch.float32: 1e-7,
        torch.float64: 1e-15
    }

    MIN_NORM = {
        torch.float16: 1e-4,
        torch.float32: 1e-15,
        torch.float64: 1e-20
    }

    @classmethod
    def get_eps(cls, tensor):
        """获取适合张量类型的 epsilon"""
        return cls.EPS.get(tensor.dtype, 1e-7)

    @classmethod
    def get_min_norm(cls, tensor):
        """获取适合张量类型的最小范数"""
        return cls.MIN_NORM.get(tensor.dtype, 1e-15)


# ============================================================================
# 4. 零点映射优化（避免一般点映射的复杂计算）
# ============================================================================

def exp_map_zero(v, c=1.0):
    """
    零点的指数映射（从切空间到双曲空间）
    exp_0(v) = tanh(√c * ||v||) / (√c * ||v||) * v

    Args:
        v: shape [..., d] 切空间向量
        c: 曲率参数
    Returns:
        x: shape [..., d] 双曲空间点
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=v.dtype, device=v.device))
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-15)

    # 使用稳定的 tanh
    tanh_arg = sqrt_c * v_norm
    tanh_arg = torch.clamp(tanh_arg, -15.0, 15.0)

    coef = torch.tanh(tanh_arg) / (sqrt_c * v_norm)
    return coef * v


def log_map_zero(x, c=1.0):
    """
    零点的对数映射（从双曲空间到切空间）
    log_0(x) = arctanh(√c * ||x||) / (√c * ||x||) * x

    Args:
        x: shape [..., d] 双曲空间点
        c: 曲率参数
    Returns:
        v: shape [..., d] 切空间向量
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    x_norm = torch.clamp(x_norm, min=1e-15)

    # 使用自定义 artanh
    artanh_arg = sqrt_c * x_norm
    artanh_arg = torch.clamp(artanh_arg, -1.0 + 1e-7, 1.0 - 1e-7)
    artanh_val = Artanh.apply(artanh_arg)

    coef = artanh_val / (sqrt_c * x_norm)
    return coef * x


# ============================================================================
# 5. 距离计算优化（带截断）
# ============================================================================

def poincare_distance_with_clipping(x, y, c=1.0, max_dist=50.0):
    """
    Poincaré 球距离计算，带最大距离截断
    防止 Fermi-Dirac decoder 中的 NaN

    Args:
        x, y: 双曲空间点
        c: 曲率
        max_dist: 最大距离阈值
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))

    # 计算 ||x - y||^2
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)

    # 计算 (1 - c||x||^2) 和 (1 - c||y||^2)
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)

    eps = AdaptiveEpsilon.get_eps(x)
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = torch.clamp(denom, min=eps)

    # 距离公式
    dist_arg = 1 + 2 * c * diff_norm_sq / denom
    dist_arg = torch.clamp(dist_arg, min=1.0 + eps)

    # 使用自定义 acosh
    dist = Arcosh.apply(dist_arg) / sqrt_c

    # 关键：截断最大距离
    dist = torch.clamp(dist, max=max_dist)

    return dist


def lorentz_distance_with_clipping(x, y, eps=1e-7, max_dist=50.0):
    """
    Lorentz 距离计算，带最大距离截断
    d(x,y) = acosh(-<x,y>)

    Args:
        x, y: shape [..., d+1] Lorentz 空间点
        eps: 数值稳定性阈值
        max_dist: 最大距离阈值
    """
    # Minkowski 内积
    prod = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
    prod = torch.clamp(prod, max=-1.0 - eps)

    # acosh 输入
    acosh_input = -prod
    acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)

    # 使用自定义 acosh
    dist = Arcosh.apply(acosh_input)

    # 关键：截断最大距离
    dist = torch.clamp(dist, max=max_dist)

    return dist


# ============================================================================
# 6. Minkowski 范数优化
# ============================================================================

def minkowski_norm(u, keepdim=True):
    """
    Minkowski 范数计算
    ||u|| = sqrt(-<u,u>)

    Args:
        u: shape [..., d+1] Lorentz 空间向量
    """
    # Minkowski 内积
    dot = torch.sum(u * u, dim=-1, keepdim=keepdim)
    dot = dot - 2 * u[..., 0:1] ** 2 if keepdim else dot - 2 * u[..., 0] ** 2

    # 取负数并开方
    eps = AdaptiveEpsilon.get_eps(u)
    return torch.sqrt(torch.clamp(-dot, min=eps))


# ============================================================================
# 7. 稀疏注意力优化（仅计算图中存在的边）
# ============================================================================

def sparse_hyperbolic_attention(query, key, value, adj_indices, temperature=0.07):
    """
    稀疏双曲注意力机制
    只计算图中实际存在的边，复杂度 O(E) 而不是 O(N^2)

    Args:
        query: [N, d] 查询向量
        key: [N, d] 键向量
        value: [N, d] 值向量
        adj_indices: [2, E] 边的索引 (source, target)
        temperature: 温度参数
    Returns:
        output: [N, d] 输出特征
    """
    src_idx = adj_indices[0]  # [E]
    dst_idx = adj_indices[1]  # [E]

    # 只提取需要的节点特征（稀疏化）
    query_src = query[src_idx]  # [E, d]
    key_dst = key[dst_idx]      # [E, d]
    value_dst = value[dst_idx]  # [E, d]

    # 计算注意力分数（只对边计算）
    # 使用负距离作为相似度
    scores = -lorentz_distance_with_clipping(query_src, key_dst)  # [E]
    scores = scores / temperature

    # 对每个目标节点的所有边进行 softmax
    # 使用 scatter_softmax 实现
    scores = torch.exp(scores - torch.max(scores))

    # 归一化
    sum_scores = torch.zeros(query.size(0), device=query.device)
    sum_scores.scatter_add_(0, dst_idx, scores)
    scores = scores / (sum_scores[dst_idx] + 1e-15)

    # 加权聚合
    output = torch.zeros_like(query)
    weighted_values = value_dst * scores.unsqueeze(-1)
    output.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(weighted_values), weighted_values)

    return output


# ============================================================================
# 8. 批处理优化工具
# ============================================================================

def batch_hyperbolic_operation(func, x, batch_size=1000):
    """
    批处理双曲操作，避免一次性计算大矩阵

    Args:
        func: 操作函数
        x: 输入张量
        batch_size: 批大小
    """
    n = x.size(0)
    results = []

    for i in range(0, n, batch_size):
        batch = x[i:i+batch_size]
        result = func(batch)
        results.append(result)

    return torch.cat(results, dim=0)


# ============================================================================
# 9. 投影优化（使用 torch.where 避免不必要计算）
# ============================================================================

def project_to_poincare_ball(x, c=1.0):
    """
    投影到 Poincaré 球
    只对超出边界的点进行投影
    """
    eps = AdaptiveEpsilon.get_eps(x)
    min_norm = AdaptiveEpsilon.get_min_norm(x)

    norm = torch.clamp(x.norm(dim=-1, keepdim=True, p=2), min=min_norm)
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))
    maxnorm = (1 - eps) / sqrt_c

    # 只对 norm > maxnorm 的点进行投影
    cond = norm > maxnorm
    projected = x / norm * maxnorm

    # 使用 torch.where 避免不必要的计算
    return torch.where(cond, projected, x)


# ============================================================================
# 10. 梯度检查工具
# ============================================================================

def check_hyperbolic_gradients(tensor, name=""):
    """
    检查双曲计算中的梯度健康状况
    """
    if tensor.grad is not None:
        grad = tensor.grad
        has_nan = torch.isnan(grad).any()
        has_inf = torch.isinf(grad).any()
        grad_norm = grad.norm()

        if has_nan or has_inf:
            print(f"⚠️  Gradient issue in {name}:")
            print(f"   - NaN: {has_nan}")
            print(f"   - Inf: {has_inf}")
            print(f"   - Norm: {grad_norm}")
            return False
    return True
