"""
Hyperbolic Utilities for Lorentz Model
基于HyCoCLIP的双曲空间操作工具
优化版本：应用数值稳定性和性能优化
包含切空间优化：在切空间做线性操作，在双曲空间做距离计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .hyperbolic_optimizations import (
    cosh, sinh, tanh, Arcosh, Artanh, Acos, AdaptiveEpsilon,
    lorentz_distance_with_clipping, exp_map_zero, log_map_zero
)


class LorentzManifold:
    """
    Lorentz双曲空间操作类
    使用(d+1)维Lorentz模型，其中第一个坐标是时间分量
    Lorentz内积: <x,y> = -x_0*y_0 + x_1*y_1 + ... + x_d*y_d
    """

    def __init__(self, eps=1e-7):
        self.eps = eps  # 保留基础 eps，但会在需要时使用自适应值
        self.min_norm = 1e-15
        self.max_dist = 50.0  # 最大距离截断，防止 NaN

    def minkowski_dot(self, x, y, keepdim=True):
        """
        Lorentz内积 (Minkowski内积)
        Args:
            x: shape [..., d+1]
            y: shape [..., d+1]
        Returns:
            <x,y> = -x_0*y_0 + sum(x_i*y_i for i>0)
        """
        res = torch.sum(x * y, dim=-1, keepdim=keepdim)
        res = res - 2 * x[..., 0:1] * y[..., 0:1]
        return res

    def lorentz_distance(self, x, y):
        """
        计算Lorentz距离（优化版本：带截断和自定义acosh）
        d(x,y) = arcosh(-<x,y>)
        """
        prod = self.minkowski_dot(x, y, keepdim=False)
        # 为数值稳定性，限制prod的范围
        eps = AdaptiveEpsilon.get_eps(x)
        prod = torch.clamp(prod, max=-1.0 - eps)

        # acosh requires input >= 1.0, use strict clamping
        acosh_input = -prod
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)

        # 使用自定义 acosh，带最大距离截断
        dist = Arcosh.apply(acosh_input)
        dist = torch.clamp(dist, max=self.max_dist)

        return dist

    def project_to_lorentz(self, x, k=1.0):
        """
        将欧氏空间向量投影到Lorentz双曲空间
        给定d维向量，返回(d+1)维双曲向量
        满足约束: -x_0^2 + ||x_{1:d}||^2 = -1/k

        Args:
            x: shape [..., d] 欧氏空间向量
            k: 曲率参数（正数）
        Returns:
            h: shape [..., d+1] 双曲空间向量
        """
        # 计算空间部分的范数
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        # 时间分量: x_0 = sqrt(1/k + ||x||^2)
        x_0 = torch.sqrt(1.0 / k + x_norm_sq + self.eps)
        # 拼接 [x_0, x_1, ..., x_d]
        h = torch.cat([x_0, x], dim=-1)
        return h

    def exp_map(self, x, v):
        """
        指数映射: exp_x(v) (优化版本：使用包装的双曲函数)
        将切空间向量v从点x映射到流形上

        Args:
            x: shape [..., d+1] 流形上的点
            v: shape [..., d+1] 切向量
        Returns:
            y: shape [..., d+1] 流形上的新点
        """
        # 计算v的Lorentz范数
        min_norm = AdaptiveEpsilon.get_min_norm(v)
        v_norm = torch.sqrt(torch.clamp(
            self.minkowski_dot(v, v, keepdim=True),
            min=min_norm
        ))

        # exp_x(v) = cosh(||v||)*x + sinh(||v||)*v/||v||
        # 使用包装的 cosh/sinh 防止溢出
        y = cosh(v_norm) * x + sinh(v_norm) * v / v_norm
        return y

    def log_map(self, x, y):
        """
        对数映射: log_x(y)
        将流形上的点y映射到x处的切空间

        Args:
            x: shape [..., d+1] 起点
            y: shape [..., d+1] 终点
        Returns:
            v: shape [..., d+1] 切向量
        """
        # 计算内积
        xy = self.minkowski_dot(x, y, keepdim=True)
        eps = AdaptiveEpsilon.get_eps(x)
        xy = torch.clamp(xy, max=-1.0 - eps)

        # 计算距离 - acosh requires input >= 1.0
        acosh_input = -xy
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)
        dist = Arcosh.apply(acosh_input)

        # log_x(y) = dist * (y + <x,y>*x) / ||y + <x,y>*x||
        # 使用包装的 sinh 防止溢出
        coef = dist / (sinh(dist) + eps)
        v = coef * (y + xy * x)
        return v

    def parallel_transport(self, x, y, v):
        """
        平行移动: 将x处的切向量v平行移动到y处

        Args:
            x: shape [..., d+1] 起点
            y: shape [..., d+1] 终点
            v: shape [..., d+1] x处的切向量
        Returns:
            v_transported: shape [..., d+1] y处的切向量
        """
        xy = self.minkowski_dot(x, y, keepdim=True)
        vy = self.minkowski_dot(v, y, keepdim=True)

        v_transported = v - vy / (xy + 1) * (x + y)
        return v_transported

    def mobius_add(self, x, y, k=1.0):
        """
        Möbius加法（双曲空间中的"加法"）
        Args:
            x, y: shape [..., d+1]
            k: 曲率
        """
        # 使用log和exp实现
        v = self.log_map(x, y)
        return self.exp_map(x, v)


class HyperbolicEmbedding(nn.Module):
    """
    双曲嵌入层
    将欧氏特征映射到Lorentz双曲空间
    """

    def __init__(self, euclidean_dim, hyperbolic_dim, manifold=None):
        """
        Args:
            euclidean_dim: 输入欧氏特征维度
            hyperbolic_dim: 输出双曲空间维度（实际输出为hyperbolic_dim+1）
            manifold: LorentzManifold实例
        """
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.manifold = manifold if manifold is not None else LorentzManifold()

        # 线性变换层
        self.linear = nn.Linear(euclidean_dim, hyperbolic_dim)

    def forward(self, x):
        """
        Args:
            x: shape [batch, euclidean_dim] 欧氏特征
        Returns:
            h: shape [batch, hyperbolic_dim+1] 双曲嵌入
        """
        # 线性变换
        x_transformed = self.linear(x)
        # 投影到双曲空间
        h = self.manifold.project_to_lorentz(x_transformed)
        return h


class EntailmentCone:
    """
    蕴含锥（Entailment Cone）
    在Lorentz模型中定义层次蕴含关系
    """

    def __init__(self, manifold=None, eps=1e-7):
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.eps = eps

    def aperture_angle(self, x):
        """
        计算蕴含锥的半孔径角度
        靠近原点的点具有更宽的孔径

        Args:
            x: shape [..., d+1] 双曲空间中的点
        Returns:
            theta: shape [..., 1] 半孔径角度
        """
        # 使用Lorentz范数计算距离原点的距离
        # 原点在Lorentz模型中是 [1, 0, ..., 0]
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0

        dist = self.manifold.lorentz_distance(x, origin)

        # 半孔径角度随距离增加而减小
        # theta = 2 * arcsin(1 / cosh(dist))
        # 使用包装的 cosh 防止溢出
        cosh_val = cosh(dist + self.eps)
        theta = 2 * torch.arcsin(torch.clamp(1.0 / cosh_val, max=1.0 - 1e-7))
        return theta

    def angle_between(self, x, y):
        """
        计算两点之间的角度（优化版本：使用稳定的计算方法）

        Args:
            x: shape [..., d+1] 锥顶点
            y: shape [..., d+1] 被测点
        Returns:
            angle: shape [...] 角度
        """
        # 对于Lorentz空间中的点，使用双曲距离计算角度更稳定
        # 避免直接使用acos，改用arcosh + 几何关系

        # 方法1：如果两点在双曲空间，使用双曲距离
        eps = AdaptiveEpsilon.get_eps(x)
        min_norm = AdaptiveEpsilon.get_min_norm(x)

        # 计算Minkowski内积（应该是负数）
        xy = self.manifold.minkowski_dot(x, y, keepdim=False)

        # 计算Minkowski范数
        xx = self.manifold.minkowski_dot(x, x, keepdim=False)
        yy = self.manifold.minkowski_dot(y, y, keepdim=False)

        # 确保范数是负数（Lorentz约束）
        xx = torch.clamp(xx, max=-eps)
        yy = torch.clamp(yy, max=-eps)

        x_norm = torch.sqrt(-xx + min_norm)
        y_norm = torch.sqrt(-yy + min_norm)

        # 计算余弦值
        cos_angle = xy / (x_norm * y_norm + eps)

        # 使用自定义Acos（带梯度截断和数值稳定性保护）
        angle = Acos.apply(cos_angle)

        # 确保angle是有限的
        angle = torch.clamp(angle, min=1e-7, max=3.14159 - 1e-7)

        return angle

    def entailment_score(self, parent, child):
        """
        计算蕴含分数
        如果child在parent的蕴含锥内，返回正值；否则返回负值

        Args:
            parent: shape [..., d+1] 父概念（更通用）
            child: shape [..., d+1] 子概念（更具体）
        Returns:
            score: shape [...] 蕴含分数
        """
        # 计算parent的半孔径
        theta_p = self.aperture_angle(parent).squeeze(-1)

        # 计算parent和child之间的角度
        angle = self.angle_between(parent, child)

        # score = theta_p - angle (如果为正，则child在锥内)
        score = theta_p - angle
        return score


class HyperbolicGraphConv(nn.Module):
    """
    双曲空间图卷积层（切空间优化版本）
    在Lorentz空间中进行消息传递

    优化策略：
    1. 输入映射到切空间（log_map_zero）
    2. 在切空间聚合邻居（欧式操作）
    3. 在切空间线性变换
    4. 映射回双曲空间（exp_map_zero）
    """

    def __init__(self, in_dim, out_dim, manifold=None, use_bias=True, c=1.0, use_tangent_opt=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.c = c
        self.use_tangent_opt = use_tangent_opt  # 是否使用切空间优化

        # 切空间中的线性变换
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.0)  # 修复：使用正常的gain
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: shape [N, in_dim+1] 双曲空间节点特征
            adj: shape [N, N] or sparse tensor 邻接矩阵
        Returns:
            out: shape [N, out_dim+1] 更新后的双曲特征
        """
        N = x.size(0)

        if self.use_tangent_opt:
            # ========== 切空间优化版本 ==========
            # 1. 映射到切空间（使用零点映射）
            x_tangent = log_map_zero(x[:, 1:], self.c)  # [N, in_dim]

            # 2. 在切空间中聚合邻居（欧式操作）
            if adj.is_sparse:
                # 稀疏矩阵操作（O(E)复杂度）
                adj_values = adj._values()
                adj_indices = adj._indices()

                # 计算度数
                deg = torch.zeros(N, 1, device=adj.device, dtype=adj_values.dtype)
                deg.index_add_(0, adj_indices[0], adj_values.unsqueeze(1))
                deg = deg + 1e-7

                # 归一化边权重
                adj_norm_values = adj_values / deg[adj_indices[0]].squeeze()
                adj_norm = torch.sparse_coo_tensor(
                    adj_indices, adj_norm_values, adj.size(),
                    dtype=adj.dtype, device=adj.device
                )

                # 稀疏聚合
                agg_tangent = torch.sparse.mm(adj_norm, x_tangent)  # [N, in_dim]
            else:
                # 密集矩阵操作
                deg = adj.sum(dim=1, keepdim=True) + 1e-7
                adj_norm = adj / deg
                agg_tangent = torch.matmul(adj_norm, x_tangent)  # [N, in_dim]

            # 3. 在切空间做线性变换（欧式操作）
            out_tangent = torch.matmul(agg_tangent, self.weight)  # [N, out_dim]
            if self.bias is not None:
                out_tangent = out_tangent + self.bias

            # 4. 映射回双曲空间（使用零点映射）
            out_spatial = exp_map_zero(out_tangent, self.c)  # [N, out_dim]

            # 5. 投影到Lorentz流形
            out = self.manifold.project_to_lorentz(out_spatial, k=self.c)

        else:
            # ========== 原始版本（直接提取空间部分）==========
            x_tangent = x[:, 1:]  # [N, in_dim]

            # 聚合邻居
            if adj.is_sparse:
                adj_values = adj._values()
                adj_indices = adj._indices()
                deg = torch.zeros(N, 1, device=adj.device, dtype=adj_values.dtype)
                deg.index_add_(0, adj_indices[0], adj_values.unsqueeze(1))
                deg = deg + 1e-7
                adj_norm_values = adj_values / deg[adj_indices[0]].squeeze()
                adj_norm = torch.sparse_coo_tensor(
                    adj_indices, adj_norm_values, adj.size(),
                    dtype=adj.dtype, device=adj.device
                )
                agg = torch.sparse.mm(adj_norm, x_tangent)
            else:
                deg = adj.sum(dim=1, keepdim=True) + 1e-7
                adj_norm = adj / deg
                agg = torch.matmul(adj_norm, x_tangent)

            # 线性变换
            out_tangent = torch.matmul(agg, self.weight)
            if self.bias is not None:
                out_tangent = out_tangent + self.bias

            # 投影回双曲空间
            out = self.manifold.project_to_lorentz(out_tangent)

        # 范数保持：防止图卷积导致范数坍缩
        # 计算输入和输出的空间范数
        input_spatial_norm = torch.norm(x[:, 1:], dim=1, keepdim=True).mean()
        output_spatial_norm = torch.norm(out[:, 1:], dim=1, keepdim=True).mean()

        # 优化：提高保留率从80%到90%，检测阈值从10%放宽到15%
        # 如果输出范数过小（<输入的15%），则放大到输入的90%
        if output_spatial_norm < input_spatial_norm * 0.15:
            target_norm = input_spatial_norm * 0.9  # 保留90% (优化前: 80%)
            current_norms = torch.norm(out[:, 1:], dim=1, keepdim=True)
            scale = target_norm / (current_norms + 1e-10)
            out_spatial_scaled = out[:, 1:] * scale
            # 重新投影到Lorentz流形
            out = self.manifold.project_to_lorentz(out_spatial_scaled, k=self.c)

        return out


def create_hyperbolic_features(euclidean_features, dim, manifold=None):
    """
    辅助函数：批量创建双曲特征

    Args:
        euclidean_features: shape [N, d] 欧氏特征
        dim: 双曲空间维度
        manifold: LorentzManifold实例
    Returns:
        hyperbolic_features: shape [N, dim+1]
    """
    if manifold is None:
        manifold = LorentzManifold()

    # 如果维度不匹配，先做线性变换
    if euclidean_features.shape[1] != dim:
        linear = nn.Linear(euclidean_features.shape[1], dim)
        with torch.no_grad():
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        euclidean_features = linear(euclidean_features)

    return manifold.project_to_lorentz(euclidean_features)
