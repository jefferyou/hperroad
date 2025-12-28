"""
Hyperbolic Utilities for Lorentz Model
基于HyCoCLIP的双曲空间操作工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LorentzManifold:
    """
    Lorentz双曲空间操作类
    使用(d+1)维Lorentz模型，其中第一个坐标是时间分量
    Lorentz内积: <x,y> = -x_0*y_0 + x_1*y_1 + ... + x_d*y_d
    """

    def __init__(self, eps=1e-7):
        self.eps = eps
        self.min_norm = 1e-15

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
        计算Lorentz距离
        d(x,y) = arcosh(-<x,y>)
        """
        prod = self.minkowski_dot(x, y, keepdim=False)
        # 为数值稳定性，限制prod的范围
        prod = torch.clamp(prod, max=-1.0 - self.eps)
        dist = torch.acosh(-prod + self.eps)
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
        指数映射: exp_x(v)
        将切空间向量v从点x映射到流形上

        Args:
            x: shape [..., d+1] 流形上的点
            v: shape [..., d+1] 切向量
        Returns:
            y: shape [..., d+1] 流形上的新点
        """
        # 计算v的Lorentz范数
        v_norm = torch.sqrt(torch.clamp(
            self.minkowski_dot(v, v, keepdim=True),
            min=self.min_norm
        ))

        # exp_x(v) = cosh(||v||)*x + sinh(||v||)*v/||v||
        y = torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm
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
        xy = torch.clamp(xy, max=-1.0 - self.eps)

        # 计算距离
        dist = torch.acosh(-xy + self.eps)

        # log_x(y) = dist * (y + <x,y>*x) / ||y + <x,y>*x||
        coef = dist / torch.sinh(dist + self.eps)
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
        theta = 2 * torch.arcsin(1.0 / torch.cosh(dist + self.eps))
        return theta

    def angle_between(self, x, y):
        """
        计算两点之间的角度

        Args:
            x: shape [..., d+1] 锥顶点
            y: shape [..., d+1] 被测点
        Returns:
            angle: shape [...] 角度
        """
        # 计算角度 arccos(<x,y> / (||x|| * ||y||))
        xy = self.manifold.minkowski_dot(x, y, keepdim=False)
        x_norm = torch.sqrt(-self.manifold.minkowski_dot(x, x, keepdim=False))
        y_norm = torch.sqrt(-self.manifold.minkowski_dot(y, y, keepdim=False))

        cos_angle = xy / (x_norm * y_norm + self.eps)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos_angle)
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
    双曲空间图卷积层
    在Lorentz空间中进行消息传递
    """

    def __init__(self, in_dim, out_dim, manifold=None, use_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold if manifold is not None else LorentzManifold()

        # 切空间中的线性变换
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: shape [N, in_dim+1] 双曲空间节点特征
            adj: shape [N, N] 邻接矩阵
        Returns:
            out: shape [N, out_dim+1] 更新后的双曲特征
        """
        # 1. 映射到切空间（在原点处）
        origin = torch.zeros_like(x[:1])
        origin[0, 0] = 1.0

        # 提取空间部分进行变换
        x_tangent = x[:, 1:]  # shape [N, in_dim]

        # 2. 在切空间中聚合邻居
        # 归一化邻接矩阵
        # Handle sparse adjacency matrix
        if adj.is_sparse:
            adj_dense = adj.to_dense()
            deg = adj_dense.sum(dim=1, keepdim=True) + 1e-7
            adj_norm = adj_dense / deg
        else:
            deg = adj.sum(dim=1, keepdim=True) + 1e-7
            adj_norm = adj / deg

        # 聚合
        agg = torch.matmul(adj_norm, x_tangent)  # [N, in_dim]

        # 3. 线性变换
        out_tangent = torch.matmul(agg, self.weight)  # [N, out_dim]
        if self.bias is not None:
            out_tangent = out_tangent + self.bias

        # 4. 投影回双曲空间
        out = self.manifold.project_to_lorentz(out_tangent)

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
