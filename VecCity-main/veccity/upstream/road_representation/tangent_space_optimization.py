"""
Tangent Space Optimization for Hyperbolic Neural Networks
切空间优化：在切空间做线性操作，在双曲空间做距离计算

核心思想：
1. 输入先映射到切空间（Log Map）
2. 在切空间做线性变换（欧式空间，简单高效）
3. 映射回双曲空间（Exp Map）
4. 在双曲空间计算距离（保持几何特性）
5. 聚合结果再映射回双曲空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperbolic_utils import LorentzManifold
from .hyperbolic_optimizations import exp_map_zero, log_map_zero, cosh, sinh


class TangentSpaceLinear(nn.Module):
    """
    在切空间中进行线性变换

    流程：
    1. 双曲空间 -> 切空间 (log_map_zero)
    2. 线性变换 (欧式空间)
    3. 切空间 -> 双曲空间 (exp_map_zero)
    """

    def __init__(self, in_dim, out_dim, manifold=None, use_bias=True, c=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.c = c

        # 线性变换权重（在切空间中）
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.01)  # 小初始化避免越界
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_hyp):
        """
        Args:
            x_hyp: [N, in_dim+1] 双曲空间中的点
        Returns:
            out_hyp: [N, out_dim+1] 双曲空间中的输出
        """
        # 1. 映射到切空间（在原点）
        x_tangent = log_map_zero(x_hyp[:, 1:], self.c)  # [N, in_dim]

        # 2. 在切空间做线性变换（欧式操作）
        out_tangent = torch.matmul(x_tangent, self.weight)  # [N, out_dim]
        if self.bias is not None:
            out_tangent = out_tangent + self.bias

        # 3. 映射回双曲空间
        out_spatial = exp_map_zero(out_tangent, self.c)  # [N, out_dim]

        # 4. 投影到Lorentz流形（添加时间分量）
        out_hyp = self.manifold.project_to_lorentz(out_spatial, k=self.c)

        return out_hyp


class TangentSpaceGraphConv(nn.Module):
    """
    在切空间中进行图卷积

    流程：
    1. 双曲空间 -> 切空间
    2. 在切空间聚合邻居（欧式聚合）
    3. 在切空间做线性变换
    4. 切空间 -> 双曲空间
    5. 在双曲空间计算注意力（可选）
    """

    def __init__(self, in_dim, out_dim, manifold=None, use_attention=False, c=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.use_attention = use_attention
        self.c = c

        # 线性变换（在切空间）
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        if use_attention:
            # 注意力参数（在切空间）
            self.att_weight = nn.Parameter(torch.Tensor(in_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.01)
        nn.init.zeros_(self.bias)
        if self.use_attention:
            nn.init.xavier_uniform_(self.att_weight)

    def forward(self, x_hyp, adj):
        """
        Args:
            x_hyp: [N, in_dim+1] 双曲空间节点特征
            adj: [N, N] 或 sparse tensor 邻接矩阵
        Returns:
            out_hyp: [N, out_dim+1] 更新后的双曲特征
        """
        N = x_hyp.size(0)

        # 1. 映射到切空间
        x_tangent = log_map_zero(x_hyp[:, 1:], self.c)  # [N, in_dim]

        # 2. 在切空间聚合邻居
        if adj.is_sparse:
            # 稀疏矩阵乘法（只计算实际存在的边）
            adj_values = adj._values()
            adj_indices = adj._indices()

            # 归一化：每个节点的度数
            deg = torch.zeros(N, 1, device=adj.device, dtype=adj_values.dtype)
            deg.index_add_(0, adj_indices[0], adj_values.unsqueeze(1))
            deg = deg + 1e-7

            # 归一化边权重
            adj_norm_values = adj_values / deg[adj_indices[0]].squeeze()
            adj_norm = torch.sparse_coo_tensor(
                adj_indices, adj_norm_values, adj.size(),
                dtype=adj.dtype, device=adj.device
            )

            # 稀疏聚合（O(E)复杂度）
            agg_tangent = torch.sparse.mm(adj_norm, x_tangent)  # [N, in_dim]
        else:
            # 密集矩阵
            deg = adj.sum(dim=1, keepdim=True) + 1e-7
            adj_norm = adj / deg
            agg_tangent = torch.matmul(adj_norm, x_tangent)  # [N, in_dim]

        # 3. 在切空间做线性变换（欧式操作）
        out_tangent = torch.matmul(agg_tangent, self.weight) + self.bias  # [N, out_dim]

        # 4. 映射回双曲空间
        out_spatial = exp_map_zero(out_tangent, self.c)  # [N, out_dim]
        out_hyp = self.manifold.project_to_lorentz(out_spatial, k=self.c)

        return out_hyp


class TangentSpaceAggregation(nn.Module):
    """
    在切空间中进行聚合操作（用于pooling）

    流程：
    1. 双曲空间 -> 切空间
    2. 在切空间加权聚合（欧式操作）
    3. 切空间 -> 双曲空间
    """

    def __init__(self, manifold=None, c=1.0):
        super().__init__()
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.c = c

    def forward(self, x_hyp, assignment_matrix):
        """
        Args:
            x_hyp: [N, d+1] 双曲空间节点特征
            assignment_matrix: [N, M] 分配矩阵（soft assignment）
        Returns:
            cluster_hyp: [M, d+1] 聚类中心（双曲空间）
        """
        # 1. 映射到切空间
        x_tangent = log_map_zero(x_hyp[:, 1:], self.c)  # [N, d]

        # 2. 在切空间聚合（加权平均）
        # 归一化分配矩阵
        assign_norm = assignment_matrix / (assignment_matrix.sum(dim=0, keepdim=True) + 1e-7)

        # 加权聚合
        cluster_tangent = torch.mm(assign_norm.t(), x_tangent)  # [M, d]

        # 3. 映射回双曲空间
        cluster_spatial = exp_map_zero(cluster_tangent, self.c)  # [M, d]
        cluster_hyp = self.manifold.project_to_lorentz(cluster_spatial, k=self.c)

        return cluster_hyp


class TangentSpaceGating(nn.Module):
    """
    在切空间中进行门控操作

    流程：
    1. 双曲空间 -> 切空间
    2. 在切空间拼接和计算门控值（欧式操作）
    3. 在切空间应用门控
    4. 切空间 -> 双曲空间
    """

    def __init__(self, dim, manifold=None, c=1.0):
        super().__init__()
        self.dim = dim
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.c = c

        # 门控参数（在切空间）
        self.gate_linear = nn.Linear(dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_hyp, y_hyp):
        """
        Args:
            x_hyp: [N, d+1] 双曲空间特征1
            y_hyp: [N, d+1] 双曲空间特征2
        Returns:
            out_hyp: [N, d+1] 门控后的双曲特征
        """
        # 1. 映射到切空间
        x_tangent = log_map_zero(x_hyp[:, 1:], self.c)  # [N, d]
        y_tangent = log_map_zero(y_hyp[:, 1:], self.c)  # [N, d]

        # 2. 在切空间计算门控值（欧式操作）
        concat = torch.cat([x_tangent, y_tangent], dim=1)  # [N, 2d]
        gate = self.sigmoid(self.gate_linear(concat))  # [N, 1]

        # 3. 在切空间应用门控（加权组合）
        out_tangent = gate * x_tangent + (1 - gate) * y_tangent  # [N, d]

        # 4. 映射回双曲空间
        out_spatial = exp_map_zero(out_tangent, self.c)  # [N, d]
        out_hyp = self.manifold.project_to_lorentz(out_spatial, k=self.c)

        return out_hyp, gate


class HyperbolicDistanceAttention(nn.Module):
    """
    在双曲空间计算距离注意力（稀疏版本）

    流程：
    1. 输入特征在切空间中变换（如果需要）
    2. 在双曲空间计算距离（只对边）
    3. 距离转换为注意力分数
    4. 在切空间进行加权聚合
    5. 映射回双曲空间
    """

    def __init__(self, dim, manifold=None, temperature=0.07, c=1.0):
        super().__init__()
        self.dim = dim
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.temperature = temperature
        self.c = c

    def forward(self, query_hyp, key_hyp, value_hyp, edge_index):
        """
        Args:
            query_hyp: [N, d+1] 查询（双曲空间）
            key_hyp: [N, d+1] 键（双曲空间）
            value_hyp: [N, d+1] 值（双曲空间）
            edge_index: [2, E] 边索引 (source, target)
        Returns:
            output_hyp: [N, d+1] 输出（双曲空间）
        """
        N = query_hyp.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        E = src_idx.size(0)

        # 1. 提取边对应的节点特征
        query_edges = query_hyp[src_idx]  # [E, d+1]
        key_edges = key_hyp[dst_idx]      # [E, d+1]
        value_edges = value_hyp[dst_idx]  # [E, d+1]

        # 2. 在双曲空间计算距离（只对边，O(E)复杂度）
        dist = self.manifold.lorentz_distance(query_edges, key_edges)  # [E]

        # 3. 距离转换为注意力分数（负距离 = 相似度）
        scores = -dist / self.temperature  # [E]
        scores = torch.exp(scores - scores.max())  # 数值稳定性

        # 4. 归一化（每个目标节点）
        sum_scores = torch.zeros(N, device=scores.device)
        sum_scores.index_add_(0, dst_idx, scores)
        scores_norm = scores / (sum_scores[dst_idx] + 1e-15)  # [E]

        # 5. 在切空间进行加权聚合
        value_tangent = log_map_zero(value_edges[:, 1:], self.c)  # [E, d]

        # 加权聚合
        output_tangent = torch.zeros(N, self.dim, device=value_tangent.device)
        weighted_values = value_tangent * scores_norm.unsqueeze(1)
        output_tangent.index_add_(0, dst_idx, weighted_values)  # [N, d]

        # 6. 映射回双曲空间
        output_spatial = exp_map_zero(output_tangent, self.c)
        output_hyp = self.manifold.project_to_lorentz(output_spatial, k=self.c)

        return output_hyp


# ============================================================================
# 工具函数
# ============================================================================

def hyperbolic_to_tangent_batch(x_hyp, manifold, c=1.0):
    """批量映射到切空间"""
    return log_map_zero(x_hyp[:, 1:], c)


def tangent_to_hyperbolic_batch(x_tangent, manifold, c=1.0):
    """批量映射回双曲空间"""
    spatial = exp_map_zero(x_tangent, c)
    return manifold.project_to_lorentz(spatial, k=c)


def hyperbolic_midpoint(x_hyp, y_hyp, manifold, c=1.0):
    """
    计算两个双曲点的中点（在切空间计算）

    Args:
        x_hyp, y_hyp: [N, d+1] 双曲空间点
        manifold: LorentzManifold实例
        c: 曲率
    Returns:
        mid_hyp: [N, d+1] 中点（双曲空间）
    """
    # 映射到切空间
    x_tangent = log_map_zero(x_hyp[:, 1:], c)
    y_tangent = log_map_zero(y_hyp[:, 1:], c)

    # 在切空间计算中点（欧式平均）
    mid_tangent = (x_tangent + y_tangent) / 2.0

    # 映射回双曲空间
    mid_spatial = exp_map_zero(mid_tangent, c)
    mid_hyp = manifold.project_to_lorentz(mid_spatial, k=c)

    return mid_hyp
