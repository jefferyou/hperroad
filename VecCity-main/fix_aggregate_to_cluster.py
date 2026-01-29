#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 _aggregate_to_cluster 方法
正确的双曲空间聚合实现
"""

import torch

# 正确的双曲空间聚合方法
def _aggregate_to_cluster_FIXED(self, embeddings, assignment_matrix):
    """
    在双曲空间中聚合到聚类中心
    使用 log_map -> 切空间平均 -> exp_map 的正确流程

    Args:
        embeddings: [N, d+1] 双曲空间中的点
        assignment_matrix: [N, M] 分配矩阵，assignment_matrix[i,j] 表示点i对聚类j的权重

    Returns:
        cluster_emb: [M, d+1] 聚类中心的双曲嵌入
    """
    N, d_plus_1 = embeddings.shape
    M = assignment_matrix.shape[1]
    device = embeddings.device

    cluster_embeddings = []

    for cluster_idx in range(M):
        # 获取属于这个聚类的节点及其权重
        weights = assignment_matrix[:, cluster_idx]  # [N]

        # 找到权重>0的节点
        mask = weights > 1e-10
        if not mask.any():
            # 如果没有节点属于这个聚类，使用原点
            origin = torch.zeros(d_plus_1, device=device)
            origin[0] = 1.0
            cluster_embeddings.append(origin)
            continue

        node_embs = embeddings[mask]  # [K, d+1]
        node_weights = weights[mask]  # [K]
        node_weights = node_weights / node_weights.sum()  # 归一化权重

        # 方法1: Fréchet mean (双曲空间的加权中心)
        # 初始化为第一个点
        mean_emb = node_embs[0].clone()

        # 迭代优化 (5-10次通常足够)
        for _ in range(5):
            # 将所有点映射到mean_emb的切空间
            tangent_vecs = []
            for i, emb in enumerate(node_embs):
                v = self.manifold.log_map(mean_emb.unsqueeze(0), emb.unsqueeze(0)).squeeze(0)
                tangent_vecs.append(v * node_weights[i])

            # 在切空间中加权平均
            avg_tangent = torch.stack(tangent_vecs).sum(dim=0)

            # 映射回流形
            mean_emb = self.manifold.exp_map(
                mean_emb.unsqueeze(0),
                avg_tangent.unsqueeze(0)
            ).squeeze(0)

        cluster_embeddings.append(mean_emb)

    return torch.stack(cluster_embeddings)


# 简化版（更快但稍微不精确）
def _aggregate_to_cluster_FAST(self, embeddings, assignment_matrix):
    """
    快速版本：使用切空间平均
    在原点的切空间中做平均，然后exp_map回流形

    这个版本速度更快，但假设所有点都不太远离原点
    """
    N, d_plus_1 = embeddings.shape
    M = assignment_matrix.shape[1]
    device = embeddings.device

    # Lorentz原点
    origin = torch.zeros(1, d_plus_1, device=device)
    origin[0, 0] = 1.0

    cluster_embeddings = []

    for cluster_idx in range(M):
        weights = assignment_matrix[:, cluster_idx]  # [N]

        # 归一化权重
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            cluster_embeddings.append(origin.squeeze(0))
            continue

        weights_normalized = weights / weight_sum

        # 将所有点映射到原点的切空间
        tangent_vecs = self.manifold.log_map(
            origin.expand(N, -1),
            embeddings
        )  # [N, d+1]

        # 在切空间中做加权平均
        avg_tangent = (tangent_vecs * weights_normalized.unsqueeze(1)).sum(dim=0)  # [d+1]

        # 映射回流形
        cluster_emb = self.manifold.exp_map(
            origin,
            avg_tangent.unsqueeze(0)
        ).squeeze(0)  # [d+1]

        cluster_embeddings.append(cluster_emb)

    return torch.stack(cluster_embeddings)


# 最简化版（用于快速验证）
def _aggregate_to_cluster_SIMPLE(self, embeddings, assignment_matrix):
    """
    最简化版本：直接在双曲空间缩放
    避免错误的欧氏平均，但保持结构简单

    关键：在聚合前保持双曲结构，而不是丢弃时间分量
    """
    # 使用完整的双曲向量做聚合
    # 然后重新投影到双曲空间
    cluster_hyp = torch.mm(assignment_matrix.t(), embeddings)  # [M, d+1]

    # 归一化（除以每个聚类的节点数）
    cluster_counts = assignment_matrix.sum(dim=0, keepdim=True).t()  # [M, 1]
    cluster_hyp = cluster_hyp / (cluster_counts + 1e-10)

    # 取出空间部分并重新投影
    # 但这次先做缩放，防止范数太小
    cluster_spatial = cluster_hyp[:, 1:]

    # 方法A: 直接缩放到合理的范数
    current_norms = torch.norm(cluster_spatial, dim=1, keepdim=True)
    target_norm = 5.0  # 目标范数，可以是超参数
    scale = target_norm / (current_norms + 1e-10)
    cluster_spatial_scaled = cluster_spatial * scale

    # 投影回双曲空间
    cluster_hyp_final = self.manifold.project_to_lorentz(cluster_spatial_scaled)

    return cluster_hyp_final


if __name__ == '__main__':
    print("""
修复方案说明：
=============

问题：当前的 _aggregate_to_cluster 在欧氏空间中聚合，导致空间分量坍缩

推荐实现（按优先级）：

1. _aggregate_to_cluster_FAST (推荐)
   - 在原点切空间中平均，速度快
   - 适合大多数情况
   - 数学上正确

2. _aggregate_to_cluster_SIMPLE (快速验证)
   - 最简单的修复
   - 通过缩放防止范数过小
   - 可快速验证是否能解决问题

3. _aggregate_to_cluster_FIXED (最精确)
   - Fréchet mean，数学上最严格
   - 计算成本较高
   - 适合对精度要求高的场景

使用方法：
在 HyperbolicGraphEncoderTLCore 类中替换 _aggregate_to_cluster 方法

位置：veccity/upstream/road_representation/HRNR_Hyperbolic.py:642
    """)
