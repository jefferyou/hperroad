"""
Embedding Wrapper for Downstream Tasks
提供一个包装类，让下游任务可以直接使用预计算的embeddings而不需要重新运行模型
"""

import torch
import numpy as np


class EmbeddingWrapper:
    """
    包装预计算的embeddings，提供encode接口供下游任务使用
    这样可以避免在下游任务中重新运行模型的前向传播，大幅提升速度
    """

    def __init__(self, embeddings, device='cpu'):
        """
        Args:
            embeddings: numpy数组或torch tensor, shape=[N, embed_dim]
            device: 设备（'cpu' 或 'cuda'）
        """
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings).float()
        else:
            self.embeddings = embeddings.float()

        self.device = device
        self.embeddings = self.embeddings.to(device)

    def encode(self, indices):
        """
        根据索引返回对应的embeddings

        Args:
            indices: torch.Tensor，shape可以是任意形状，如[batch_size, seq_len]

        Returns:
            embeddings: torch.Tensor，shape=[*indices.shape, embed_dim]
        """
        # 保存原始形状
        original_shape = indices.shape

        # 展平索引
        flat_indices = indices.reshape(-1).long()

        # 查找embeddings
        flat_embeddings = self.embeddings[flat_indices]

        # 恢复原始形状
        output_shape = list(original_shape) + [self.embeddings.shape[1]]
        embeddings = flat_embeddings.view(*output_shape)

        return embeddings

    def to(self, device):
        """支持设备转换"""
        self.device = device
        self.embeddings = self.embeddings.to(device)
        return self

    def __getitem__(self, idx):
        """
        支持直接索引访问（向后兼容）
        返回numpy数组以兼容使用numpy操作的下游任务
        """
        result = self.embeddings[idx]
        # 如果是torch tensor，转换为numpy
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        return result

    @property
    def shape(self):
        """返回embeddings的形状"""
        return self.embeddings.shape
