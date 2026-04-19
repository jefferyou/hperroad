"""
HRNR with Hyperbolic Embeddings
基于HyCoCLIP思路改进的HRNR模型，使用Lorentz双曲空间
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.nn import Module
from torch.nn.parameter import Parameter
from logging import getLogger
from sklearn.metrics import roc_auc_score

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.upstream.road_representation.hyperbolic_utils import (
    LorentzManifold, HyperbolicEmbedding, EntailmentCone, HyperbolicGraphConv
)
import pdb


class HRNR_Hyperbolic(AbstractReprLearningModel):
    """
    基于双曲空间的层次化道路网络表示模型
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get("device", torch.device("cpu"))
        self.special_spmm = SpecialSpmm()
        self.dataloader = data_feature.get('dataloader')
        self._logger = getLogger()
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 128)
        self.embed_size = config.get('embed_size', 128)  # Road embedding size
        self.label_num = data_feature.get('label_class')

        self.struct_assign = data_feature.get("struct_assign").to(self.device)
        self.fnc_assign = data_feature.get("fnc_assign").to(self.device)
        adj = data_feature.get("adj_mx")
        self.adj = get_sparse_adj(adj, self.device)
        self.lane_feature = data_feature.get("lane_feature").to(self.device)
        self.type_feature = data_feature.get("type_feature").to(self.device)
        self.length_feature = data_feature.get("length_feature").to(self.device)
        self.node_feature = data_feature.get("node_feature").to(self.device)
        self.hidden_dims = config.get("hidden_dims")
        hparams = dict_to_object(config.config)

        hparams.lane_num = data_feature.get("lane_num")
        hparams.length_num = data_feature.get("length_num")
        hparams.type_num = data_feature.get("type_num")
        hparams.node_num = data_feature.get("num_nodes")

        # 双曲空间参数
        self.hyperbolic_dim = config.get('hyperbolic_dim', self.hidden_dims)
        self.lambda_ce = config.get('lambda_ce', 0.1)  # 蕴含损失权重
        self.lambda_cc = config.get('lambda_cc', 0.1)  # 对比损失权重
        self.temperature = config.get('temperature', 0.07)  # 对比学习温度

        # 双曲空间工具
        self.manifold = LorentzManifold()
        self.entailment_cone = EntailmentCone(self.manifold)

        edge = self.adj.indices()
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        struct_inter = self.special_spmm(edge, edge_e, torch.Size([self.adj.shape[0], self.adj.shape[1]]),
                                         self.struct_assign)
        struct_adj = torch.mm(self.struct_assign.t(), struct_inter)

        # 使用双曲图编码器
        self.graph_enc = HyperbolicGraphEncoderTL(
            hparams, self.struct_assign, self.fnc_assign, struct_adj,
            self.device, self.manifold, self.hyperbolic_dim
        )

        # 输出层从双曲空间映射回欧氏空间
        # 双曲空间维度是 hyperbolic_dim+1
        self.linear = torch.nn.Linear((self.hyperbolic_dim + 1) * 2, self.output_dim).to(self.device)

        self.node_emb, self.init_emb = None, None

        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.embed_size)

    def encode(self, x):
        """编码节点为双曲空间表示"""
        # 前向传播得到双曲空间嵌入
        self.node_emb = self.graph_enc(
            self.node_feature, self.type_feature, self.length_feature, self.lane_feature, self.adj
        )
        self.init_emb = self.graph_enc.init_feat

        # 拼接初始特征和编码后特征
        output_state = torch.cat((self.node_emb[x], self.init_emb[x]), -1)
        output_state = self.linear(output_state)

        return output_state

    def _save_final_embeddings(self):
        """在训练结束时保存最终的embeddings"""
        try:
            # 检查embeddings是否已经在训练过程中保存过
            if os.path.exists(self.road_embedding_path):
                file_size = os.path.getsize(self.road_embedding_path) / (1024 * 1024)  # MB
                self._logger.info(f"Embeddings already saved during training: {self.road_embedding_path}")
                self._logger.info(f"File size: {file_size:.2f} MB")
                return

            self._logger.info("Saving final embeddings...")

            # 检查segment_hyp_emb是否已初始化
            if self.graph_enc.segment_hyp_emb is None:
                self._logger.warning("segment_hyp_emb is None - embeddings were not generated during training")
                self._logger.warning("This usually means the training loop didn't complete any iterations")
                self._logger.warning("Embeddings will be generated during evaluation instead")
                return

            node_embedding = self.graph_enc.segment_hyp_emb.data.cpu().numpy()

            # 确保目录存在
            embedding_dir = os.path.dirname(self.road_embedding_path)
            os.makedirs(embedding_dir, exist_ok=True)

            # 保存
            np.save(self.road_embedding_path, node_embedding)
            self._logger.info(f"Final embeddings saved to {self.road_embedding_path}")
            self._logger.info(f"Embedding shape: {node_embedding.shape}")

            # 验证文件是否创建成功
            if os.path.exists(self.road_embedding_path):
                file_size = os.path.getsize(self.road_embedding_path) / (1024 * 1024)  # MB
                self._logger.info(f"✓ Verified: File exists ({file_size:.2f} MB)")
            else:
                self._logger.error(f"✗ ERROR: File was not created at {self.road_embedding_path}")
        except Exception as e:
            self._logger.error(f"Failed to save final embeddings: {e}")
            import traceback
            self._logger.error(traceback.format_exc())

    def compute_entailment_loss(self):
        """
        计算蕴含损失 (批量向量化版本)
        三类蕴含关系：
        1. Region 蕴含 Locality
        2. Locality 蕴含 Segment (采样)
        3. 拓扑连接的 Segment 互相蕴含 (采样)
        """
        segment_emb = self.graph_enc.segment_hyp_emb      # [N, d+1]
        locality_emb = self.graph_enc.locality_hyp_emb    # [N_loc, d+1]
        region_emb = self.graph_enc.region_hyp_emb        # [N_reg, d+1]

        # 1. Region -> Locality: 所有 fnc_assign[loc, reg] > 0 的对
        rl_pairs = (self.fnc_assign > 0).nonzero(as_tuple=False)  # [K, 2]: (loc, reg)
        if rl_pairs.numel() > 0:
            parents = region_emb[rl_pairs[:, 1]]
            children = locality_emb[rl_pairs[:, 0]]
            scores = self.entailment_cone.batched_entailment_score(parents, children)
            loss_rl = F.relu(-scores).mean()
        else:
            loss_rl = torch.zeros((), device=self.device)

        # 2. Locality -> Segment: 全部对中采样一批
        ls_pairs = (self.struct_assign > 0).nonzero(as_tuple=False)  # [K', 2]: (seg, loc)
        if ls_pairs.numel() > 0:
            max_samples = 10 * locality_emb.shape[0]
            n_pairs = ls_pairs.shape[0]
            n_sample = min(max_samples, n_pairs)
            sel = torch.randperm(n_pairs, device=self.device)[:n_sample]
            sampled = ls_pairs[sel]
            parents = locality_emb[sampled[:, 1]]
            children = segment_emb[sampled[:, 0]]
            scores = self.entailment_cone.batched_entailment_score(parents, children)
            loss_ls = F.relu(-scores).mean()
        else:
            loss_ls = torch.zeros((), device=self.device)

        # 3. Segment <-> Segment: 采样邻接边
        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        if num_edges > 0:
            sample_edges = min(1000, num_edges)
            sel = torch.randperm(num_edges, device=self.device)[:sample_edges]
            i_idx = edge_indices[0, sel]
            j_idx = edge_indices[1, sel]
            scores_ij = self.entailment_cone.batched_entailment_score(
                segment_emb[i_idx], segment_emb[j_idx]
            )
            scores_ji = self.entailment_cone.batched_entailment_score(
                segment_emb[j_idx], segment_emb[i_idx]
            )
            loss_ss = (F.relu(-scores_ij).mean() + F.relu(-scores_ji).mean()) * 0.5
        else:
            loss_ss = torch.zeros((), device=self.device)

        return (loss_rl + loss_ls + loss_ss) / 3.0

    def compute_contrastive_loss(self):
        """
        层次对比学习损失 (批量向量化 InfoNCE)
        1. Segment 层: 相邻为正对, 共享随机负样本池
        2. 跨层: Locality 与其所属 Segment 为正对, 共享负样本池
        """
        segment_emb = self.graph_enc.segment_hyp_emb         # [N, d+1]
        locality_emb = self.graph_enc.locality_hyp_emb       # [N_loc, d+1]

        neg_pool_size = 64
        losses = []

        # 1. Segment 层 InfoNCE
        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        if num_edges > 0:
            B = min(500, num_edges)
            sel = torch.randperm(num_edges, device=self.device)[:B]
            anchors = edge_indices[0, sel]
            positives = edge_indices[1, sel]

            anchor_emb = segment_emb[anchors]                # [B, d+1]
            pos_emb = segment_emb[positives]                 # [B, d+1]

            neg_idx = torch.randint(
                0, segment_emb.shape[0], (neg_pool_size,), device=self.device
            )
            neg_emb = segment_emb[neg_idx]                   # [K, d+1]

            pos_dist = self.manifold.lorentz_distance(anchor_emb, pos_emb)     # [B]
            neg_dist = self.manifold.pairwise_lorentz_distance(anchor_emb, neg_emb)  # [B, K]

            logits = torch.cat(
                [-pos_dist.unsqueeze(1), -neg_dist], dim=1
            ) / self.temperature                             # [B, K+1]
            labels = torch.zeros(B, dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits, labels))

        # 2. 跨层 InfoNCE (Locality anchor <-> Segment positive)
        ls_pairs = (self.struct_assign > 0).nonzero(as_tuple=False)  # [K', 2]: (seg, loc)
        if ls_pairs.numel() > 0:
            M = min(500, ls_pairs.shape[0])
            sel = torch.randperm(ls_pairs.shape[0], device=self.device)[:M]
            sampled = ls_pairs[sel]
            pos_seg_idx = sampled[:, 0]
            anchor_loc_idx = sampled[:, 1]

            anchor_emb = locality_emb[anchor_loc_idx]        # [M, d+1]
            pos_emb = segment_emb[pos_seg_idx]               # [M, d+1]

            neg_idx = torch.randint(
                0, segment_emb.shape[0], (neg_pool_size,), device=self.device
            )
            neg_emb = segment_emb[neg_idx]                   # [K, d+1]

            pos_dist = self.manifold.lorentz_distance(anchor_emb, pos_emb)     # [M]
            neg_dist = self.manifold.pairwise_lorentz_distance(anchor_emb, neg_emb)  # [M, K]

            logits = torch.cat(
                [-pos_dist.unsqueeze(1), -neg_dist], dim=1
            ) / self.temperature
            labels = torch.zeros(M, dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits, labels))

        if len(losses) == 0:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def run(self, train_dataloader, eval_dataloader):
        """训练循环"""
        self._logger.info("Starting training with Hyperbolic Embeddings...")
        hparams = dict_to_object(self.config.config)
        ce_criterion = torch.nn.CrossEntropyLoss()
        max_f1 = 0
        max_auc = 0
        count = 0
        model_optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lp_learning_rate)
        eval_dataloader_iter = iter(eval_dataloader)
        patience = 50

        for i in range(hparams.max_epoch):
            self._logger.info("epoch " + str(i) + ", processed " + str(count))
            for step, (train_set, train_label) in enumerate(train_dataloader):
                model_optimizer.zero_grad()
                train_set = train_set.clone().detach().to(self.device)
                train_label = train_label.clone().detach().to(self.device)

                # 分类损失
                pred = self.encode(train_set)
                loss_struct = ce_criterion(pred, train_label)

                # 蕴含损失
                loss_ce = self.compute_entailment_loss()

                # 对比损失
                loss_cc = self.compute_contrastive_loss()

                # 总损失
                loss = loss_struct + self.lambda_ce * loss_ce + self.lambda_cc * loss_cc

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.lp_clip)
                model_optimizer.step()

                if count % 20 == 0:
                    self._logger.info(f"=== DEBUG: Starting evaluation at count={count} ===")
                    eval_data = get_next(eval_dataloader_iter)
                    if eval_data is None:
                        eval_dataloader_iter = iter(eval_dataloader)
                        eval_data = get_next(eval_dataloader_iter)
                    test_set, test_label = eval_data
                    self._logger.info(f"=== DEBUG: Calling test_label_pred ===")
                    precision, recall, f1, auc = self.test_label_pred(test_set, test_label, self.device)
                    self._logger.info(f"=== DEBUG: Got auc={auc}, max_auc={max_auc} ===")

                    if auc > max_auc:
                        self._logger.info(f"=== DEBUG: Entering save block (auc {auc} > max_auc {max_auc}) ===")
                        max_auc = auc
                        # 保存segment层的双曲嵌入
                        try:
                            self._logger.info(f"=== DEBUG: Accessing segment_hyp_emb ===")
                            self._logger.info(f"=== DEBUG: segment_hyp_emb type: {type(self.graph_enc.segment_hyp_emb)} ===")
                            node_embedding = self.graph_enc.segment_hyp_emb.data.cpu().numpy()
                            # 确保evaluate_cache目录存在
                            embedding_dir = os.path.dirname(self.road_embedding_path)
                            self._logger.info(f"=== DEBUG: Creating directory: {embedding_dir} ===")
                            os.makedirs(embedding_dir, exist_ok=True)
                            self._logger.info(f"=== DEBUG: Saving embeddings to: {self.road_embedding_path} ===")
                            self._logger.info(f"=== DEBUG: Embedding shape: {node_embedding.shape} ===")
                            np.save(self.road_embedding_path, node_embedding)
                            self._logger.info(f"=== DEBUG: Embeddings saved successfully ===")
                            # Verify file was created
                            if os.path.exists(self.road_embedding_path):
                                self._logger.info(f"Verified: File exists at {self.road_embedding_path}")
                            else:
                                self._logger.error(f"ERROR: File was not created at {self.road_embedding_path}")
                        except Exception as e:
                            self._logger.error(f"=== DEBUG: EXCEPTION while saving embeddings: {e} ===")
                            import traceback
                            self._logger.error(f"=== DEBUG: Traceback: {traceback.format_exc()} ===")

                    if f1 > max_f1:
                        max_f1 = f1

                    if auc >= max_auc and f1 >= max_f1:
                        patience = 50
                    else:
                        patience -= 1
                        if patience == 0:
                            self._logger.info("early stop")
                            self._logger.info("max_auc: " + str(max_auc))
                            self._logger.info("max_f1: " + str(max_f1))
                            self._logger.info("step " + str(count))
                            self._logger.info(f"loss: {loss.item()}, struct: {loss_struct.item()}, "
                                            f"ce: {loss_ce.item()}, cc: {loss_cc.item()}")
                            # 保存最终的embeddings
                            self._save_final_embeddings()
                            return

                    self._logger.info("max_auc: " + str(max_auc))
                    self._logger.info("max_f1: " + str(max_f1))
                    self._logger.info("step " + str(count))
                    self._logger.info(f"loss: {loss.item()}, struct: {loss_struct.item()}, "
                                    f"ce: {loss_ce.item()}, cc: {loss_cc.item()}")
                count += 1

        # 训练正常结束，保存最终的embeddings
        self._logger.info("Training completed normally")
        self._save_final_embeddings()

    def test_label_pred(self, test_set, test_label, device):
        """评估函数"""
        right = 0
        sum_num = 0
        test_set = test_set.clone().detach().to(device)
        pred = self.encode(test_set)
        pred_prob = F.softmax(pred, -1)
        pred_scores = pred_prob[:, 1]
        auc = roc_auc_score(np.array(test_label), np.array(pred_scores.tolist()))
        self._logger.info("auc: " + str(auc))

        pred_loc = torch.argmax(pred, 1).tolist()
        right_pos = 0
        right_neg = 0
        wrong_pos = 0
        wrong_neg = 0
        for item1, item2 in zip(pred_loc, test_label):
            if item1 == item2:
                right += 1
                if item2 == 1:
                    right_pos += 1
                else:
                    right_neg += 1
            else:
                if item2 == 1:
                    wrong_pos += 1
                else:
                    wrong_neg += 1
            sum_num += 1
        recall_sum = right_pos + wrong_pos
        precision_sum = wrong_neg + right_pos
        if recall_sum == 0:
            recall_sum += 1
        if precision_sum == 0:
            precision_sum += 1
        recall = float(right_pos) / recall_sum
        precision = float(right_pos) / precision_sum
        if recall == 0 or precision == 0:
            self._logger.info("p/r/f:0/0/0")
            return 0.0, 0.0, 0.0, auc  # Return actual AUC even if precision/recall is 0
        f1 = 2 * recall * precision / (precision + recall)
        self._logger.info("label prediction @acc @p/r/f: " + str(float(right) / sum_num) + " " + str(precision) +
                          " " + str(recall) + " " + str(f1))
        return precision, recall, f1, auc


class HyperbolicGraphEncoderTL(Module):
    """
    双曲空间图编码器
    支持三层次结构：Segment -> Locality -> Region
    """
    def __init__(self, hparams, struct_assign, fnc_assign, struct_adj, device, manifold, hyperbolic_dim):
        super(HyperbolicGraphEncoderTL, self).__init__()
        self.hparams = hparams
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj
        self.manifold = manifold
        self.hyperbolic_dim = hyperbolic_dim

        # 原始特征嵌入（欧氏空间）
        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.device)
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.device)
        self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.device)

        # 计算欧氏特征总维度
        euclidean_dim = hparams.lane_dims + hparams.type_dims + hparams.length_dims + hparams.node_dims

        # 双曲嵌入层：将欧氏特征映射到双曲空间
        self.hyp_embedding = HyperbolicEmbedding(
            euclidean_dim, hyperbolic_dim, manifold
        ).to(self.device)

        # 三层双曲图编码器
        self.tl_layer_1 = HyperbolicGraphEncoderTLCore(
            hparams, self.struct_assign, self.fnc_assign, self.device, self.manifold, hyperbolic_dim
        )
        self.tl_layer_2 = HyperbolicGraphEncoderTLCore(
            hparams, self.struct_assign, self.fnc_assign, self.device, self.manifold, hyperbolic_dim
        )

        self.init_feat = None
        self.segment_hyp_emb = None
        self.locality_hyp_emb = None
        self.region_hyp_emb = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        """
        前向传播
        返回双曲空间中的segment表示
        """
        # 1. 获取欧氏特征
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)

        # 2. 映射到双曲空间
        hyp_feat = self.hyp_embedding(raw_feat)  # [N, hyperbolic_dim+1]
        self.init_feat = hyp_feat

        # 3. 双曲图卷积层
        hyp_feat = self.tl_layer_1(self.struct_adj, hyp_feat, adj)
        hyp_feat = self.tl_layer_2(self.struct_adj, hyp_feat, adj)

        # 4. 保存各层次嵌入用于计算蕴含损失和对比损失
        self.segment_hyp_emb = hyp_feat
        # 通过分配矩阵计算locality和region的嵌入
        self.locality_hyp_emb = self._aggregate_hyperbolic(hyp_feat, self.struct_assign.t())
        self.region_hyp_emb = self._aggregate_hyperbolic(self.locality_hyp_emb, self.fnc_assign.t())

        return hyp_feat

    def _aggregate_hyperbolic(self, embeddings, assignment_matrix):
        """
        在双曲空间中聚合嵌入（原点切空间加权质心）

        Args:
            embeddings: [N, d+1] 双曲嵌入
            assignment_matrix: [M, N] 分配矩阵
        Returns:
            aggregated: [M, d+1] 聚合后的双曲嵌入
        """
        return self.manifold.weighted_centroid_at_origin(embeddings, assignment_matrix)


class HyperbolicGraphEncoderTLCore(Module):
    """
    双曲图编码器核心层
    在双曲空间中进行层次化消息传递
    """
    def __init__(self, hparams, struct_assign, fnc_assign, device, manifold, hyperbolic_dim):
        super(HyperbolicGraphEncoderTLCore, self).__init__()
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.manifold = manifold
        self.hyperbolic_dim = hyperbolic_dim

        # 双曲图卷积层
        self.fnc_gcn = HyperbolicGraphConv(
            in_dim=hyperbolic_dim,
            out_dim=hyperbolic_dim,
            manifold=manifold
        ).to(self.device)

        self.struct_gcn = HyperbolicGraphConv(
            in_dim=hyperbolic_dim,
            out_dim=hyperbolic_dim,
            manifold=manifold
        ).to(self.device)

        self.node_gcn = HyperbolicGraphConv(
            in_dim=hyperbolic_dim,
            out_dim=hyperbolic_dim,
            manifold=manifold
        ).to(self.device)

        # 门控机制（在欧氏空间中）
        self.l_c = torch.nn.Linear((hyperbolic_dim + 1) * 2, 1).to(self.device)
        self.l_s = torch.nn.Linear((hyperbolic_dim + 1) * 2, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_adj, hyp_feat, raw_adj):
        """
        双曲空间中的层次化消息传递
        F2F -> F2C -> C2C -> C2N -> N2N
        """
        # 归一化分配矩阵
        struct_assign_norm = self.struct_assign / (F.relu(torch.sum(self.struct_assign, 0) - 1.0) + 1.0)
        fnc_assign_norm = self.fnc_assign / (F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0)

        # Forward: 自底向上聚合
        # Segment -> Locality (struct)
        struct_emb = self._aggregate_to_cluster(hyp_feat, struct_assign_norm)

        # Locality -> Region (fnc)
        fnc_emb = self._aggregate_to_cluster(struct_emb, fnc_assign_norm)

        # Backward: 自顶向下传播
        # F2F: Region内部消息传递
        fnc_adj = self._compute_hyperbolic_affinity(fnc_emb)
        fnc_adj = fnc_adj + torch.eye(fnc_adj.shape[0]).to(self.device) * 1.0
        fnc_emb = self.fnc_gcn(fnc_emb, fnc_adj)

        # F2C: Region -> Locality
        fnc_message = self._distribute_from_cluster(fnc_emb, self.fnc_assign, fnc_assign_norm)
        r_f = self.sigmoid(self.l_c(torch.cat((struct_emb, fnc_message), 1)))  # [N_loc, 1]
        struct_emb = self._hyperbolic_update(struct_emb, fnc_message, weight=r_f)

        # C2C: Locality内部消息传递
        struct_adj_processed = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.device) * 10000.0) + \
                              torch.eye(struct_adj.shape[1]).to(self.device) * 1.0
        struct_emb = self.struct_gcn(struct_emb, struct_adj_processed)

        # C2N: Locality -> Segment
        struct_message = self._distribute_from_cluster(struct_emb, self.struct_assign, struct_assign_norm)
        r_s = self.sigmoid(self.l_s(torch.cat((hyp_feat, struct_message), 1)))  # [N, 1]
        hyp_feat = self._hyperbolic_update(hyp_feat, struct_message, weight=r_s)

        # N2N: Segment内部消息传递
        hyp_feat = self.node_gcn(hyp_feat, raw_adj)

        return hyp_feat

    def _aggregate_to_cluster(self, embeddings, assignment_matrix):
        """
        聚合到聚类中心（双曲空间，原点切空间加权质心）
        assignment_matrix: [N, M] (node -> cluster)
        returns: [M, d+1]
        """
        return self.manifold.weighted_centroid_at_origin(
            embeddings, assignment_matrix.t()
        )

    def _distribute_from_cluster(self, cluster_emb, raw_assign, norm_assign=None):
        """
        从聚类分发到节点（双曲空间，原点切空间加权质心）
        cluster_emb: [M, d+1]
        raw_assign:  [N, M] (node -> cluster)
        returns:     [N, d+1]
        """
        return self.manifold.weighted_centroid_at_origin(cluster_emb, raw_assign)

    def _hyperbolic_update(self, x, message, weight=0.5):
        """
        双曲空间中的加权更新
        使用指数映射和对数映射
        """
        # 计算从x到message的方向
        tangent_vec = self.manifold.log_map(x, message)
        # 缩放
        tangent_vec = tangent_vec * weight
        # 沿该方向移动
        updated = self.manifold.exp_map(x, tangent_vec)
        return updated

    def _compute_hyperbolic_affinity(self, embeddings):
        """计算双曲空间中的亲和度矩阵（批量 Minkowski）"""
        dist = self.manifold.pairwise_lorentz_distance(embeddings, embeddings)
        return torch.exp(-dist)


# ========== 辅助函数和类 ==========

def get_sparse_adj(adj, device):
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)

    adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
                               dtype=torch.long, device=device).t()
    adj_values = torch.tensor(adj.data, dtype=torch.float, device=device)
    adj_shape = adj.shape
    adj = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape).to(device)
    return adj.coalesce()


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape, device=b.device)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


def get_next(it):
    res = None
    try:
        res = next(it)
    except StopIteration:
        pass
    return res
