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
from veccity.upstream.road_representation.hyperbolic_optimizations import (
    Arcosh, AdaptiveEpsilon
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
        self.label_num = data_feature.get('label_class')

        self.struct_assign = data_feature.get("struct_assign")
        self.fnc_assign = data_feature.get("fnc_assign")
        adj = data_feature.get("adj_mx")
        self.adj = get_sparse_adj(adj, self.device)
        self.lane_feature = data_feature.get("lane_feature")
        self.type_feature = data_feature.get("type_feature")
        self.length_feature = data_feature.get("length_feature")
        self.node_feature = data_feature.get("node_feature")
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
            format(self.exp_id, self.model, self.dataset, self.output_dim)

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

    def compute_entailment_loss(self):
        """
        计算蕴含损失（优化版本：批量计算减少循环）
        定义三类蕴含关系：
        1. Region蕴含Locality
        2. Locality蕴含Segment
        3. 拓扑连接的Segment互相蕴含
        """
        loss = 0.0
        count = 0

        # 获取三个层次的嵌入
        segment_emb = self.graph_enc.segment_hyp_emb  # [N, d+1]
        locality_emb = self.graph_enc.locality_hyp_emb  # [N_locality, d+1]
        region_emb = self.graph_enc.region_hyp_emb  # [N_region, d+1]

        # 1. Region蕴含Locality（批量计算）
        region_to_locality = self.fnc_assign.t()  # [N_region, N_locality]
        # 找到所有有连接的region-locality对
        region_idx, locality_idx = (region_to_locality > 0).nonzero(as_tuple=True)
        if len(region_idx) > 0:
            # 批量计算蕴含分数
            scores = self.entailment_cone.entailment_score(
                region_emb[region_idx], locality_emb[locality_idx]
            )
            loss += F.relu(-scores).mean() * len(region_idx)
            count += len(region_idx)

        # 2. Locality蕴含Segment（批量采样）
        locality_to_segment = self.struct_assign.t()  # [N_locality, N]
        # 采样locality
        num_localities_sample = min(50, locality_emb.shape[0])
        sampled_localities = torch.randperm(locality_emb.shape[0], device=self.device)[:num_localities_sample]

        for loc_idx in sampled_localities:
            segments_idx = (locality_to_segment[loc_idx] > 0).nonzero(as_tuple=True)[0]
            if len(segments_idx) > 0:
                # 采样segment
                sample_size = min(10, len(segments_idx))
                sampled_seg_idx = segments_idx[torch.randperm(len(segments_idx), device=self.device)[:sample_size]]

                # 批量计算
                scores = self.entailment_cone.entailment_score(
                    locality_emb[loc_idx:loc_idx+1].expand(len(sampled_seg_idx), -1),
                    segment_emb[sampled_seg_idx]
                )
                loss += F.relu(-scores).mean() * len(sampled_seg_idx)
                count += len(sampled_seg_idx)

        # 3. 拓扑连接的Segment互相蕴含（批量计算）
        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        sample_edges = min(1000, num_edges)
        sampled_edge_idx = torch.randperm(num_edges, device=self.device)[:sample_edges]

        # 批量提取边的端点
        edge_i = edge_indices[0, sampled_edge_idx]
        edge_j = edge_indices[1, sampled_edge_idx]

        # 批量计算双向蕴含
        scores_ij = self.entailment_cone.entailment_score(
            segment_emb[edge_i], segment_emb[edge_j]
        )
        scores_ji = self.entailment_cone.entailment_score(
            segment_emb[edge_j], segment_emb[edge_i]
        )
        loss += (F.relu(-scores_ij).mean() + F.relu(-scores_ji).mean()) * sample_edges
        count += 2 * sample_edges

        return loss / (count + 1e-7)

    def compute_contrastive_loss(self):
        """
        计算层次对比学习损失（优化版本：批量计算）
        1. Segment层：相邻路段为正对，非邻路段为负对
        2. 跨层：Segment与其所属Locality为正对
        """
        loss = 0.0

        segment_emb = self.graph_enc.segment_hyp_emb
        locality_emb = self.graph_enc.locality_hyp_emb

        # 1. Segment层对比（批量计算）
        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        sample_edges = min(500, num_edges)
        sampled_edge_idx = torch.randperm(num_edges, device=self.device)[:sample_edges]

        # 批量提取anchor和positive
        anchors = edge_indices[0, sampled_edge_idx]  # [sample_edges]
        positives = edge_indices[1, sampled_edge_idx]  # [sample_edges]

        # 批量计算正样本相似度
        anchor_emb = segment_emb[anchors]  # [sample_edges, d+1]
        pos_emb = segment_emb[positives]  # [sample_edges, d+1]

        # 批量计算距离
        pos_dist = self._batch_lorentz_distance(anchor_emb, pos_emb)  # [sample_edges]
        pos_sim = -pos_dist / self.temperature  # [sample_edges]

        # 批量采样负样本
        num_neg = 10
        neg_samples = torch.randint(0, segment_emb.shape[0], (sample_edges, num_neg), device=self.device)  # [sample_edges, 10]
        neg_emb = segment_emb[neg_samples]  # [sample_edges, 10, d+1]

        # 批量计算负样本相似度
        anchor_emb_expanded = anchor_emb.unsqueeze(1)  # [sample_edges, 1, d+1]
        neg_dist = self._batch_lorentz_distance_pairwise(anchor_emb_expanded, neg_emb)  # [sample_edges, 10]
        neg_sim = -neg_dist / self.temperature  # [sample_edges, 10]

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [sample_edges, 11]
        labels = torch.zeros(sample_edges, dtype=torch.long, device=self.device)  # [sample_edges]
        loss += F.cross_entropy(logits, labels)

        # 2. 跨层对比（批量计算）
        locality_to_segment = self.struct_assign.t()
        num_localities = min(50, locality_emb.shape[0])
        sampled_localities = torch.randperm(locality_emb.shape[0], device=self.device)[:num_localities]

        # 为每个locality选择一个正样本segment
        pos_segments = []
        valid_localities = []
        for loc_idx in sampled_localities:
            segments_idx = (locality_to_segment[loc_idx] > 0).nonzero(as_tuple=True)[0]
            if len(segments_idx) > 0:
                pos_seg = segments_idx[torch.randint(0, len(segments_idx), (1,), device=self.device)].item()
                pos_segments.append(pos_seg)
                valid_localities.append(loc_idx.item())

        if len(valid_localities) > 0:
            valid_localities = torch.tensor(valid_localities, device=self.device)
            pos_segments = torch.tensor(pos_segments, device=self.device)

            # 批量计算正样本相似度
            loc_emb = locality_emb[valid_localities]  # [num_valid, d+1]
            pos_seg_emb = segment_emb[pos_segments]  # [num_valid, d+1]
            pos_dist = self._batch_lorentz_distance(loc_emb, pos_seg_emb)
            pos_sim = -pos_dist / self.temperature  # [num_valid]

            # 批量采样负样本
            neg_samples = torch.randint(0, segment_emb.shape[0], (len(valid_localities), num_neg), device=self.device)
            neg_emb = segment_emb[neg_samples]  # [num_valid, 10, d+1]

            # 批量计算负样本相似度
            loc_emb_expanded = loc_emb.unsqueeze(1)  # [num_valid, 1, d+1]
            neg_dist = self._batch_lorentz_distance_pairwise(loc_emb_expanded, neg_emb)
            neg_sim = -neg_dist / self.temperature  # [num_valid, 10]

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [num_valid, 11]
            labels = torch.zeros(len(valid_localities), dtype=torch.long, device=self.device)
            loss += F.cross_entropy(logits, labels)

        return loss / (sample_edges + len(valid_localities) + 1e-7)

    def _batch_lorentz_distance(self, x, y):
        """
        批量计算Lorentz距离（对应位置）
        Args:
            x: [N, d+1]
            y: [N, d+1]
        Returns:
            dist: [N]
        """
        # Minkowski内积
        prod = torch.sum(x * y, dim=-1) - 2 * x[:, 0] * y[:, 0]
        # 数值稳定性（使用自适应epsilon）
        eps = AdaptiveEpsilon.get_eps(x)
        prod = torch.clamp(prod, max=-1.0 - eps)

        # acosh requires input >= 1.0
        acosh_input = -prod
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)

        # 使用自定义acosh + 距离截断
        dist = Arcosh.apply(acosh_input)
        dist = torch.clamp(dist, max=self.manifold.max_dist)
        return dist

    def _batch_lorentz_distance_pairwise(self, x, y):
        """
        批量计算Lorentz距离（成对）
        Args:
            x: [N, 1, d+1]
            y: [N, M, d+1]
        Returns:
            dist: [N, M]
        """
        # Minkowski内积
        prod = torch.sum(x * y, dim=-1) - 2 * x[:, :, 0] * y[:, :, 0]
        # 数值稳定性（使用自适应epsilon）
        eps = AdaptiveEpsilon.get_eps(x)
        prod = torch.clamp(prod, max=-1.0 - eps)

        # acosh requires input >= 1.0
        acosh_input = -prod
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)

        # 使用自定义acosh + 距离截断
        dist = Arcosh.apply(acosh_input)
        dist = torch.clamp(dist, max=self.manifold.max_dist)
        return dist

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

                loss.backward(retain_graph=True)
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
                            return

                    self._logger.info("max_auc: " + str(max_auc))
                    self._logger.info("max_f1: " + str(max_f1))
                    self._logger.info("step " + str(count))
                    self._logger.info(f"loss: {loss.item()}, struct: {loss_struct.item()}, "
                                    f"ce: {loss_ce.item()}, cc: {loss_cc.item()}")
                count += 1

    def test_label_pred(self, test_set, test_label, device):
        """评估函数"""
        right = 0
        sum_num = 0
        test_set = test_set.clone().detach().to(device)
        pred = self.encode(test_set)
        pred_prob = F.softmax(pred, -1)
        pred_scores = pred_prob[:, 1]
        auc = roc_auc_score(np.array(test_label), np.array(pred_scores.cpu().tolist()))
        self._logger.info("auc: " + str(auc))

        pred_loc = torch.argmax(pred, 1).cpu().tolist()
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

        # 三层双曲图编码器（与原HRNR保持一致，虽然原HRNR只使用了前2层）
        self.tl_layer_1 = HyperbolicGraphEncoderTLCore(
            hparams, self.struct_assign, self.fnc_assign, self.device, self.manifold, hyperbolic_dim
        )
        self.tl_layer_2 = HyperbolicGraphEncoderTLCore(
            hparams, self.struct_assign, self.fnc_assign, self.device, self.manifold, hyperbolic_dim
        )
        self.tl_layer_3 = HyperbolicGraphEncoderTLCore(
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
        在双曲空间中聚合嵌入
        优化版本：使用批量矩阵操作替代循环

        Args:
            embeddings: [N, d+1] 双曲嵌入
            assignment_matrix: [M, N] 分配矩阵
        Returns:
            aggregated: [M, d+1] 聚合后的双曲嵌入
        """
        # 归一化分配矩阵的每一行（确保每个聚类的权重和为1）
        row_sums = assignment_matrix.sum(dim=1, keepdim=True) + 1e-7
        normalized_assignment = assignment_matrix / row_sums  # [M, N]

        # 方法1：简化版 - 在切空间中聚合
        # 映射到切空间（使用原点作为参考点）
        origin = torch.zeros_like(embeddings[0])
        origin[0] = 1.0

        # 批量映射到切空间
        tangent_embeddings = self.manifold.log_map(
            origin.unsqueeze(0).expand(embeddings.shape[0], -1),
            embeddings
        )  # [N, d+1]

        # 使用矩阵乘法进行加权聚合
        aggregated_tangent = torch.matmul(normalized_assignment, tangent_embeddings)  # [M, d+1]

        # 批量映射回双曲空间
        aggregated = self.manifold.exp_map(
            origin.unsqueeze(0).expand(aggregated_tangent.shape[0], -1),
            aggregated_tangent
        )  # [M, d+1]

        return aggregated


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
        r_f = self.sigmoid(self.l_c(torch.cat((struct_emb, fnc_message), 1)))
        struct_emb = self._hyperbolic_update(struct_emb, fnc_message, weight=0.15)

        # C2C: Locality内部消息传递
        struct_adj_processed = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.device) * 10000.0) + \
                              torch.eye(struct_adj.shape[1]).to(self.device) * 1.0
        struct_emb = self.struct_gcn(struct_emb, struct_adj_processed)

        # C2N: Locality -> Segment
        struct_message = self._distribute_from_cluster(struct_emb, self.struct_assign, struct_assign_norm)
        r_s = self.sigmoid(self.l_s(torch.cat((hyp_feat, struct_message), 1)))
        hyp_feat = self._hyperbolic_update(hyp_feat, struct_message, weight=0.5)

        # N2N: Segment内部消息传递
        hyp_feat = self.node_gcn(hyp_feat, raw_adj)

        return hyp_feat

    def _aggregate_to_cluster(self, embeddings, assignment_matrix):
        """聚合到聚类中心（双曲空间）"""
        # 简化版：直接使用矩阵乘法聚合空间部分
        spatial_part = embeddings[:, 1:]  # [N, d]
        cluster_spatial = torch.mm(assignment_matrix.t(), spatial_part)  # [M, d]
        # 投影回双曲空间
        cluster_hyp = self.manifold.project_to_lorentz(cluster_spatial)
        return cluster_hyp

    def _distribute_from_cluster(self, cluster_emb, raw_assign, norm_assign):
        """从聚类分发到节点（双曲空间）"""
        # 简化版：使用矩阵乘法分发
        cluster_spatial = cluster_emb[:, 1:]
        node_spatial = torch.mm(raw_assign, cluster_spatial)
        # 归一化
        node_spatial = torch.div(node_spatial,
                                (F.relu(torch.sum(norm_assign, 1) - 1.0) + 1.0).unsqueeze(1))
        # 投影回双曲空间
        node_hyp = self.manifold.project_to_lorentz(node_spatial)
        return node_hyp

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
        """
        计算双曲空间中的亲和度矩阵
        优化版本：向量化计算所有对之间的距离
        """
        N = embeddings.shape[0]

        # 向量化计算所有对的Minkowski内积
        # embeddings: [N, d+1]
        # 时间分量 (第0维)
        t = embeddings[:, 0:1]  # [N, 1]
        # 空间分量 (第1:维)
        x = embeddings[:, 1:]   # [N, d]

        # Minkowski内积矩阵: <x_i, x_j> = -t_i*t_j + x_i·x_j
        # 空间部分内积: [N, d] @ [d, N] = [N, N]
        spatial_prod = torch.matmul(x, x.t())  # [N, N]
        # 时间部分内积: [N, 1] @ [1, N] = [N, N]
        temporal_prod = torch.matmul(t, t.t())  # [N, N]
        # Minkowski内积
        minkowski_prod = spatial_prod - temporal_prod  # [N, N]

        # 计算距离: d(x,y) = arcosh(-<x,y>)
        # 数值稳定性：限制prod的范围（使用自适应epsilon）
        eps = AdaptiveEpsilon.get_eps(embeddings)
        minkowski_prod = torch.clamp(minkowski_prod, max=-1.0 - eps)

        # acosh requires input >= 1.0
        acosh_input = -minkowski_prod
        acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)

        # 使用自定义acosh + 距离截断
        dist_matrix = Arcosh.apply(acosh_input)  # [N, N]
        dist_matrix = torch.clamp(dist_matrix, max=self.manifold.max_dist)

        # 计算亲和度: exp(-距离)
        affinity = torch.exp(-dist_matrix)

        return affinity


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
        a = torch.sparse_coo_tensor(indices, values, shape)
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
