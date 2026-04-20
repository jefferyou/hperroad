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
    LorentzManifold, HyperbolicEmbedding, EntailmentCone, HyperbolicGraphConv,
    adaptive_temperature,
)
import pdb


class HyperbolicMomentumQueue(nn.Module):
    """
    FIFO Lorentz-space memory bank for large-scale contrastive negatives.
    Keys are detached current-encoder outputs (memory-bank variant — no separate
    key encoder, avoids doubling graph_enc cost). Used for:
      (a) expanding InfoNCE negative pool (64 → thousands)
      (b) hard negative mining via Lorentz-distance topk
    """

    def __init__(self, dim, size, device):
        super().__init__()
        self.size = int(size)
        self.dim = int(dim)
        # Initialize at Lorentz origin [1, 0, ..., 0]
        init = torch.zeros(self.size, self.dim, device=device)
        init[:, 0] = 1.0
        self.register_buffer('queue', init)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer('queue_filled', torch.zeros(1, dtype=torch.long, device=device))

    @torch.no_grad()
    def enqueue(self, keys):
        """Append `keys` ([B, d+1] Lorentz, detached) to the ring buffer."""
        if keys.numel() == 0:
            return
        keys = keys.detach()
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if B >= self.size:
            self.queue.copy_(keys[-self.size:])
            self.queue_ptr[0] = 0
            self.queue_filled[0] = self.size
            return
        end = ptr + B
        if end <= self.size:
            self.queue[ptr:end] = keys
        else:
            first = self.size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[:B - first] = keys[first:]
        self.queue_ptr[0] = (ptr + B) % self.size
        self.queue_filled[0] = min(self.size, int(self.queue_filled.item()) + B)

    def get(self):
        """Returns filled portion of queue, or None if empty."""
        filled = int(self.queue_filled.item())
        if filled == 0:
            return None
        return self.queue[:filled]


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
        # node_feature stores geo_uid, which may not be contiguous — size the
        # embedding table by max id to avoid CUDA index-out-of-bounds.
        num_nodes = data_feature.get("num_nodes")
        max_node_id = int(self.node_feature.max().item()) + 1
        hparams.node_num = max(num_nodes, max_node_id)

        # 双曲空间参数
        self.hyperbolic_dim = config.get('hyperbolic_dim', self.hidden_dims)
        self.lambda_ce = config.get('lambda_ce', 0.1)  # 蕴含损失权重
        self.lambda_cc = config.get('lambda_cc', 0.1)  # 对比损失权重
        self.temperature = config.get('temperature', 0.07)  # 对比学习温度

        # Sequence-branch task weights (MTR / TCL / align)
        self.lambda_mtr = config.get('lambda_mtr', 0.1)
        self.lambda_tcl = config.get('lambda_tcl', 0.1)
        self.lambda_align = config.get('lambda_align', 0.1)
        self.mtr_mask_ratio = config.get('mtr_mask_ratio', 0.15)
        self.mtr_neg_samples = config.get('mtr_neg_samples', 512)
        self.align_samples = config.get('align_samples', 500)
        self.align_neg_samples = config.get('align_neg_samples', 64)

        # ========== Phase 1: STS MR long-tail fix ==========
        self.use_momentum_queue = config.get('use_momentum_queue', True)
        self.queue_size = int(config.get('queue_size', 8192))
        self.queue_enqueue_size = int(config.get('queue_enqueue_size', 256))
        self.hard_neg_ratio = float(config.get('hard_neg_ratio', 0.5))
        self.contrast_neg_samples = int(config.get('contrast_neg_samples', 256))
        self.queue_warmup_steps = int(config.get('queue_warmup_steps', 50))
        self.lambda_rank = float(config.get('lambda_rank', 0.05))
        self.triplet_margin_base = float(config.get('triplet_margin_base', 0.2))
        self.tau_min = float(config.get('tau_min', 0.03))
        self.use_adaptive_tau = config.get('use_adaptive_tau', True)
        self.rank_sample_pairs = int(config.get('rank_sample_pairs', 256))
        self.rank_neg_samples = int(config.get('rank_neg_samples', 1024))
        self._global_step = 0

        # ========== Phase 2: TTE RMSE tail fix ==========
        self.lambda_tte_aux = float(config.get('lambda_tte_aux', 0.05))
        self.lambda_tt_robust = float(config.get('lambda_tt_robust', 0.05))
        self.lambda_time_var = float(config.get('lambda_time_var', 0.03))
        self.use_step_emb = config.get('use_step_emb', True)
        self.n_step_buckets = int(config.get('n_step_buckets', 16))
        self.huber_delta_ratio = float(config.get('huber_delta_ratio', 0.5))

        # ========== Phase 3: trajectory memory + multi-level readout ==========
        self.use_traj_memory = config.get('use_traj_memory', True)
        self.traj_memory_momentum = float(config.get('traj_memory_momentum', 0.99))
        self.use_multi_level_readout = config.get('use_multi_level_readout', True)
        self.readout_alpha_init = float(config.get('readout_alpha_init', 0.2))
        self.readout_beta_init = float(config.get('readout_beta_init', 0.1))
        self.readout_gamma_init = float(config.get('readout_gamma_init', 0.2))

        # ========== Phase 4: training strategy refinement ==========
        self.curvature_warmup_epochs = int(config.get('curvature_warmup_epochs', 20))
        self.curvature_init = float(config.get('curvature_init', 0.1))
        self.log_k_lr_mult = float(config.get('log_k_lr_mult', 0.1))
        self.hyp_grad_clip = float(config.get('hyp_grad_clip', 1.0))
        self.use_loss_schedule = config.get('use_loss_schedule', True)
        self.loss_stage1_epochs = int(config.get('loss_stage1_epochs', 30))
        self.loss_stage2_epochs = int(config.get('loss_stage2_epochs', 70))
        self._current_epoch = 0

        # 双曲空间工具
        self.manifold = LorentzManifold()
        self.entailment_cone = EntailmentCone(self.manifold)

        # Per-level learnable log-curvature (softplus > 0). Init at 0 → k=ln(2)≈0.69.
        # Scaled Lorentz distance d_k = sqrt(k) * d acts as per-level curvature.
        init_log_k = config.get('init_log_k', 0.5413)  # softplus(0.5413)=1.0
        self.log_k_seg = nn.Parameter(torch.tensor(float(init_log_k)))
        self.log_k_loc = nn.Parameter(torch.tensor(float(init_log_k)))
        self.log_k_reg = nn.Parameter(torch.tensor(float(init_log_k)))

        # Cross-view gate: sigmoid( MLP( [graph_emb; seq_emb] ) ) per position.
        # Modulates how strongly L_align anchor is pulled from graph-view toward seq-view.
        gate_hidden = config.get('xview_gate_hidden', 64)
        d1 = self.hyperbolic_dim + 1
        self.xview_gate = nn.Sequential(
            nn.Linear(2 * d1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        ).to(self.device)

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

        # Phase 1: Lorentz momentum queue
        if self.use_momentum_queue:
            self.seg_queue = HyperbolicMomentumQueue(
                dim=self.hyperbolic_dim + 1,
                size=self.queue_size,
                device=self.device,
            )
        else:
            self.seg_queue = None

        # =============== Sequence 分支（可选） ===============
        self.traj_dataloader = data_feature.get('traj_dataloader')
        self.traj_pad_idx = data_feature.get('traj_pad_idx', hparams.node_num)
        traj_max_len = data_feature.get('traj_max_len', config.get('max_len', 128))

        if self.traj_dataloader is not None:
            n_step_buckets = self.n_step_buckets if self.use_step_emb else 0
            self.traj_encoder = HyperbolicTrajEncoder(
                d=self.hyperbolic_dim,
                max_len=traj_max_len,
                n_layers=config.get('traj_n_layers', 2),
                n_heads=config.get('traj_n_heads', 4),
                dropout=config.get('traj_dropout', 0.1),
                manifold=self.manifold,
                n_step_buckets=n_step_buckets,
            ).to(self.device)
            self._logger.info(
                f"Sequence branch enabled (traj_max_len={traj_max_len}, "
                f"pad_idx={self.traj_pad_idx}, step_buckets={n_step_buckets})"
            )
        else:
            self.traj_encoder = None

        # ========== Phase 2: aux heads + targets ==========
        d1 = self.hyperbolic_dim + 1
        # Segment-level length recovery head (TTE-aligned info preservation)
        self.head_seg_time_mu = nn.Sequential(
            nn.Linear(d1, 64), nn.GELU(), nn.Linear(64, 1)
        ).to(self.device)
        # Segment-level variance-proxy head (uses topological degree as target)
        self.head_seg_time_var = nn.Sequential(
            nn.Linear(d1, 64), nn.GELU(), nn.Linear(64, 1)
        ).to(self.device)
        # Trajectory-level total-length head (Huber regression, TTE magnitude proxy)
        self.head_tt_total = nn.Sequential(
            nn.Linear(d1, 128), nn.GELU(), nn.Linear(128, 1)
        ).to(self.device)

        # Precompute normalized per-segment length & degree targets (used as TTE proxies
        # because real trajectory timestamps are not exposed via traj_dataloader).
        with torch.no_grad():
            length_t = self.length_feature.float()
            log_len = torch.log1p(length_t)
            seg_len_norm = (log_len - log_len.mean()) / (log_len.std() + 1e-6)
            self.register_buffer('_seg_len_target', seg_len_norm.detach().unsqueeze(-1))

            # Degree from adjacency (as variance proxy: branchier segments → more time variance)
            adj_indices = self.adj.indices()
            N_all = self.adj.shape[0]
            deg = torch.zeros(N_all, device=self.device)
            ones = torch.ones(adj_indices.shape[1], device=self.device)
            deg.index_add_(0, adj_indices[0], ones)
            log_deg = torch.log1p(deg)
            seg_deg_norm = (log_deg - log_deg.mean()) / (log_deg.std() + 1e-6)
            self.register_buffer('_seg_deg_target', seg_deg_norm.detach().unsqueeze(-1))

        # ========== Phase 3: trajectory memory + multi-level readout ==========
        # Segment-wise EMA memory of tangent-space trajectory token embeddings.
        # Carries trajectory semantics into segment representations without
        # changing encode() interface.
        if self.use_traj_memory:
            N_all = self.length_feature.shape[0]
            # Stored in tangent-at-origin (space dims only), zero-initialized
            traj_mem = torch.zeros(N_all, self.hyperbolic_dim, device=self.device)
            self.register_buffer('seg_traj_memory', traj_mem)
            self.register_buffer('seg_traj_mem_init', torch.zeros(N_all, dtype=torch.bool, device=self.device))
        else:
            self.seg_traj_memory = None

        # Learnable readout gates (in log-space to keep positive; softplus to blend)
        self.readout_alpha = nn.Parameter(torch.tensor(float(np.log(np.exp(self.readout_alpha_init) - 1 + 1e-6))))
        self.readout_beta = nn.Parameter(torch.tensor(float(np.log(np.exp(self.readout_beta_init) - 1 + 1e-6))))
        self.readout_gamma = nn.Parameter(torch.tensor(float(np.log(np.exp(self.readout_gamma_init) - 1 + 1e-6))))
        # 最近一次 sequence 前向产生的嵌入；预留给 step 2 的 MTR/TCL/align
        self.seq_token_hyp_emb = None  # [B, T, d+1]
        self.traj_hyp_emb = None       # [B, d+1]
        # =====================================================

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

    def traj_encode(self, seq, pad_mask):
        """
        Sequence 分支前向：把轨迹 (road-id 序列) 映射为 Lorentz 空间中的
        逐位置嵌入 token_hyp 与轨迹级嵌入 traj_hyp。
        复用 graph_enc.segment_hyp_emb 作为 token 表征来源。

        Args:
            seq:      [B, T] long, 含 pad_idx
            pad_mask: [B, T] bool, True 为有效位
        Returns:
            token_hyp: [B, T, d+1]
            traj_hyp:  [B, d+1]
        """
        if self.traj_encoder is None:
            raise RuntimeError("sequence 分支未启用 (traj_encoder is None)")
        # 确保 graph 前向已执行、segment_hyp_emb 可用
        if self.graph_enc.segment_hyp_emb is None:
            _ = self.graph_enc(
                self.node_feature, self.type_feature,
                self.length_feature, self.lane_feature, self.adj,
            )
        segment_emb = self.graph_enc.segment_hyp_emb
        token_hyp, traj_hyp = self.traj_encoder(segment_emb, seq, pad_mask)
        self.seq_token_hyp_emb = token_hyp
        self.traj_hyp_emb = traj_hyp
        return token_hyp, traj_hyp

    # ---------- Multi-curvature helpers ----------
    def _k(self, level):
        """
        Positive per-level curvature scalar with Phase-4 warmup.
        Ramps effective k from `curvature_init` (Euclidean-ish) toward the
        learned softplus(log_k) across `curvature_warmup_epochs` epochs.
        """
        log_k = getattr(self, f'log_k_{level}')
        k_learned = F.softplus(log_k) + 1e-6
        if self.curvature_warmup_epochs > 0 and self._current_epoch < self.curvature_warmup_epochs:
            progress = float(self._current_epoch) / float(self.curvature_warmup_epochs)
            k_init_t = torch.tensor(self.curvature_init, device=k_learned.device, dtype=k_learned.dtype)
            return k_init_t + progress * (k_learned - k_init_t)
        return k_learned

    def _sqrtk(self, level):
        return torch.sqrt(self._k(level))

    def _loss_schedule(self, epoch):
        """
        3-stage loss reweighting:
          stage1 (warm-start)  : down-weight contrastive/entailment/rank/aux, stabilize base
          stage2 (balanced)    : base weights from config
          stage3 (specialize)  : up-weight rank + TTE-aux for task-specific fine-tuning
        Returns dict of multipliers applied on top of base lambdas.
        """
        if not self.use_loss_schedule:
            return {'ce': 1.0, 'cc': 1.0, 'rank': 1.0, 'mtr': 1.0, 'tcl': 1.0,
                    'align': 1.0, 'tte': 1.0}
        if epoch < self.loss_stage1_epochs:
            return {'ce': 0.5, 'cc': 0.5, 'rank': 0.0, 'mtr': 1.0, 'tcl': 1.0,
                    'align': 1.0, 'tte': 0.3}
        if epoch < self.loss_stage2_epochs:
            return {'ce': 1.0, 'cc': 1.0, 'rank': 1.0, 'mtr': 1.0, 'tcl': 1.0,
                    'align': 1.0, 'tte': 1.0}
        return {'ce': 1.0, 'cc': 1.2, 'rank': 2.0, 'mtr': 0.8, 'tcl': 0.8,
                'align': 1.0, 'tte': 1.5}

    def _build_optimizer(self, base_lr):
        """
        Phase 4: separate param groups so hyperbolic params (log_k_*, Lorentz
        embedding layer, hyperbolic graph convs) train with a reduced lr.
        """
        hyp_params, other_params = [], []
        hyp_keywords = ('log_k_', 'hyp_embedding', 'fnc_gcn', 'struct_gcn', 'node_gcn',
                        'readout_alpha', 'readout_beta', 'readout_gamma')
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in n for kw in hyp_keywords):
                hyp_params.append(p)
            else:
                other_params.append(p)
        self._hyp_params_list = hyp_params
        self._other_params_list = other_params
        return torch.optim.Adam([
            {'params': other_params, 'lr': base_lr},
            {'params': hyp_params, 'lr': base_lr * self.log_k_lr_mult},
        ])

    def _tau(self, level):
        """
        Temperature: adaptive per level (scales with curvature) or fixed.
        """
        if self.use_adaptive_tau:
            log_k = getattr(self, f'log_k_{level}')
            return adaptive_temperature(log_k, self.temperature, self.tau_min)
        return torch.tensor(self.temperature, device=self.device)

    def _infonce(self, anchor, pos, neg, level):
        """
        Lorentz-distance InfoNCE with per-level curvature scaling.
            anchor: [B, d+1]
            pos:    [B, d+1]
            neg:    [K, d+1]
            level:  'seg' | 'loc' | 'reg'
        """
        s = self._sqrtk(level)
        tau = self._tau(level)
        pos_dist = self.manifold.lorentz_distance(anchor, pos) * s          # [B]
        neg_dist = self.manifold.pairwise_lorentz_distance(anchor, neg) * s # [B, K]
        logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist], dim=1) / tau
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    def _sample_negatives(self, anchor, exclude_ids, segment_emb, level='seg'):
        """
        Build a mixed negative pool:
          • In-batch shuffled (always)
          • Hard-mined from queue (via smallest Lorentz distance, exclude positives)
          • Random from queue / segment pool
        Args:
            anchor: [B, d+1]
            exclude_ids: [B] segment-ids to avoid when sampling (current positives)
            segment_emb: [N, d+1] fallback pool
        Returns:
            neg: [K, d+1]
        """
        K_total = self.contrast_neg_samples
        N_seg = segment_emb.shape[0]

        use_queue = (
            self.seg_queue is not None
            and int(self.seg_queue.queue_filled.item()) >= self.queue_enqueue_size
            and self._global_step >= self.queue_warmup_steps
        )

        if not use_queue:
            # Cold start: random sample from segment pool
            K_eff = min(K_total, N_seg)
            idx = torch.randint(0, N_seg, (K_eff,), device=self.device)
            return segment_emb[idx]

        pool = self.seg_queue.get()  # [Q, d+1]
        Q = pool.shape[0]
        n_hard = int(K_total * self.hard_neg_ratio)
        n_rand = K_total - n_hard

        # Random negatives from queue
        rand_idx = torch.randint(0, Q, (n_rand,), device=self.device)
        rand_neg = pool[rand_idx]

        if n_hard > 0:
            # Hard negatives: for each anchor, find top-n_hard nearest in queue,
            # aggregate a shared pool by union (with cap)
            s = self._sqrtk(level)
            with torch.no_grad():
                # Subsample queue to bound cost when Q is very large
                probe_cap = min(Q, 2048)
                if Q > probe_cap:
                    pidx = torch.randperm(Q, device=self.device)[:probe_cap]
                    probe = pool[pidx]
                else:
                    probe = pool
                dist = self.manifold.pairwise_lorentz_distance(anchor, probe) * s  # [B, P]
                # Pick a few hardest per anchor then flatten & dedup
                per_anchor_k = max(1, n_hard // max(1, anchor.shape[0]) + 2)
                per_anchor_k = min(per_anchor_k, probe.shape[0])
                _, top_idx = torch.topk(-dist, k=per_anchor_k, dim=1)  # smallest dist
                flat = top_idx.flatten()
                # Dedup & cap
                flat = torch.unique(flat)
                if flat.numel() > n_hard:
                    sel = torch.randperm(flat.numel(), device=self.device)[:n_hard]
                    flat = flat[sel]
            hard_neg = probe[flat]
            neg = torch.cat([hard_neg, rand_neg], dim=0)
        else:
            neg = rand_neg

        return neg

    def _xview_reliability(self, graph_emb, seq_emb):
        """
        Cross-view sigmoid gate scoring per-position reliability of graph/seq
        agreement. Used as a sample weight on the L_align InfoNCE term (not a
        fusion — fusion on Lorentz is collapse-prone when paired with a
        contrastive target).
            shapes: graph_emb / seq_emb: [S, d+1]
            returns: g: [S] in (0, 1)
        """
        return torch.sigmoid(
            self.xview_gate(torch.cat([graph_emb, seq_emb], dim=-1))
        ).squeeze(-1)

    def compute_seq_losses(self, seq, pad_mask):
        """
        Sequence 分支三个多样化预训练任务:
          - L_MTR  : masked trajectory recovery (sampled softmax 在 segment 全集上)
          - L_TCL  : trajectory-level InfoNCE  (view A vs view B, 对称)
          - L_align: segment-level graph<->sequence alignment (InfoNCE)
        Args:
            seq:      [B, T] long
            pad_mask: [B, T] bool (True 为有效)
        Returns:
            (L_MTR, L_TCL, L_align) — 每个都是 scalar tensor
        """
        zero = torch.zeros((), device=self.device)
        if self.traj_encoder is None:
            return zero, zero, zero

        # graph 前向已在 encode() 被调用；兜底
        if self.graph_enc.segment_hyp_emb is None:
            _ = self.graph_enc(
                self.node_feature, self.type_feature,
                self.length_feature, self.lane_feature, self.adj,
            )
        segment_emb = self.graph_enc.segment_hyp_emb        # [N, d+1]
        N = segment_emb.shape[0]
        B, T = seq.shape
        tau = self.temperature
        s_seg = self._sqrtk('seg')  # per-level curvature scaling for segment level

        # ---------- view A (unmasked) ----------
        token_A, traj_A = self.traj_encoder(segment_emb, seq, pad_mask)
        self.seq_token_hyp_emb = token_A
        self.traj_hyp_emb = traj_A

        # ---------- view B (masked) ----------
        rand = torch.rand(B, T, device=self.device)
        mask_positions = pad_mask & (rand < self.mtr_mask_ratio)  # [B, T]
        token_B, traj_B = self.traj_encoder(
            segment_emb, seq, pad_mask, mask_positions=mask_positions
        )

        # ==================== L_MTR ====================
        mp = mask_positions.nonzero(as_tuple=False)  # [P, 2]
        if mp.shape[0] > 0:
            anchor = token_B[mp[:, 0], mp[:, 1]]              # [P, d+1]
            pos_ids = seq[mp[:, 0], mp[:, 1]].clamp(max=N - 1)
            pos_emb = segment_emb[pos_ids]                    # [P, d+1]
            K = min(self.mtr_neg_samples, N)
            neg_ids = torch.randint(0, N, (K,), device=self.device)
            neg_emb = segment_emb[neg_ids]                    # [K, d+1]
            L_MTR = self._infonce(anchor, pos_emb, neg_emb, level='seg')
        else:
            L_MTR = zero

        # ==================== L_TCL ====================
        # 对称 InfoNCE: traj_A[i] 与 traj_B[i] 为正对, 其他轨迹为负
        # 轨迹级仍居于 segment 曲率（trajectory 由 segment tokens 构成）
        if B > 1:
            dist_ab = self.manifold.pairwise_lorentz_distance(traj_A, traj_B) * s_seg  # [B, B]
            logits_ab = -dist_ab / tau
            labels_b = torch.arange(B, device=self.device)
            L_TCL = 0.5 * (
                F.cross_entropy(logits_ab, labels_b)
                + F.cross_entropy(logits_ab.t(), labels_b)
            )
        else:
            L_TCL = zero

        # ==================== L_align ====================
        # anchor=seq_view, pos=graph_view, with per-position reliability gate.
        # Gate weights each sample's CE; regularizer keeps gate away from 0.
        valid = pad_mask.nonzero(as_tuple=False)  # [V, 2]
        if valid.shape[0] > 0:
            S = min(self.align_samples, valid.shape[0])
            sel = torch.randperm(valid.shape[0], device=self.device)[:S]
            vb = valid[sel]
            anchor = token_A[vb[:, 0], vb[:, 1]]               # [S, d+1] seq view
            pos_ids = seq[vb[:, 0], vb[:, 1]].clamp(max=N - 1)
            pos_emb = segment_emb[pos_ids]                     # [S, d+1] graph view (positive)

            K = min(self.align_neg_samples, N)
            neg_ids = torch.randint(0, N, (K,), device=self.device)
            neg_emb = segment_emb[neg_ids]                     # [K, d+1]

            pos_dist = self.manifold.lorentz_distance(anchor, pos_emb) * s_seg        # [S]
            neg_dist = self.manifold.pairwise_lorentz_distance(anchor, neg_emb) * s_seg  # [S, K]
            logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist], dim=1) / tau
            labels = torch.zeros(S, dtype=torch.long, device=self.device)
            per_sample = F.cross_entropy(logits, labels, reduction='none')  # [S]

            g = self._xview_reliability(pos_emb, anchor)       # [S] in (0, 1)
            gate_reg = self.config.get('xview_gate_reg', 0.01)
            # Non-degenerate weighted CE: entropy-style reg keeps gate from collapsing to 0.
            L_align = (g * per_sample).mean() - gate_reg * torch.log(g + 1e-6).mean()
        else:
            L_align = zero

        return L_MTR, L_TCL, L_align

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

            # Phase 3: use multi-level enriched embedding (shape unchanged [N, d+1])
            enriched = self._build_enriched_segment_embedding()
            if enriched is None:
                enriched = self.graph_enc.segment_hyp_emb
            node_embedding = enriched.data.cpu().numpy()

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
        Phase 1 hierarchical contrastive:
          1. Segment-level InfoNCE with queue + hard-negative mining
          2. Cross-level (Locality ↔ Segment) InfoNCE with mixed negatives
        Temperature is curvature-adaptive when use_adaptive_tau=True.
        Enqueues a random batch of current-segment Lorentz embeddings each call.
        """
        segment_emb = self.graph_enc.segment_hyp_emb         # [N, d+1]
        locality_emb = self.graph_enc.locality_hyp_emb       # [N_loc, d+1]
        N_seg = segment_emb.shape[0]

        # Enqueue a random subset so that queue tracks the evolving encoder
        if self.seg_queue is not None:
            with torch.no_grad():
                enq_k = min(self.queue_enqueue_size, N_seg)
                enq_idx = torch.randperm(N_seg, device=self.device)[:enq_k]
                self.seg_queue.enqueue(segment_emb[enq_idx])

        losses = []

        # 1. Segment level: adjacent pairs as positives
        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        if num_edges > 0:
            B = min(500, num_edges)
            sel = torch.randperm(num_edges, device=self.device)[:B]
            anchor_ids = edge_indices[0, sel]
            pos_ids = edge_indices[1, sel]
            anchor_emb = segment_emb[anchor_ids]
            pos_emb = segment_emb[pos_ids]
            neg_emb = self._sample_negatives(anchor_emb, pos_ids, segment_emb, level='seg')
            losses.append(self._infonce(anchor_emb, pos_emb, neg_emb, level='seg'))

        # 2. Cross-level: Locality anchor <-> Segment positive
        ls_pairs = (self.struct_assign > 0).nonzero(as_tuple=False)  # [K', 2]: (seg, loc)
        if ls_pairs.numel() > 0:
            M = min(500, ls_pairs.shape[0])
            sel = torch.randperm(ls_pairs.shape[0], device=self.device)[:M]
            sampled = ls_pairs[sel]
            anchor_emb = locality_emb[sampled[:, 1]]
            pos_ids = sampled[:, 0]
            pos_emb = segment_emb[pos_ids]
            neg_emb = self._sample_negatives(anchor_emb, pos_ids, segment_emb, level='loc')
            losses.append(self._infonce(anchor_emb, pos_emb, neg_emb, level='loc'))

        if len(losses) == 0:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _update_traj_memory(self, token_hyp, seq, pad_mask):
        """
        Phase 3: EMA update of per-segment trajectory memory in tangent-at-origin.
        Args:
            token_hyp: [B, T, d+1] Lorentz
            seq:       [B, T] long
            pad_mask:  [B, T] bool
        """
        if not self.use_traj_memory or self.seg_traj_memory is None:
            return
        if token_hyp is None or seq is None or pad_mask is None:
            return

        N = self.seg_traj_memory.shape[0]
        # Flatten valid positions
        mask_flat = pad_mask.reshape(-1)
        idx_flat = seq.reshape(-1).clamp(max=N - 1)
        # Work in origin tangent space (space dims only) — stable EMA there
        tok_tangent = self.manifold.origin_log_map(token_hyp)[..., 1:]   # [B, T, d]
        tok_flat = tok_tangent.reshape(-1, self.hyperbolic_dim)          # [B*T, d]

        valid_idx = idx_flat[mask_flat]
        valid_tok = tok_flat[mask_flat].detach()

        if valid_idx.numel() == 0:
            return

        # For segments touched in this batch, aggregate mean of tokens, then EMA update
        # Use index_add_ for mean accumulation
        sum_buf = torch.zeros_like(self.seg_traj_memory)
        cnt_buf = torch.zeros(N, device=self.device)
        sum_buf.index_add_(0, valid_idx, valid_tok)
        cnt_buf.index_add_(0, valid_idx, torch.ones_like(valid_idx, dtype=sum_buf.dtype))

        touched = cnt_buf > 0
        if not touched.any():
            return
        mean_buf = torch.zeros_like(self.seg_traj_memory)
        mean_buf[touched] = sum_buf[touched] / cnt_buf[touched].unsqueeze(-1)

        m = self.traj_memory_momentum
        # Where not initialized yet: set directly to avoid slow cold-start
        fresh = touched & ~self.seg_traj_mem_init
        warm = touched & self.seg_traj_mem_init

        if fresh.any():
            self.seg_traj_memory[fresh] = mean_buf[fresh]
            self.seg_traj_mem_init[fresh] = True
        if warm.any():
            self.seg_traj_memory[warm] = (
                m * self.seg_traj_memory[warm] + (1.0 - m) * mean_buf[warm]
            )

    def _traj_memory_lorentz(self):
        """Project current tangent-stored traj memory back to Lorentz at origin."""
        if not self.use_traj_memory or self.seg_traj_memory is None:
            return None
        v_space = self.seg_traj_memory                                   # [N, d]
        v_full = torch.cat([torch.zeros_like(v_space[..., :1]), v_space], dim=-1)
        return self.manifold.origin_exp_map(v_full)                      # [N, d+1]

    def _build_enriched_segment_embedding(self):
        """
        Phase 3: multi-level readout — fuse segment + locality + region + traj-memory
        in the origin tangent space, then exp_map back to Lorentz.
        Output shape is the same [N, d+1] as segment_hyp_emb so downstream code is
        unchanged.
        """
        seg = self.graph_enc.segment_hyp_emb
        if seg is None:
            return None
        if not self.use_multi_level_readout and not self.use_traj_memory:
            return seg

        N = seg.shape[0]
        v_seg = self.manifold.origin_log_map(seg)                        # [N, d+1]

        fused = v_seg.clone()
        if self.use_multi_level_readout:
            loc = self.graph_enc.locality_hyp_emb                        # [N_loc, d+1]
            reg = self.graph_enc.region_hyp_emb                          # [N_reg, d+1]
            if loc is not None:
                loc_bcast = self.struct_assign @ self.manifold.origin_log_map(loc)  # [N, d+1]
                alpha = F.softplus(self.readout_alpha)
                fused = fused + alpha * loc_bcast
            if reg is not None and loc is not None:
                reg_bcast = (self.struct_assign @ self.fnc_assign) @ self.manifold.origin_log_map(reg)
                beta = F.softplus(self.readout_beta)
                fused = fused + beta * reg_bcast

        if self.use_traj_memory and self.seg_traj_memory is not None:
            init_mask = self.seg_traj_mem_init.unsqueeze(-1).float()     # [N, 1]
            if init_mask.sum() > 0:
                traj_L = self._traj_memory_lorentz()
                v_traj = self.manifold.origin_log_map(traj_L)            # [N, d+1]
                gamma = F.softplus(self.readout_gamma)
                fused = fused + gamma * v_traj * init_mask

        # Zero time component (tangent at origin invariant) then exp_map back
        fused[..., 0] = 0.0
        enriched = self.manifold.origin_exp_map(fused)
        return enriched

    def compute_tte_aux_loss(self, seq=None, pad_mask=None, traj_hyp=None):
        """
        Phase 2: TTE-aligned auxiliary losses targeting RMSE tail behavior.
        Three heads, all operating on current Lorentz embeddings:

        1. Segment length recovery (μ-head): predict normalized log(length) per segment.
           Length correlates strongly with travel time; enforcing recoverability keeps
           TTE-relevant information in segment_hyp_emb after graph encoding.

        2. Segment degree prediction (variance-proxy head): predict normalized log(degree).
           Branchier intersections have higher travel-time variance; this signal pushes
           RMSE-relevant uncertainty into the representation.

        3. Trajectory total-length (Huber): predict Σ length[seq[:]] from traj_hyp.
           Huber δ tied to target median — robust to long-tail outliers that would
           otherwise dominate RMSE during training.

        Returns scalar tensor (0 if disabled or no data).
        """
        zero = torch.zeros((), device=self.device)
        if self.lambda_tte_aux <= 0 and self.lambda_time_var <= 0 and self.lambda_tt_robust <= 0:
            return zero

        segment_emb = self.graph_enc.segment_hyp_emb  # [N, d+1]
        if segment_emb is None:
            return zero
        N_seg = segment_emb.shape[0]

        losses = []

        # Head 1: segment length recovery
        if self.lambda_tte_aux > 0 and self._seg_len_target is not None:
            # Sample a subset to bound cost
            B_s = min(4096, N_seg)
            sel = torch.randperm(N_seg, device=self.device)[:B_s]
            pred_mu = self.head_seg_time_mu(segment_emb[sel]).squeeze(-1)     # [B_s]
            target_mu = self._seg_len_target[sel].squeeze(-1)                 # [B_s]
            losses.append(self.lambda_tte_aux * F.smooth_l1_loss(pred_mu, target_mu))

        # Head 2: variance proxy via degree
        if self.lambda_time_var > 0 and self._seg_deg_target is not None:
            B_s = min(4096, N_seg)
            sel = torch.randperm(N_seg, device=self.device)[:B_s]
            pred_var = self.head_seg_time_var(segment_emb[sel]).squeeze(-1)
            target_var = self._seg_deg_target[sel].squeeze(-1)
            losses.append(self.lambda_time_var * F.smooth_l1_loss(pred_var, target_var))

        # Head 3: trajectory total-length Huber regression
        if (
            self.lambda_tt_robust > 0
            and traj_hyp is not None
            and seq is not None
            and pad_mask is not None
        ):
            with torch.no_grad():
                safe_seq = seq.clamp(max=N_seg - 1)
                seg_len = self._seg_len_target.squeeze(-1)            # [N]
                # Per-position normalized length, masked then summed per trajectory
                per_pos_len = seg_len[safe_seq] * pad_mask.float()    # [B, T]
                tt_target = per_pos_len.sum(dim=1)                    # [B]
                # Robust delta: fraction of the absolute median magnitude
                med = torch.median(tt_target.abs()) + 1e-6
                delta = med * self.huber_delta_ratio

            pred_tt = self.head_tt_total(traj_hyp).squeeze(-1)        # [B]
            losses.append(self.lambda_tt_robust * F.huber_loss(
                pred_tt, tt_target, delta=float(delta.item())
            ))

        if len(losses) == 0:
            return zero
        return torch.stack(losses).sum()

    def compute_listwise_rank_loss(self):
        """
        Phase 1: Rank-aware hyperbolic triplet loss over large negative pool.
        For each (anchor, pos) edge pair, penalize every negative whose
        Lorentz distance is closer than (d_pos + margin).
          L = mean_b[ mean_n( ReLU(d_pos - d_neg + m) ) ]
        This directly targets Beijing STS MR long-tail (ACC@3 unaffected since
        it only penalizes ordering errors in the tail).
        """
        if self.lambda_rank <= 0.0:
            return torch.zeros((), device=self.device)

        segment_emb = self.graph_enc.segment_hyp_emb
        N_seg = segment_emb.shape[0]

        edge_indices = self.adj.indices()
        num_edges = edge_indices.shape[1]
        if num_edges == 0:
            return torch.zeros((), device=self.device)

        B = min(self.rank_sample_pairs, num_edges)
        sel = torch.randperm(num_edges, device=self.device)[:B]
        anchor = segment_emb[edge_indices[0, sel]]
        pos = segment_emb[edge_indices[1, sel]]

        # Build a large negative pool (prefer queue)
        if self.seg_queue is not None and int(self.seg_queue.queue_filled.item()) >= 512:
            pool = self.seg_queue.get()
            K_eff = min(self.rank_neg_samples, pool.shape[0])
            idx = torch.randperm(pool.shape[0], device=self.device)[:K_eff]
            neg = pool[idx]
        else:
            K_eff = min(self.rank_neg_samples, N_seg)
            idx = torch.randint(0, N_seg, (K_eff,), device=self.device)
            neg = segment_emb[idx]

        s = self._sqrtk('seg')
        d_pos = self.manifold.lorentz_distance(anchor, pos) * s              # [B]
        d_neg = self.manifold.pairwise_lorentz_distance(anchor, neg) * s      # [B, K]

        # Curvature-coupled margin: larger k → larger margin (distances spread out)
        margin = self.triplet_margin_base * (1.0 + s.detach())
        triplet = F.relu(d_pos.unsqueeze(1) - d_neg + margin)                 # [B, K]
        return triplet.mean()

    def run(self, train_dataloader, eval_dataloader):
        """训练循环"""
        self._logger.info("Starting training with Hyperbolic Embeddings...")
        hparams = dict_to_object(self.config.config)
        ce_criterion = torch.nn.CrossEntropyLoss()
        max_f1 = 0
        max_auc = 0
        count = 0
        # Phase 4: param-grouped optimizer (hyperbolic params get reduced lr)
        model_optimizer = self._build_optimizer(hparams.lp_learning_rate)
        self._logger.info(
            f"Phase4 optimizer: other_params={len(self._other_params_list)}, "
            f"hyp_params={len(self._hyp_params_list)} (lr_mult={self.log_k_lr_mult})"
        )
        eval_dataloader_iter = iter(eval_dataloader)
        patience = 50
        traj_iter = iter(self.traj_dataloader) if self.traj_dataloader is not None else None

        for i in range(hparams.max_epoch):
            self._current_epoch = i
            sched = self._loss_schedule(i)
            self._logger.info(
                f"epoch {i}, processed {count}, sched={sched}, "
                f"k_warmup={min(1.0, i / max(1, self.curvature_warmup_epochs)):.2f}"
            )
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

                # Phase 1: listwise rank loss for MR long-tail
                loss_rank = self.compute_listwise_rank_loss()

                # Sequence 分支: MTR / TCL / align 多任务
                loss_mtr = torch.zeros((), device=self.device)
                loss_tcl = torch.zeros((), device=self.device)
                loss_align = torch.zeros((), device=self.device)
                loss_tte = torch.zeros((), device=self.device)
                last_seq, last_mask = None, None
                if self.traj_encoder is not None:
                    traj_batch = get_next(traj_iter)
                    if traj_batch is None:
                        traj_iter = iter(self.traj_dataloader)
                        traj_batch = get_next(traj_iter)
                    if traj_batch is not None:
                        seq_b, mask_b = traj_batch
                        seq_b = seq_b.to(self.device)
                        mask_b = mask_b.to(self.device).bool()
                        loss_mtr, loss_tcl, loss_align = self.compute_seq_losses(seq_b, mask_b)
                        last_seq, last_mask = seq_b, mask_b
                        # Phase 3: EMA-update per-segment trajectory memory
                        self._update_traj_memory(self.seq_token_hyp_emb, seq_b, mask_b)

                # Phase 2: TTE aux loss (segment μ/var + trajectory Huber)
                loss_tte = self.compute_tte_aux_loss(
                    seq=last_seq, pad_mask=last_mask, traj_hyp=self.traj_hyp_emb,
                )

                # Phase 4: scheduled per-loss multipliers
                loss = (
                    loss_struct
                    + sched['ce'] * self.lambda_ce * loss_ce
                    + sched['cc'] * self.lambda_cc * loss_cc
                    + sched['rank'] * self.lambda_rank * loss_rank
                    + sched['mtr'] * self.lambda_mtr * loss_mtr
                    + sched['tcl'] * self.lambda_tcl * loss_tcl
                    + sched['align'] * self.lambda_align * loss_align
                    + sched['tte'] * loss_tte
                )

                loss.backward()
                # Phase 4: group-specific grad clipping — tighter bound for hyperbolic params
                if self._hyp_params_list:
                    torch.nn.utils.clip_grad_norm_(self._hyp_params_list, self.hyp_grad_clip)
                if self._other_params_list:
                    torch.nn.utils.clip_grad_norm_(self._other_params_list, hparams.lp_clip)
                model_optimizer.step()
                self._global_step += 1

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
                            # Phase 3: save the enriched multi-level + traj-memory fused embedding
                            enriched = self._build_enriched_segment_embedding()
                            if enriched is None:
                                enriched = self.graph_enc.segment_hyp_emb
                            node_embedding = enriched.data.cpu().numpy()
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
                    self._logger.info(
                        f"loss: {loss.item()}, struct: {loss_struct.item()}, "
                        f"ce: {loss_ce.item()}, cc: {loss_cc.item()}, "
                        f"rank: {loss_rank.item()}, tte: {loss_tte.item()}, "
                        f"mtr: {loss_mtr.item()}, tcl: {loss_tcl.item()}, "
                        f"align: {loss_align.item()}, "
                        f"k_seg: {self._k('seg').item():.3f}, "
                        f"k_loc: {self._k('loc').item():.3f}, "
                        f"k_reg: {self._k('reg').item():.3f}"
                    )
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


class HyperbolicTrajEncoder(nn.Module):
    """
    切空间 Transformer 轨迹编码器：
      1) 在原点对 Lorentz 输入做 log_map → 切空间表示
      2) 加可学习的 positional embedding + Phase2 step-scale (Δt-proxy) 嵌入
      3) 标准 Transformer encoder (batch_first)
      4) 原点 exp_map 映回 Lorentz 空间
      5) 对 padding 位做 mask，对轨迹级嵌入做 masked mean pool
    """
    def __init__(self, d, max_len, n_layers=2, n_heads=4, dropout=0.1, manifold=None,
                 n_step_buckets=0):
        super().__init__()
        self.manifold = manifold if manifold is not None else LorentzManifold()
        self.d = d
        self.max_len = max_len

        self.pos_emb = nn.Parameter(torch.zeros(max_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Phase 2: Δt-proxy step-scale bucket embedding (log-bucketed token position).
        # When actual timestamps are unavailable, log-position gives scale-aware
        # information beyond the linear positional embedding.
        self.n_step_buckets = int(n_step_buckets)
        if self.n_step_buckets > 0:
            self.step_emb = nn.Parameter(torch.zeros(self.n_step_buckets, d))
            nn.init.trunc_normal_(self.step_emb, std=0.02)
            # Precompute bucket id for each position: floor(log2(t+1)), clamped
            bucket_ids = torch.zeros(max_len, dtype=torch.long)
            for t in range(max_len):
                b = int(np.floor(np.log2(t + 1))) if t > 0 else 0
                bucket_ids[t] = min(b, self.n_step_buckets - 1)
            self.register_buffer('step_bucket_ids', bucket_ids)
        else:
            self.step_emb = None

        # Learnable [MASK] token in tangent space (space dims only)
        self.mask_tangent = nn.Parameter(torch.zeros(d))
        nn.init.trunc_normal_(self.mask_tangent, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, segment_hyp_emb, seq, pad_mask, mask_positions=None):
        """
        Args:
            segment_hyp_emb: [N, d+1] graph-view Lorentz 嵌入
            seq:             [B, T] long (含 pad_idx)
            pad_mask:        [B, T] bool, True 为有效位
            mask_positions:  [B, T] bool or None. True 处用可学习 [MASK] 替换（仅影响有效位）
        Returns:
            token_hyp: [B, T, d+1]  Lorentz 空间逐位置嵌入
            traj_hyp:  [B, d+1]     Lorentz 空间轨迹级嵌入
        """
        N = segment_hyp_emb.shape[0]
        # Pad 位置用任意合法 id 兜底 (mask 会屏蔽其贡献)
        safe_ids = seq.clamp(max=N - 1)
        h = segment_hyp_emb[safe_ids]                       # [B, T, d+1]

        # 原点切空间
        v = self.manifold.origin_log_map(h)                 # [B, T, d+1], v[..., 0] == 0
        v_space = v[..., 1:]                                # [B, T, d]

        # [MASK] 替换（在加 pos_emb 之前）
        if mask_positions is not None:
            mf = mask_positions.unsqueeze(-1).float()       # [B, T, 1]
            v_space = v_space * (1.0 - mf) + self.mask_tangent.view(1, 1, -1) * mf

        T = v_space.shape[1]
        v_space = v_space + self.pos_emb[:T].unsqueeze(0)   # broadcast 到 [B, T, d]
        # Phase 2: add step-scale (log-bucketed) embedding
        if self.step_emb is not None:
            bucket_ids = self.step_bucket_ids[:T]            # [T]
            step_e = self.step_emb[bucket_ids].unsqueeze(0)  # [1, T, d]
            v_space = v_space + step_e

        key_padding_mask = ~pad_mask                        # True = ignore
        out = self.transformer(v_space, src_key_padding_mask=key_padding_mask)  # [B, T, d]

        # 逐位置 Lorentz 嵌入
        zeros_time = torch.zeros_like(out[..., :1])
        v_out = torch.cat([zeros_time, out], dim=-1)        # [B, T, d+1]
        token_hyp = self.manifold.origin_exp_map(v_out)     # [B, T, d+1]

        # 轨迹级嵌入 (masked mean in tangent space, then exp_map)
        mask_f = pad_mask.unsqueeze(-1).float()             # [B, T, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)            # [B, 1]
        traj_space = (out * mask_f).sum(dim=1) / denom      # [B, d]
        traj_tangent = torch.cat(
            [torch.zeros_like(traj_space[..., :1]), traj_space], dim=-1
        )                                                    # [B, d+1]
        traj_hyp = self.manifold.origin_exp_map(traj_tangent)
        return token_hyp, traj_hyp


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
