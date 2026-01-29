# 空间分量坍缩问题的修复方案

## 问题诊断

当前状态:
- 空间分量不是严格的0,但非常小 (~1e-4)
- 所有embedding聚集在Lorentz原点 [1, 0, 0, ..., 0] 附近
- 导致所有点之间距离相似,模型无法区分不同的segment

**重要更新**: 经过深入诊断发现:
- ✅ HyperbolicEmbedding层是**正常**的 (空间范数~8.9)
- ❌ 问题出在**图编码器的层次化传播过程**中
- 空间分量在经过3层图编码器后从~8.9坍缩到~0.0003

根本原因:
- **不是** HyperbolicEmbedding层的问题
- **是** 图编码器中的某一层导致空间分量坍缩
- 需要运行 `diagnose_graph_encoder_collapse.py` 定位具体是哪一层

## 修复方案

**注意**: 以下方案1-4针对HyperbolicEmbedding层的问题。根据最新诊断，问题在图编码器中，请优先查看**方案5**。

### 方案1: 增加线性层初始化scale (可能不适用)

**位置**: `veccity/upstream/road_representation/hyperbolic_utils.py:181`

**当前代码**:
```python
self.linear = nn.Linear(euclidean_dim, hyperbolic_dim)
```

**修改为**:
```python
self.linear = nn.Linear(euclidean_dim, hyperbolic_dim)
# 增大初始化scale,使输出向量更分散
nn.init.xavier_uniform_(self.linear.weight, gain=5.0)  # 尝试gain=5.0或10.0
if self.linear.bias is not None:
    nn.init.zeros_(self.linear.bias)
```

**原理**:
- Xavier初始化默认gain=1.0,输出方差可能太小
- 增大gain使线性变换输出更大的向量
- 从而使投影后的空间分量更大

### 方案2: 使用exp_map从原点映射

**位置**: `veccity/upstream/road_representation/hyperbolic_utils.py:183-194`

**当前代码**:
```python
def forward(self, x):
    x_transformed = self.linear(x)
    h = self.manifold.project_to_lorentz(x_transformed)
    return h
```

**修改为**:
```python
def forward(self, x):
    x_transformed = self.linear(x)
    # 使用exp_map从原点出发,而不是直接投影
    origin = torch.zeros(x.shape[0], x_transformed.shape[1] + 1, device=x.device)
    origin[:, 0] = 1.0  # Lorentz原点
    # x_transformed作为切向量
    tangent = torch.cat([torch.zeros(x.shape[0], 1, device=x.device), x_transformed], dim=-1)
    h = self.manifold.exp_map(origin, tangent)
    return h
```

**原理**:
- exp_map可以更好地控制从原点到目标点的映射
- 切向量的范数直接控制在流形上的距离

### 方案3: 添加缩放因子 (最简单)

**位置**: `veccity/upstream/road_representation/hyperbolic_utils.py:191`

**当前代码**:
```python
x_transformed = self.linear(x)
h = self.manifold.project_to_lorentz(x_transformed)
```

**修改为**:
```python
x_transformed = self.linear(x)
# 添加缩放因子,使空间分量更大
scale = 10.0  # 可调节的超参数
x_transformed = x_transformed * scale
h = self.manifold.project_to_lorentz(x_transformed)
```

**原理**:
- 直接放大线性变换的输出
- 最简单,容易调试
- scale可以作为超参数grid search

### 方案4: 检查embedding层初始化

**位置**: `veccity/upstream/road_representation/HRNR_Hyperbolic.py:468-471`

**检查**:
```python
self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.device)
self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.device)
self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.device)
self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.device)
```

**确认**:
- 这些embedding层是否被正确初始化
- 可以在`__init__`后添加初始化:
  ```python
  nn.init.normal_(self.node_emb_layer.weight, mean=0.0, std=0.1)
  nn.init.normal_(self.type_emb_layer.weight, mean=0.0, std=0.1)
  nn.init.normal_(self.length_emb_layer.weight, mean=0.0, std=0.1)
  nn.init.normal_(self.lane_emb_layer.weight, mean=0.0, std=0.1)
  ```

### 方案5: 修复图编码器中的空间坍缩 (推荐 - 根据最新诊断)

**诊断确认**: 运行 `diagnose_graph_encoder_collapse.py` 确定是哪一层导致坍缩。

可能的问题点：

#### 5.1 检查聚合操作的数值稳定性

**位置**: `veccity/upstream/road_representation/HRNR_Hyperbolic.py:700-730` (HyperbolicGraphEncoderTLCore)

聚合操作 `_aggregate_to_cluster` 可能导致空间分量归零。检查：

```python
def _aggregate_to_cluster(self, hyp_feat, assign_matrix):
    """从细粒度聚合到粗粒度"""
    # 当前实现可能有问题
```

**可能修复**: 检查是否在聚合时使用了不当的平均操作，导致双曲空间中的向量被错误地平均。

#### 5.2 检查双曲图卷积的实现

**位置**: `veccity/upstream/road_representation/hyperbolic_utils.py` (HyperbolicGraphConv类)

检查：
- log_map 和 exp_map 是否正确实现
- 是否有数值下溢导致切向量过小
- 聚合后的切向量是否被错误地缩放

**调试建议**:
```python
# 在 HyperbolicGraphConv.forward() 中添加调试输出
def forward(self, x, adj):
    # 添加这些检查
    print(f"输入空间范数: {self.spatial_norm(x).mean():.6f}")

    # ... 现有代码 ...

    print(f"输出空间范数: {self.spatial_norm(output).mean():.6f}")

    # 如果输出范数 << 输入范数，说明这一层有问题
```

#### 5.3 检查门控机制

**位置**: `veccity/upstream/road_representation/HRNR_Hyperbolic.py:660-680`

门控操作可能导致空间分量被过度抑制：

```python
# 检查门控值
gate_values = self.sigmoid(self.l_c(...))
print(f"门控值范围: [{gate_values.min():.6f}, {gate_values.max():.6f}]")

# 如果门控值接近0，会导致空间分量被抑制
```

**可能修复**: 如果门控值过小，调整门控网络的初始化或添加偏置。

#### 5.4 添加残差连接（推荐快速测试）

**位置**: 在每层图编码器输出时添加残差

**修改** `HyperbolicGraphEncoderTLCore.forward()`:

```python
def forward(self, struct_adj, hyp_feat, raw_adj):
    # 保存输入
    input_feat = hyp_feat

    # ... 现有的前向传播 ...

    # 在输出前添加残差连接（在双曲空间中）
    # 使用 log_map 和 exp_map 实现双曲残差
    v_input = self.manifold.log_map(output, input_feat)
    output_with_residual = self.manifold.exp_map(output, 0.1 * v_input)  # 0.1是残差权重

    return output_with_residual
```

**原理**: 残差连接可以防止空间分量在多层传播中完全消失。

## 推荐流程（更新）

**根据最新诊断，问题在图编码器中，按此流程操作：**

1. **运行图编码器诊断** `diagnose_graph_encoder_collapse.py`
   - 确定是哪一层导致空间分量坍缩
   - 查看每层的空间范数变化比例

2. **根据诊断结果选择修复方案**:
   - 如果某一层导致范数大幅减小 (比例<0.1):
     - 检查该层的聚合操作 (方案5.1)
     - 检查双曲图卷积实现 (方案5.2)
     - 检查门控机制 (方案5.3)
   - **快速测试**: 添加残差连接 (方案5.4) - 最简单的修复

3. **如果问题不在图编码器** (运行 `diagnose_spatial_collapse.py` 确认):
   - 如果线性层权重太小: 使用方案1或方案3
   - 如果原始特征太小: 使用方案4

4. **实施修复后重新测试**

## 验证修复效果

修复后,期望看到:
- ✅ 空间分量范数 > 0.01 (而不是当前的 ~2e-4)
- ✅ 不同点之间的空间分量有明显差异
- ✅ Lorentz约束 <x,x> ≈ -1 依然满足
- ✅ 点之间的Minkowski内积不全是-1
- ✅ cos_angle的分布不是全都在-1附近

## 下一步

选择一个方案,修改代码,然后:
1. 重新训练模型
2. 运行 `verify_origin_collapse.py` 检查空间分量
3. 运行 `check_embeddings.py` 检查整体嵌入质量
4. 检查STS等下游任务的性能是否改善
