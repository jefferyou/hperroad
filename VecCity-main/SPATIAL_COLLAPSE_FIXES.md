# 空间分量坍缩问题的修复方案

## 问题诊断

当前状态:
- 空间分量不是严格的0,但非常小 (~1e-4)
- 所有embedding聚集在Lorentz原点 [1, 0, 0, ..., 0] 附近
- 导致所有点之间距离相似,模型无法区分不同的segment

根本原因:
- HyperbolicEmbedding层的线性变换输出向量范数太小
- 根据公式 h = [sqrt(1 + ||x||^2), x],当||x||很小时,所有点都接近原点

## 修复方案

### 方案1: 增加线性层初始化scale (推荐)

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

## 推荐流程

1. **先运行诊断脚本** `diagnose_spatial_collapse.py` 确定问题出在哪一步
2. **如果线性层权重太小**: 使用方案1 (推荐) 或方案3 (快速测试)
3. **如果原始特征太小**: 使用方案4
4. **如果以上都不行**: 考虑方案2 (需要更多改动)

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
