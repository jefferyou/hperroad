# HRNR with Hyperbolic Embeddings

## 概述

本项目对VecCity中的HRNR模型进行了改进，借鉴HyCoCLIP的思路，引入了Lorentz双曲空间表示学习，以更好地建模道路网络的层次结构。

## 主要改进

### 1. 双曲空间嵌入 (Hyperbolic Embeddings)

- **Lorentz模型**: 使用Lorentz双曲空间模型表示道路网络
  - 时间分量 + 空间分量：`[x₀, x₁, ..., xₐ]`
  - 满足约束：`-x₀² + ||x₁:ₐ||² = -1`

- **双曲嵌入层**: `HyperbolicEmbedding`
  - 将欧氏特征映射到双曲空间
  - 保持层次信息的几何结构

### 2. 组合蕴含学习 (Compositional Entailment Learning)

基于**蕴含锥 (Entailment Cone)** 的层次关系建模：

```
蕴含锥特性：
- 靠近原点的点具有更宽的孔径角度
- 可以蕴含更多的子概念
- 父概念蕴含子概念 ⟺ 子概念在父概念的蕴含锥内
```

**三类蕴含关系**:

1. **Region → Locality**: 区域蕴含其包含的所有局部区域
2. **Locality → Segment**: 局部区域蕴含其包含的所有路段
3. **Segment ↔ Segment**: 拓扑连接的路段互相蕴含

**蕴含损失函数**:
```
L_CE = Σ ReLU(-score(parent, child))
score = θ_parent - angle(parent, child)
```

### 3. 层次对比学习 (Hierarchical Contrastive Learning)

使用Lorentz距离作为相似度度量：

- **Segment层对比**:
  - 正对：相邻路段
  - 负对：非邻路段

- **跨层对比**:
  - 正对：Segment与其所属Locality
  - 负对：Segment与其他Locality

**对比损失 (InfoNCE)**:
```
L_CC = -log(exp(sim(anchor, pos)/τ) / Σ exp(sim(anchor, neg_i)/τ))
sim = -distance_lorentz
```

### 4. 双曲空间消息传递

在双曲空间中进行层次化消息传递：

```
F2F: Region内部传播 (Function-to-Function)
  ↓
F2C: Region → Locality (Function-to-Cluster)
  ↓
C2C: Locality内部传播 (Cluster-to-Cluster)
  ↓
C2N: Locality → Segment (Cluster-to-Node)
  ↓
N2N: Segment内部传播 (Node-to-Node)
```

**核心操作**:
- 指数映射 (Exponential Map): 从切空间映射到流形
- 对数映射 (Logarithmic Map): 从流形映射到切空间
- 双曲图卷积: `HyperbolicGraphConv`

### 5. 总损失函数

```
L_total = L_struct + λ₁ · L_CE + λ₂ · L_CC
```

其中：
- `L_struct`: 结构重构损失（分类任务损失）
- `L_CE`: 蕴含损失
- `L_CC`: 对比损失
- `λ₁`, `λ₂`: 平衡超参数

## 文件结构

```
VecCity-main/veccity/upstream/road_representation/
├── hyperbolic_utils.py         # 双曲空间工具模块
│   ├── LorentzManifold        # Lorentz流形操作
│   ├── HyperbolicEmbedding    # 双曲嵌入层
│   ├── EntailmentCone         # 蕴含锥
│   └── HyperbolicGraphConv    # 双曲图卷积
│
├── HRNR_Hyperbolic.py          # 改进的HRNR模型
│   ├── HRNR_Hyperbolic              # 主模型类
│   ├── HyperbolicGraphEncoderTL     # 双曲图编码器
│   └── HyperbolicGraphEncoderTLCore # 核心编码层
│
└── __init__.py                 # 模块导出（已更新）

VecCity-main/veccity/config/model/segment/
└── HRNR_Hyperbolic.json        # 配置文件
```

## 配置参数

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hyperbolic_dim` | 224 | 双曲空间维度（实际维度为 d+1） |
| `lambda_ce` | 0.1 | 蕴含损失权重 |
| `lambda_cc` | 0.1 | 对比损失权重 |
| `temperature` | 0.07 | 对比学习温度参数 |
| `curvature` | 1.0 | 双曲空间曲率 |
| `max_epoch` | 100 | 最大训练轮数 |

### 原有参数

保持与原HRNR模型一致：

- `hidden_dims`: 隐层维度（224）
- `struct_cmt_num`: 结构聚类数（300，对应Locality数量）
- `fnc_cmt_num`: 功能聚类数（30，对应Region数量）
- `alpha`, `dropout`: GAT参数
- `lp_learning_rate`: 学习率

## 使用方法

### 1. 数据准备

确保数据集包含以下文件：
- `adj_mx`: 邻接矩阵
- `lane_feature`, `type_feature`, `length_feature`, `node_feature`: 节点特征
- `struct_assign`: Segment到Locality的分配矩阵（谱聚类生成）
- `fnc_assign`: Locality到Region的分配矩阵（轨迹数据增强）

### 2. 训练模型

```python
from veccity.upstream.road_representation import HRNR_Hyperbolic

# 加载配置
config = {...}  # 从HRNR_Hyperbolic.json加载

# 创建模型
model = HRNR_Hyperbolic(config, data_feature)

# 训练
model.run(train_dataloader, eval_dataloader)
```

### 3. 评估

模型支持以下下游任务：
- **STS**: Similarity Search（相似度搜索）
- **TTE**: Travel Time Estimation（行程时间估计）
- **TSI**: Traffic Speed Inference（速度推断）

## 技术细节

### Lorentz双曲空间

**Lorentz内积**:
```
⟨x, y⟩ = -x₀y₀ + Σᵢ₌₁ᵈ xᵢyᵢ
```

**Lorentz距离**:
```
d(x, y) = arcosh(-⟨x, y⟩)
```

**投影到双曲空间**:
```
给定欧氏向量 v ∈ ℝᵈ
x₀ = √(1 + ||v||²)
h = [x₀, v₁, ..., vₐ]
```

### 蕴含锥角度

```
半孔径角度: θ(x) = 2 · arcsin(1 / cosh(d(x, origin)))
```

特性：距离原点越近，θ越大，可蕴含的概念越多

### 双曲空间聚合

使用指数/对数映射在双曲空间中进行加权平均：

```python
# 映射到切空间
tangent_vecs = [log_map(ref, point_i) * weight_i]

# 切空间中求平均
avg_tangent = Σ tangent_vecs

# 映射回双曲空间
result = exp_map(ref, avg_tangent)
```

## 理论优势

1. **层次建模能力**
   - 双曲空间天然适合表示树状/层次结构
   - 指数增长的体积容纳更多层次信息

2. **蕴含关系建模**
   - 通过蕴含锥几何地表示"通用-具体"关系
   - Region → Locality → Segment 的层次结构自然编码

3. **对比学习增强**
   - Lorentz距离作为语义相似度
   - 层内和跨层对比学习提升表示质量

4. **消息传递效率**
   - 自顶向下的层次化更新
   - 充分利用双曲几何的优势

## 实验建议

### 超参数调优

1. **λ₁ (蕴含损失权重)**
   - 范围：[0.05, 0.2]
   - 过大可能导致过度约束

2. **λ₂ (对比损失权重)**
   - 范围：[0.05, 0.2]
   - 与任务相关性调整

3. **temperature (温度)**
   - 范围：[0.05, 0.1]
   - 影响对比学习的难度

4. **hyperbolic_dim (双曲维度)**
   - 建议与hidden_dims一致
   - 过小可能损失信息

### 训练技巧

1. **学习率调度**
   - 建议使用warm-up策略
   - 可尝试cosine annealing

2. **批次大小**
   - 双曲操作计算密集，建议适当减小batch_size
   - 如原来64，可尝试32

3. **早停策略**
   - 同时监控AUC和F1
   - patience=50（已设置）

## 对比原HRNR的改进

| 方面 | 原HRNR | HRNR_Hyperbolic |
|------|--------|-----------------|
| **表示空间** | 欧氏空间 | Lorentz双曲空间 |
| **层次建模** | 谱聚类 + 可学习分配 | 双曲几何 + 蕴含锥 |
| **消息传递** | 欧氏GCN/GAT | 双曲图卷积 |
| **损失函数** | 交叉熵 | 交叉熵 + 蕴含 + 对比 |
| **几何结构** | 平坦空间 | 负曲率空间 |
| **层次关系** | 隐式学习 | 显式约束（蕴含锥） |

## 引用

本实现借鉴了以下工作的思路：

1. **HyCoCLIP**: Hyperbolic Compositional Learning (蕴含锥、对比学习)
2. **HRNR**: Hierarchical Road Network Representation
3. **Lorentz Model**: 双曲空间表示学习

## 注意事项

1. **数值稳定性**
   - 双曲操作中使用了eps=1e-7防止数值问题
   - 建议使用float32精度

2. **计算开销**
   - 双曲操作比欧氏操作更耗时
   - 采样策略减少计算量（蕴含损失、对比损失中已实现）

3. **GPU加速**
   - 建议使用GPU训练
   - 设置 `"gpu": true` 和 `"device": "cuda"`

4. **内存占用**
   - 三层嵌入（Segment、Locality、Region）同时存储
   - 大规模网络需注意内存

## 未来改进方向

1. **多曲率学习**: 为不同层次学习不同曲率
2. **混合双曲空间**: 结合Poincaré球模型
3. **动态层次**: 根据数据自适应调整层次数量
4. **轨迹建模**: 在双曲空间中建模轨迹序列
5. **时空融合**: 结合时间信息的双曲时空嵌入

## 许可

遵循VecCity原有许可协议。

---

**创建日期**: 2025-12-27
**版本**: v1.0
**作者**: Claude Code
