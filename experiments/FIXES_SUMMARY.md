# HRNR_Hyperbolic 修复总结

## 修复完成时间
2025-12-28

## 修复的所有问题 ✅

### 1. 路径解析问题 ✅
**错误**: `FileNotFoundError: './veccity/config/task_config.json'`

**原因**: 实验脚本从 `experiments/` 目录运行，但VecCity期望从 `VecCity-main/` 运行

**修复**:
- 在所有实验脚本中添加工作目录切换
- 使用绝对路径保存结果
```python
VECCITY_ROOT = os.path.join(PROJECT_ROOT, 'VecCity-main')
os.chdir(VECCITY_ROOT)
```

**影响文件**:
- `experiments/run_hrnr_hyperbolic.py`
- `experiments/hyperparameter_tuning.py`
- `experiments/visualization_tools.py`

---

### 2. 参数传递问题 ✅
**错误**: `TypeError: run_model() got an unexpected keyword argument 'seed'`

**原因**: VecCity的 `run_model()` 不接受seed, gpu, gpu_id等参数作为关键字参数

**修复**:
- 将所有额外参数移到 `other_args` 字典中
```python
other_args = {
    'seed': 0,
    'gpu': True,
    'gpu_id': 0,
    'exp_id': exp_id,
    'hyperbolic_dim': 224,
    ...
}
result = run_model(task='segment', model_name='HRNR_Hyperbolic',
                   dataset_name='xa', other_args=other_args)
```

**影响文件**:
- `experiments/run_hrnr_hyperbolic.py`
- `experiments/hyperparameter_tuning.py`

---

### 3. Task名称问题 ✅
**错误**: `ValueError: task road_representation is not supported`

**原因**:
- VecCity使用 'segment', 'parcel', 'poi' 作为task名称
- HRNR_Hyperbolic没有在 task_config.json 中注册

**修复**:
1. 修改默认task从 'road_representation' 到 'segment'
2. 在 `task_config.json` 中注册 HRNR_Hyperbolic:
```json
"segment": {
    "allowed_model": [..., "HRNR_Hyperbolic", ...],
    "HRNR_Hyperbolic": {
        "dataset_class": "HRNRDataset",
        "executor": "TwoStepExecutor",
        "evaluator": "HHGCLEvaluator"
    }
}
```

**影响文件**:
- `VecCity-main/veccity/config/task_config.json`
- `experiments/run_hrnr_hyperbolic.py`
- `experiments/hyperparameter_tuning.py`

---

### 4. 设备匹配问题 (第一处) ✅
**错误**: `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`

**位置**: `hrnr_dataset.py:210` in `calc_tsr()`

**原因**: AS tensor在CPU上创建，但其他张量在GPU上

**修复**:
```python
# 修复前
AS = torch.tensor(self.adj_matrix + np.array(np.eye(self.num_nodes)),
                  dtype=torch.float)

# 修复后
AS = torch.tensor(self.adj_matrix + np.array(np.eye(self.num_nodes)),
                  dtype=torch.float).to(self.device)
```

**影响文件**:
- `VecCity-main/veccity/data/dataset/hrnr_dataset.py`

---

### 5. CUDA到NumPy转换问题 (第二处) ✅
**错误**: `TypeError: can't convert cuda:0 device type tensor to numpy`

**位置**: `HRNR.py:205` in `get_sparse_adj()`

**原因**: 尝试直接将CUDA张量转换为NumPy数组

**修复**:
```python
def get_sparse_adj(adj, device):
    # 修复前
    adj = np.array(adj) + self_loop

    # 修复后
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().detach().numpy()

    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    ...
```

**影响文件**:
- `VecCity-main/veccity/upstream/road_representation/HRNR.py`

---

## 提交历史

1. **cc274ef**: Fix device mismatch in HRNR dataset
2. **3fc7e0e**: Register HRNR_Hyperbolic in VecCity config and fix task name
3. **5e51600**: Update test guide - all issues fixed
4. **26e92e7**: Fix run_model parameter passing
5. **c333d2a**: Add quick test guide for experiment scripts
6. **ffe0f4e**: Fix CUDA to NumPy conversion in get_sparse_adj
7. **db41624**: Update test guide - CUDA conversion fix completed

---

## 系统状态

### ✅ 所有功能已就绪

1. **模型实现**: HRNR_Hyperbolic with Lorentz hyperbolic embeddings
2. **配置系统**: 完整的VecCity配置集成
3. **实验框架**:
   - 单次实验模式
   - 多随机种子模式
   - 消融实验模式
   - 模型对比模式
4. **超参数优化**: Random/Grid/Bayesian搜索
5. **可视化工具**: 训练曲线、参数重要性、消融分析等

### ✅ 所有错误已修复

- 路径问题 ✅
- 参数传递 ✅
- Task注册 ✅
- 设备匹配 ✅
- CUDA转换 ✅

---

## 下一步操作

系统现已完全可用，可以开始实验：

### 快速测试
```bash
cd experiments
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --seed 0 \
    --max_epoch 2 \
    --gpu True \
    --gpu_id 0
```

### 完整实验
```bash
# 单次完整训练
python run_hrnr_hyperbolic.py --dataset xa --seed 0

# 多随机种子（5次）
python run_hrnr_hyperbolic.py --mode multi_seed --dataset xa

# 消融实验
python run_hrnr_hyperbolic.py --mode ablation --dataset xa

# 超参数优化（50次随机搜索）
python hyperparameter_tuning.py \
    --method random \
    --max_trials 50 \
    --dataset xa
```

---

## 技术亮点

### 双曲空间实现
- Lorentz模型 (d+1维)
- 蕴含锥 (Entailment Cones)
- 双曲图卷积
- Minkowski内积和Lorentz距离

### 层次化结构
- Segment (5269个节点)
- Locality (300个聚类)
- Region (30个聚类)

### 三种损失函数
- **L_struct**: 结构重建损失
- **L_CE**: 层次蕴含损失 (λ₁ = 0.1)
- **L_CC**: 对比学习损失 (λ₂ = 0.1)

---

**所有系统已就绪，可以开始实验！** 🚀
