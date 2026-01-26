# 断点续传功能说明

## ⭐ 核心功能

**脚本现在支持自动断点续传！** 无论何时中断，重新运行相同命令即可从中断处继续。

## 工作原理

### 1. 自动保存进度

脚本在每个任务完成后，自动保存到 `.benchmark_state` 文件：

```bash
RESULTS_DIR="results/benchmark_20260125_145751"
xian_training=completed
xian_exp_id="hrnr_hyp_xa_s0_20260125_145753"
xian_tsi=completed
xian_tte=completed
xian_sts=completed
beijing_training=completed
beijing_exp_id="hrnr_hyp_bj_s0_20260125_152130"
beijing_tsi=completed
# ... Beijing TTE 在这里中断了
```

### 2. 自动检测和恢复

再次运行时，脚本会：
- ✅ 检测到 `.benchmark_state` 文件
- ✅ 读取已完成的任务
- ✅ 跳过所有已完成的任务
- ✅ 从 Beijing TTE 开始继续运行

### 3. 显示进度

运行结束时显示：
```
Progress: 6/16 tasks completed

⚠ Benchmark incomplete. To resume from where you left off:
  bash run_full_benchmark.sh
```

## 使用场景

### 场景1：网络中断

```bash
# 运行到一半，SSH断线
bash run_full_benchmark.sh
# Xi'an完成，Beijing训练完成，TTE进行到一半...
# [网络断线]

# 重新连接后
bash run_full_benchmark.sh
# 输出：RESUMING FROM PREVIOUS RUN
#      [SKIP] Xi'an training
#      [SKIP] Xi'an tsi
#      [SKIP] Xi'an tte
#      [SKIP] Xi'an sts
#      [SKIP] Beijing training
#      [SKIP] Beijing tsi
#      Starting Beijing tte...  ← 从这里继续
```

### 场景2：GPU错误

```bash
# Beijing训练时GPU出错
bash run_full_benchmark.sh
# Xi'an完成 ✓
# Beijing训练失败 ✗
# 继续Chengdu...

# 修复GPU后重新运行
bash run_full_benchmark.sh
# 自动跳过Xi'an
# 自动跳过Chengdu（如果也完成了）
# 重试Beijing训练
```

### 场景3：手动停止

```bash
# 运行一段时间后用Ctrl+C停止
bash run_full_benchmark.sh
# Xi'an完成
# Beijing部分完成
# ^C 手动中断

# 稍后继续
bash run_full_benchmark.sh
# 从Beijing未完成的部分继续
```

## 管理命令

### 查看当前进度

```bash
cat .benchmark_state
```

### 查看统计信息

运行脚本会显示：
```
Progress: 12/16 tasks completed
```

### 重新开始（丢弃进度）

```bash
rm .benchmark_state
bash run_full_benchmark.sh
```

### 手动标记任务完成

```bash
# 如果确定某任务已完成但脚本不识别
echo "beijing_tte=completed" >> .benchmark_state
```

### 查看结果目录

```bash
# 状态文件中保存了结果目录
source .benchmark_state
echo $RESULTS_DIR
# 输出：results/benchmark_20260125_145751

cd $RESULTS_DIR
ls -lh
```

## 优势

### 1. 节省时间

- Xi'an完成需要 4-5小时
- 如果在Beijing时中断，恢复后自动跳过Xi'an
- 节省 4-5小时重复工作

### 2. 容错能力强

- 网络中断 → 自动恢复 ✓
- 停电 → 自动恢复 ✓
- GPU错误 → 继续其他城市 ✓
- 手动停止 → 稍后继续 ✓

### 3. 灵活性

- 可以随时停止
- 可以选择性重跑某个城市
- 可以手动调整进度
- 不影响已完成的结果

## 技术细节

### 状态文件格式

```bash
RESULTS_DIR="results/benchmark_TIMESTAMP"
<city>_<task>=completed
<city>_exp_id="experiment_id"
```

### 任务标识

每个城市有4个任务：
- `training` - 预训练（100 epochs，~3.5小时）
- `tsi` - 速度推理（<1分钟）
- `tte` - 行程时间估计（100 epochs，~30-60分钟）
- `sts` - 相似性搜索（50 epochs，~10-20分钟）

总共 4城市 × 4任务 = 16个任务

### 状态检查逻辑

```bash
task_completed() {
    local city=$1
    local task=$2
    grep -q "${city}_${task}=completed" .benchmark_state
}

if task_completed "xian" "training"; then
    echo "[SKIP] Training already completed"
    # 跳过
else
    # 执行训练
    mark_completed "xian" "training"
fi
```

## 测试

运行测试脚本验证功能：

```bash
bash test_resume.sh
```

预期输出：
```
==========================================
✓ 断点续传功能测试通过
==========================================
```

## 常见问题

### Q1: 如果.benchmark_state损坏怎么办？

```bash
# 删除并重新开始
rm .benchmark_state
bash run_full_benchmark.sh
```

### Q2: 可以修改GPU配置后继续吗？

可以！编辑 `run_full_benchmark.sh` 中的GPU配置，脚本会使用新配置继续未完成的任务。

### Q3: 如何只重跑某个失败的任务？

```bash
# 1. 从状态文件中删除该任务
sed -i '/beijing_tte=completed/d' .benchmark_state

# 2. 重新运行
bash run_full_benchmark.sh
# 会重新执行Beijing TTE
```

### Q4: 结果保存在哪里？

所有结果保存在状态文件中记录的目录：

```bash
source .benchmark_state
ls -lh $RESULTS_DIR/
```

### Q5: 可以同时运行多个benchmark吗？

不建议。使用不同的目录：

```bash
# 终端1
cd experiments
bash run_full_benchmark.sh

# 终端2（新benchmark）
cd experiments_backup
bash run_full_benchmark.sh
```

## 总结

断点续传功能让长时间运行的benchmark变得：
- ✅ 更可靠（不怕中断）
- ✅ 更灵活（随时停止/继续）
- ✅ 更省时（不重复已完成的工作）
- ✅ 更智能（自动管理进度）

完美适合16-20小时的完整benchmark！🚀
