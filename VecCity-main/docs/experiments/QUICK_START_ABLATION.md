# 🚀 消融实验快速开始

## 一分钟快速上手

```bash
cd /home/user/hperroad/experiments

# 1. 测试环境
python test_ablation_setup.py

# 2. 运行消融实验
./run_ablation.sh

# 3. 查看结果
cat results/ablation/ablation_comparison.md
```

## 📋 5个消融配置

| 配置 | 说明 | 双曲空间 | 蕴含损失 | 对比损失 |
|------|------|---------|---------|---------|
| **baseline_hrnr** | 原始HRNR | ❌ | ❌ | ❌ |
| **full_model** | 完整版 | ✅ | ✅ | ✅ |
| **no_entailment** | 无蕴含 | ✅ | ❌ | ✅ |
| **no_contrastive** | 无对比 | ✅ | ✅ | ❌ |
| **no_auxiliary** | 仅双曲 | ✅ | ❌ | ❌ |

## 💡 常用命令

```bash
# 基本运行
./run_ablation.sh

# 使用特定GPU
./run_ablation.sh --gpu-id 1

# 从中断处继续
./run_ablation.sh --resume

# 跳过baseline
./run_ablation.sh --skip-baseline

# 多个种子
for seed in 0 1 2; do
    ./run_ablation.sh --seed $seed
done
```

## 📊 查看结果

```bash
# Markdown报告（推荐）
cat results/ablation/ablation_comparison.md

# CSV表格
cat results/ablation/ablation_comparison.csv

# 组件贡献分析
cat results/ablation/component_analysis.txt

# 查看进度
cat results/ablation/ablation_progress_xa_seed0.json
```

## 🔧 断点续传

**实验自动保存进度！**

如果实验中断（网络断开、服务器重启等）：

```bash
# 直接继续，系统会跳过已完成的实验
./run_ablation.sh --resume
```

进度文件位置：
```
results/ablation/ablation_progress_{dataset}_seed{seed}.json
```

## 📈 预期结果

```
性能排序（理论）：
Full Model > w/o Contrastive ≈ w/o Entailment > w/o Auxiliary > Baseline

改进幅度（估计）：
- 双曲空间本身：5-10%
- 蕴含损失额外贡献：2-5%
- 对比损失额外贡献：2-5%
- 总体提升：10-20%
```

## 🐛 遇到问题？

### Q: 环境测试失败
```bash
python test_ablation_setup.py  # 查看详细错误
```

### Q: GPU内存不足
```bash
./run_ablation.sh --no-gpu  # 使用CPU
```

### Q: 实验中断了
```bash
./run_ablation.sh --resume  # 继续运行
```

### Q: 某个配置失败
```bash
# 查看错误
cat results/ablation/ablation_progress_xa_seed0.json | grep -A 5 "failed"

# 手动编辑进度文件，移除失败标记
# 然后 --resume 重试
```

## 📚 完整文档

- **[README_ABLATION.md](README_ABLATION.md)** - 详细说明和最佳实践
- **[ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)** - 完整使用指南

## ⚡ 快速检查清单

实验前：
- [ ] 运行 `python test_ablation_setup.py`
- [ ] 检查GPU: `nvidia-smi`
- [ ] 检查磁盘空间: `df -h`

实验中：
- [ ] 查看实时日志
- [ ] 监控GPU使用: `watch -n 1 nvidia-smi`

实验后：
- [ ] 查看结果文件
- [ ] 分析组件贡献
- [ ] 保存实验日志

---

**需要帮助？**
1. 查看 [README_ABLATION.md](README_ABLATION.md)
2. 查看 [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)
3. 运行 `./run_ablation.sh --help`

**立即开始：**
```bash
./run_ablation.sh
```
