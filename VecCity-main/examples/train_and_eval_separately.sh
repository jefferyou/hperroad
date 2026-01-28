#!/bin/bash
#
# 示例脚本：演示如何分离训练和评估
#
# 使用方法：
#   bash examples/train_and_eval_separately.sh
#

set -e  # 遇到错误立即退出

echo "=========================================="
echo "示例：分离训练和下游任务评估"
echo "=========================================="
echo ""

# 配置参数
TASK="segment"
MODEL="HRNR_Hyperbolic"
DATASET="bj_roadmap_edge"
DEVICE="cpu"  # 或 "cuda"
EXP_ID="demo_12345"

echo "配置参数："
echo "  任务: $TASK"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  设备: $DEVICE"
echo "  实验ID: $EXP_ID"
echo ""

# ========================================
# 步骤1: 仅训练模型
# ========================================
echo "=========================================="
echo "步骤1: 训练模型（生成embeddings）"
echo "=========================================="
echo ""

python run_training_only.py \
    --task $TASK \
    --model $MODEL \
    --dataset $DATASET \
    --device $DEVICE \
    --exp_id $EXP_ID \
    --saved_model

echo ""
echo "✅ 训练完成！Embeddings已保存。"
echo ""
echo "查看生成的embeddings："
echo "  ls ./veccity/cache/$EXP_ID/evaluate_cache/"
echo ""

# ========================================
# 步骤2: 运行下游任务评估
# ========================================
echo "=========================================="
echo "步骤2: 运行下游任务评估"
echo "=========================================="
echo ""

python run_downstream_only.py \
    --task $TASK \
    --model $MODEL \
    --dataset $DATASET \
    --device $DEVICE \
    --exp_id $EXP_ID \
    --evaluate_task speed_inference travel_time_estimation

echo ""
echo "✅ 下游任务评估完成！"
echo ""
echo "查看评估结果："
echo "  cat ./raw_data/new/evaluate_cache/${EXP_ID}_evaluate_${EXP_ID}_${MODEL}_${DATASET}.csv"
echo ""

# ========================================
# 总结
# ========================================
echo "=========================================="
echo "总结"
echo "=========================================="
echo ""
echo "优势："
echo "  1. ✅ 训练和评估分离，方便调试"
echo "  2. ✅ 可以多次运行评估而不重新训练"
echo "  3. ✅ 批量训练后统一评估"
echo "  4. ✅ 节省计算资源和时间"
echo ""
echo "下一步："
echo "  - 修改下游任务代码后，重新运行步骤2"
echo "  - 或训练其他模型，使用相同的评估脚本"
echo ""
echo "=========================================="
