#!/bin/bash

################################################################################
# HRNR_Hyperbolic模型多城市下游任务实验脚本 (Bash版本)
#
# 实验设置:
# - 5个城市数据集: prt, cd, bj, xa, sf
# - 每个数据集运行5次 (不同seeds: 31, 42, 53, 64, 75)
# - Embedding维度: 128
#
# 使用方法:
#   bash run_hrnr_experiments.sh
#   bash run_hrnr_experiments.sh gpu 1  # 指定设备和exp_id
################################################################################

# 配置参数
TASK="segment"
MODEL="HRNR_Hyperbolic"
OUTPUT_DIM=128
DEVICE=${1:-"gpu"}  # 默认使用gpu
EXP_ID=${2:-"1"}    # 默认实验ID为1

# 数据集列表
DATASETS=("prt" "cd" "bj" "xa" "sf")

# 随机种子列表（5次运行）
SEEDS=(31 42 53 64 75)

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./experiment_results/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# 日志文件
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"

echo "================================================================================"
echo "HRNR_Hyperbolic Multi-City Experiments (Bash Script)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Task: ${TASK}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output Dim: ${OUTPUT_DIM}"
echo "  Device: ${DEVICE}"
echo "  Exp ID: ${EXP_ID}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# 初始化日志
cat > "${LOG_FILE}" << EOF
HRNR_Hyperbolic Experiments Log
Started at: $(date)

Configuration:
  Model: ${MODEL}
  Task: ${TASK}
  Datasets: ${DATASETS[*]}
  Seeds: ${SEEDS[*]}
  Output Dim: ${OUTPUT_DIM}
  Device: ${DEVICE}
  Exp ID: ${EXP_ID}

===============================================================================
EOF

# 计数器
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
CURRENT_EXPERIMENT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 遍历所有数据集和种子
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

        echo ""
        echo "================================================================================"
        echo "[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Running Experiment"
        echo "================================================================================"
        echo "  Dataset: ${DATASET}"
        echo "  Seed: ${SEED}"
        echo "  Started at: $(date)"
        echo ""

        # 输出文件
        OUTPUT_FILE="${OUTPUT_DIR}/result_${DATASET}_seed${SEED}.log"

        # 记录到日志
        echo "" >> "${LOG_FILE}"
        echo "--------------------------------------------------------------------------------" >> "${LOG_FILE}"
        echo "[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Dataset: ${DATASET}, Seed: ${SEED}" >> "${LOG_FILE}"
        echo "Started at: $(date)" >> "${LOG_FILE}"

        # 运行实验
        python run_downstream_only.py \
            --task "${TASK}" \
            --model "${MODEL}" \
            --dataset "${DATASET}" \
            --seed "${SEED}" \
            --output_dim "${OUTPUT_DIM}" \
            --device "${DEVICE}" \
            --exp_id "${EXP_ID}" \
            > "${OUTPUT_FILE}" 2>&1

        # 检查返回状态
        if [ $? -eq 0 ]; then
            echo "  ✓ Status: SUCCESS"
            echo "  ✓ Output saved to: ${OUTPUT_FILE}"
            echo "Status: SUCCESS" >> "${LOG_FILE}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ Status: FAILED"
            echo "  ✗ Check log: ${OUTPUT_FILE}"
            echo "Status: FAILED" >> "${LOG_FILE}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        echo "Completed at: $(date)" >> "${LOG_FILE}"
        echo ""
    done
done

# 生成摘要
echo ""
echo "================================================================================"
echo "Experiment Summary"
echo "================================================================================"
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# 记录摘要到日志
cat >> "${LOG_FILE}" << EOF

===============================================================================
Experiment Summary
===============================================================================
Total experiments: ${TOTAL_EXPERIMENTS}
Successful: ${SUCCESS_COUNT}
Failed: ${FAIL_COUNT}
Completed at: $(date)
===============================================================================
EOF

echo "All experiments completed!"
echo "To analyze results, run: python analyze_results.py ${OUTPUT_DIR}"
