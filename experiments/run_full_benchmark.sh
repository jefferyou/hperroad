#!/bin/bash

# HRNR_Hyperbolic 完整Benchmark脚本
# 自动运行4个城市的完整实验（预训练 + 3个下游任务）

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
CITIES="xian beijing chengdu sanfrancisco"
DATASET_MAP=("xian:xa" "beijing:bj" "chengdu:cd" "sanfrancisco:sf")

# 训练参数
MAX_EPOCH=100           # 预训练轮数
TASK_EPOCH_TTE=100      # TTE任务轮数（HRNR论文设置）
TASK_EPOCH_STS=50       # STS任务轮数（HRNR论文设置）

# GPU配置
TRAIN_GPUS="3,4,5,6,7"  # 5个GPU用于预训练
EVAL_GPUS="3,4,5,6,7"   # 5个GPU用于评估

# 结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/benchmark_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# 打印标题
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  HRNR_Hyperbolic Full Benchmark Suite                     ║${NC}"
echo -e "${BLUE}║                    Multi-City, Multi-GPU Accelerated                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 应用所有优化
echo -e "${YELLOW}[STEP 0]${NC} Applying All Optimizations..."
python apply_all_optimizations.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All optimizations applied"
else
    echo -e "${RED}✗${NC} Failed to apply optimizations"
    exit 1
fi
echo ""

echo -e "${BLUE}Results will be saved to: ${RESULTS_DIR}${NC}"
echo ""

# 处理每个城市
for city_pair in "${DATASET_MAP[@]}"; do
    IFS=':' read -r CITY DATASET <<< "$city_pair"
    CITY_UPPER=$(echo "$CITY" | tr '[:lower:]' '[:upper:]')

    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing City: ${CITY_UPPER}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Step 1: 预训练（多GPU）
    echo -e "${YELLOW}[STEP 1/${CITY_UPPER}]${NC} Upstream Training (Pretraining with Multi-GPU)"
    echo "  Dataset: $DATASET"
    echo "  Epochs: $MAX_EPOCH"
    echo "  GPUs: $TRAIN_GPUS (DataParallel)"
    echo ""

    # 将TRAIN_GPUS转换为逻辑GPU ID
    IFS=',' read -ra GPU_ARRAY <<< "$TRAIN_GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    LOGICAL_GPU_IDS=$(seq -s ' ' 0 $((NUM_GPUS-1)))

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python run_training_only.py \
        --dataset "$DATASET" \
        --seed 0 \
        --max_epoch "$MAX_EPOCH" \
        --gpu True \
        --gpu_id 0 \
        --train_gpu_ids $LOGICAL_GPU_IDS \
        > "${RESULTS_DIR}/${CITY}_training.log" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Training completed for $CITY_UPPER"
    else
        echo -e "${RED}✗${NC} Training failed for $CITY_UPPER"
        echo "  Check log: ${RESULTS_DIR}/${CITY}_training.log"
        exit 1
    fi
    echo ""

    # 获取实验ID（从最新的cache目录）
    EXP_ID=$(ls -t ../VecCity-main/veccity/cache/ | grep "^hrnr_hyp_${DATASET}" | head -1)
    echo "  Experiment ID: $EXP_ID"
    echo ""

    # Step 2: 下游任务评估
    echo -e "${YELLOW}[STEP 2/${CITY_UPPER}]${NC} Downstream Tasks Evaluation"
    echo ""

    # TSI任务
    echo "  [2.1] Task: TSI (Speed Inference)"
    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id "$EXP_ID" \
        --task tsi \
        --task_epoch 10 \
        --gpu_id 0 \
        > "${RESULTS_DIR}/${CITY}_tsi.log" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✓${NC} TSI completed"
    else
        echo -e "    ${RED}✗${NC} TSI failed (check ${RESULTS_DIR}/${CITY}_tsi.log)"
    fi

    # TTE任务
    echo "  [2.2] Task: TTE (Travel Time Estimation) - ${TASK_EPOCH_TTE} epochs"
    IFS=',' read -ra EVAL_GPU_ARRAY <<< "$EVAL_GPUS"
    NUM_EVAL_GPUS=${#EVAL_GPU_ARRAY[@]}
    EVAL_LOGICAL_IDS=$(seq -s ' ' 0 $((NUM_EVAL_GPUS-1)))

    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id "$EXP_ID" \
        --task tte \
        --task_epoch "$TASK_EPOCH_TTE" \
        --gpu_id 0 \
        --eval_gpu_ids $EVAL_LOGICAL_IDS \
        > "${RESULTS_DIR}/${CITY}_tte.log" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✓${NC} TTE completed"
    else
        echo -e "    ${RED}✗${NC} TTE failed (check ${RESULTS_DIR}/${CITY}_tte.log)"
    fi

    # STS任务
    echo "  [2.3] Task: STS (Similarity Search) - ${TASK_EPOCH_STS} epochs"
    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id "$EXP_ID" \
        --task sts \
        --task_epoch "$TASK_EPOCH_STS" \
        --gpu_id 0 \
        --eval_gpu_ids $EVAL_LOGICAL_IDS \
        > "${RESULTS_DIR}/${CITY}_sts.log" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✓${NC} STS completed"
    else
        echo -e "    ${RED}✗${NC} STS failed (check ${RESULTS_DIR}/${CITY}_sts.log)"
    fi

    echo ""
    echo -e "${GREEN}✓${NC} All tasks completed for $CITY_UPPER"
    echo ""
done

# 汇总结果
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  BENCHMARK COMPLETED                                                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To view results:"
echo "  cd $RESULTS_DIR"
echo "  tail -50 xian_tte.log  # View TTE results for Xi'an"
echo ""
echo -e "${GREEN}✓${NC} Full benchmark completed successfully!"
