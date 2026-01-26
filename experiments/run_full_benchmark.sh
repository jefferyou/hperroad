#!/bin/bash

# HRNR_Hyperbolic 完整Benchmark脚本 - 支持断点续传
# 自动运行4个城市的完整实验，可以从中断的地方继续

set +e  # 不要立即退出，允许继续

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置参数
DATASET_MAP=("sanfrancisco:sf" "porto:po" "chengdu:cd" "beijing:bj")

# 训练参数
MAX_EPOCH=100
TASK_EPOCH_TTE=100
TASK_EPOCH_STS=50

# GPU配置
TRAIN_GPUS="3,4,5,6,7"
EVAL_GPUS="3,4,5,6,7"

# 结果目录和状态文件
if [ -f ".benchmark_state" ]; then
    # 恢复模式
    source .benchmark_state
    RESUME_MODE=true
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     RESUMING FROM PREVIOUS RUN                            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Found existing benchmark: ${RESULTS_DIR}${NC}"
    echo ""
else
    # 新运行
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="results/benchmark_${TIMESTAMP}"
    RESUME_MODE=false
    mkdir -p "$RESULTS_DIR"

    # 创建状态文件
    echo "RESULTS_DIR=\"${RESULTS_DIR}\"" > .benchmark_state

    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                  HRNR_Hyperbolic Full Benchmark Suite                     ║${NC}"
    echo -e "${BLUE}║                    Multi-City, Multi-GPU Accelerated                      ║${NC}"
    echo -e "${BLUE}║                  WITH RESUME SUPPORT (Auto-recovery)                      ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # 应用所有优化（仅在首次运行）
    echo -e "${YELLOW}[STEP 0]${NC} Applying All Optimizations..."
    python apply_all_optimizations.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All optimizations applied"
    else
        echo -e "${RED}✗${NC} Failed to apply optimizations"
        exit 1
    fi
    echo ""
fi

# 状态检查函数
task_completed() {
    local city=$1
    local task=$2
    local state_file=".benchmark_state"

    if grep -q "${city}_${task}=completed" "$state_file" 2>/dev/null; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 标记任务完成
mark_completed() {
    local city=$1
    local task=$2
    echo "${city}_${task}=completed" >> .benchmark_state
}

# 保存实验ID
save_exp_id() {
    local city=$1
    local exp_id=$2
    echo "${city}_exp_id=\"${exp_id}\"" >> .benchmark_state
}

# 获取保存的实验ID
get_saved_exp_id() {
    local city=$1
    local state_file=".benchmark_state"

    if [ -f "$state_file" ]; then
        source "$state_file"
        local var_name="${city}_exp_id"
        echo "${!var_name}"
    fi
}

echo -e "${BLUE}Results directory: ${RESULTS_DIR}${NC}"
echo -e "${CYAN}Progress file: .benchmark_state${NC}"
echo ""

# 转换GPU ID
IFS=',' read -ra GPU_ARRAY <<< "$TRAIN_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
LOGICAL_GPU_IDS=$(seq -s ' ' 0 $((NUM_GPUS-1)))

IFS=',' read -ra EVAL_GPU_ARRAY <<< "$EVAL_GPUS"
NUM_EVAL_GPUS=${#EVAL_GPU_ARRAY[@]}
EVAL_LOGICAL_IDS=$(seq -s ' ' 0 $((NUM_EVAL_GPUS-1)))

# 处理每个城市
for city_pair in "${DATASET_MAP[@]}"; do
    IFS=':' read -r CITY DATASET <<< "$city_pair"
    CITY_UPPER=$(echo "$CITY" | tr '[:lower:]' '[:upper:]')

    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing City: ${CITY_UPPER}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ========== Step 1: 预训练 ==========
    if task_completed "$CITY" "training"; then
        echo -e "${CYAN}[SKIP]${NC} Training already completed for $CITY_UPPER"
        EXP_ID=$(get_saved_exp_id "$CITY")
        echo "  Experiment ID: $EXP_ID (from previous run)"
    else
        echo -e "${YELLOW}[STEP 1/${CITY_UPPER}]${NC} Upstream Training (Pretraining with Multi-GPU)"
        echo "  Dataset: $DATASET"
        echo "  Epochs: $MAX_EPOCH"
        echo "  GPUs: $TRAIN_GPUS (DataParallel)"
        echo ""

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
            mark_completed "$CITY" "training"

            # 获取并保存实验ID
            EXP_ID=$(ls -t ../VecCity-main/veccity/cache/ | grep "^hrnr_hyp_${DATASET}" | head -1)
            save_exp_id "$CITY" "$EXP_ID"
            echo "  Experiment ID: $EXP_ID"
        else
            echo -e "${RED}✗${NC} Training failed for $CITY_UPPER"
            echo "  Check log: ${RESULTS_DIR}/${CITY}_training.log"
            echo -e "${YELLOW}  You can resume later by running: bash run_full_benchmark.sh${NC}"
            continue  # 继续下一个城市
        fi
    fi
    echo ""

    # ========== Step 2: 下游任务 ==========
    echo -e "${YELLOW}[STEP 2/${CITY_UPPER}]${NC} Downstream Tasks Evaluation"
    echo ""

    # TSI任务
    if task_completed "$CITY" "tsi"; then
        echo -e "  ${CYAN}[SKIP]${NC} TSI already completed"
    else
        echo "  [2.1] Task: TSI (Speed Inference)"
        CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
            --exp_id "$EXP_ID" \
            --task tsi \
            --task_epoch 10 \
            --gpu_id 0 \
            > "${RESULTS_DIR}/${CITY}_tsi.log" 2>&1

        if [ $? -eq 0 ]; then
            echo -e "    ${GREEN}✓${NC} TSI completed"
            mark_completed "$CITY" "tsi"
        else
            echo -e "    ${RED}✗${NC} TSI failed (check ${RESULTS_DIR}/${CITY}_tsi.log)"
            echo -e "    ${YELLOW}Continuing with other tasks...${NC}"
        fi
    fi

    # TTE任务
    if task_completed "$CITY" "tte"; then
        echo -e "  ${CYAN}[SKIP]${NC} TTE already completed"
    else
        echo "  [2.2] Task: TTE (Travel Time Estimation) - ${TASK_EPOCH_TTE} epochs"
        CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
            --exp_id "$EXP_ID" \
            --task tte \
            --task_epoch "$TASK_EPOCH_TTE" \
            --gpu_id 0 \
            --eval_gpu_ids $EVAL_LOGICAL_IDS \
            > "${RESULTS_DIR}/${CITY}_tte.log" 2>&1

        if [ $? -eq 0 ]; then
            echo -e "    ${GREEN}✓${NC} TTE completed"
            mark_completed "$CITY" "tte"
        else
            echo -e "    ${RED}✗${NC} TTE failed (check ${RESULTS_DIR}/${CITY}_tte.log)"
            echo -e "    ${YELLOW}Continuing with other tasks...${NC}"
        fi
    fi

    # STS任务
    if task_completed "$CITY" "sts"; then
        echo -e "  ${CYAN}[SKIP]${NC} STS already completed"
    else
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
            mark_completed "$CITY" "sts"
        else
            echo -e "    ${RED}✗${NC} STS failed (check ${RESULTS_DIR}/${CITY}_sts.log)"
            echo -e "    ${YELLOW}Continuing with next city...${NC}"
        fi
    fi

    echo ""

    # 检查这个城市是否全部完成
    if task_completed "$CITY" "training" && \
       task_completed "$CITY" "tsi" && \
       task_completed "$CITY" "tte" && \
       task_completed "$CITY" "sts"; then
        echo -e "${GREEN}✓${NC} All tasks completed for $CITY_UPPER"
    else
        echo -e "${YELLOW}⚠${NC} Some tasks incomplete for $CITY_UPPER (can resume later)"
    fi
    echo ""
done

# 汇总结果
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  BENCHMARK COMPLETED (OR PARTIALLY COMPLETED)                             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Progress tracked in: .benchmark_state"
echo ""

# 统计完成情况
total_tasks=0
completed_tasks=0
for city_pair in "${DATASET_MAP[@]}"; do
    IFS=':' read -r CITY DATASET <<< "$city_pair"
    for task in training tsi tte sts; do
        total_tasks=$((total_tasks + 1))
        if task_completed "$CITY" "$task"; then
            completed_tasks=$((completed_tasks + 1))
        fi
    done
done

echo "Progress: ${completed_tasks}/${total_tasks} tasks completed"
echo ""

if [ $completed_tasks -eq $total_tasks ]; then
    echo -e "${GREEN}✓ Full benchmark completed successfully!${NC}"
    echo ""
    echo "To start a new benchmark, delete the state file:"
    echo "  rm .benchmark_state"
    echo ""
else
    echo -e "${YELLOW}⚠ Benchmark incomplete. To resume from where you left off:${NC}"
    echo "  bash run_full_benchmark.sh"
    echo ""
    echo "To start fresh (discard current progress):"
    echo "  rm .benchmark_state"
    echo "  bash run_full_benchmark.sh"
    echo ""
fi

echo "To view results:"
echo "  cd $RESULTS_DIR"
echo "  ls -lh"
