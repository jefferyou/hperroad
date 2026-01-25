#!/bin/bash
#
# HRNR_Hyperbolic完整Benchmark执行脚本
# 支持多城市、多GPU加速、完整下游任务评估
#
# 用法:
#   bash run_full_benchmark.sh
#
# 或者自定义参数:
#   bash run_full_benchmark.sh --cities "xian beijing" --max_epoch 100 --task_epoch 100

set -e  # 遇到错误立即退出

# 默认参数（匹配HRNR论文）
CITIES="xian beijing chengdu sanfrancisco"
MAX_EPOCH=100           # 预训练轮数
TASK_EPOCH_TTE=100      # TTE任务轮数（HRNR默认）
TASK_EPOCH_STS=50       # STS任务轮数（HRNR默认）
TRAIN_GPUS="3,4,5,6,7"  # 预训练使用的GPU（多GPU并行）
EVAL_GPUS="3,4,5,6,7"   # 评估使用的GPU（多GPU并行）
SEED=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  HRNR_Hyperbolic Full Benchmark Suite                     ║${NC}"
echo -e "${BLUE}║                    Multi-City, Multi-GPU Accelerated                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 步骤0: 应用性能优化补丁
echo -e "${YELLOW}[STEP 0]${NC} Applying Performance Optimizations..."
python apply_downstream_optimizations.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Downstream task optimizations applied"
else
    echo -e "${RED}✗${NC} Failed to apply downstream optimizations"
    exit 1
fi

# 应用预训练多GPU优化
echo -e "${YELLOW}[STEP 0.1]${NC} Enabling Multi-GPU for Pretraining..."
python enable_pretraining_multigpu.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Pretraining multi-GPU enabled"
else
    echo -e "${RED}✗${NC} Failed to enable pretraining multi-GPU"
    exit 1
fi
echo ""

# 创建结果目录
mkdir -p results
BENCHMARK_DIR="results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_DIR"

echo -e "${BLUE}Results will be saved to: $BENCHMARK_DIR${NC}"
echo ""

# 主循环：处理每个城市
for CITY in $CITIES; do
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing City: ${CITY^^}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"

    # 确定数据集代码
    case $CITY in
        xian)        DATASET="xa" ;;
        beijing)     DATASET="bj" ;;
        chengdu)     DATASET="cd" ;;
        sanfrancisco) DATASET="sf" ;;
        *)
            echo -e "${RED}✗${NC} Unknown city: $CITY"
            continue
            ;;
    esac

    START_TIME=$(date +%s)

    # ========================================================================
    # 步骤1: 预训练（Upstream Training with Multi-GPU）
    # ========================================================================
    echo ""
    echo -e "${YELLOW}[STEP 1/${CITY^^}]${NC} Upstream Training (Pretraining with Multi-GPU)"
    echo -e "  Dataset: $DATASET"
    echo -e "  Epochs: $MAX_EPOCH"
    echo -e "  GPUs: $TRAIN_GPUS (DataParallel)"
    echo ""

    # 转换GPU列表为数组格式
    IFS=',' read -ra GPU_ARRAY <<< "$TRAIN_GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}

    # 创建逻辑GPU ID列表（CUDA_VISIBLE_DEVICES会重新映射为0,1,2...）
    LOGICAL_GPU_IDS=$(seq -s ' ' 0 $((NUM_GPUS-1)))

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python run_training_only.py \
        --dataset $DATASET \
        --seed $SEED \
        --max_epoch $MAX_EPOCH \
        --gpu True \
        --gpu_id 0 \
        --train_gpu_ids $LOGICAL_GPU_IDS  # 逻辑GPU IDs: 0,1,2,3,4

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${NC} Training failed for $CITY"
        echo "$CITY: TRAINING_FAILED" >> "$BENCHMARK_DIR/status.txt"
        continue
    fi

    echo -e "${GREEN}✓${NC} Training completed for $CITY"

    # 自动查找最新的exp_id
    EXP_ID=$(ls -t ../VecCity-main/veccity/cache/ | grep "hrnr_hyp_${DATASET}_s${SEED}" | head -1)

    if [ -z "$EXP_ID" ]; then
        echo -e "${RED}✗${NC} Could not find experiment ID for $CITY"
        echo "$CITY: EXP_ID_NOT_FOUND" >> "$BENCHMARK_DIR/status.txt"
        continue
    fi

    echo -e "${GREEN}✓${NC} Detected Experiment ID: $EXP_ID"
    echo "$CITY: $EXP_ID" >> "$BENCHMARK_DIR/exp_ids.txt"

    # ========================================================================
    # 步骤2: 下游任务评估（Downstream Tasks）
    # ========================================================================
    echo ""
    echo -e "${YELLOW}[STEP 2/${CITY^^}]${NC} Downstream Tasks Evaluation"
    echo -e "  Tasks: TSI + TTE(${TASK_EPOCH_TTE} epochs) + STS(${TASK_EPOCH_STS} epochs)"
    echo -e "  GPUs: $EVAL_GPUS (DataParallel)"
    echo ""

    # 运行所有下游任务
    # 注意：这里手动运行每个任务以使用不同的epoch设置
    # CUDA_VISIBLE_DEVICES会将物理GPU重新映射为逻辑GPU 0,1,2...，所以gpu_id始终为0

    # TSI (Speed Inference) - Ridge regression, 无训练epoch
    echo -e "${BLUE}[2.1] Running TSI...${NC}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id $EXP_ID \
        --task tsi \
        --gpu_id 0 \
        --dataset $DATASET \
        2>&1 | tee "$BENCHMARK_DIR/${CITY}_tsi.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${NC} TSI failed for $CITY"
    else
        echo -e "${GREEN}✓${NC} TSI completed"
    fi

    # TTE (Travel Time Estimation) - LSTM, HRNR默认100 epochs
    echo -e "${BLUE}[2.2] Running TTE (${TASK_EPOCH_TTE} epochs)...${NC}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id $EXP_ID \
        --task tte \
        --task_epoch $TASK_EPOCH_TTE \
        --gpu_id 0 \
        --dataset $DATASET \
        2>&1 | tee "$BENCHMARK_DIR/${CITY}_tte.log"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${NC} TTE failed for $CITY"
    else
        echo -e "${GREEN}✓${NC} TTE completed"
    fi

    # STS (Similarity Search) - Contrastive learning, HRNR默认50 epochs
    echo -e "${BLUE}[2.3] Running STS (${TASK_EPOCH_STS} epochs)...${NC}"
    CUDA_VISIBLE_DEVICES=$EVAL_GPUS python run_evaluation_only.py \
        --exp_id $EXP_ID \
        --task sts \
        --task_epoch $TASK_EPOCH_STS \
        --gpu_id 0 \
        --dataset $DATASET \
        2>&1 | tee "$BENCHMARK_DIR/${CITY}_sts.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All downstream tasks completed for $CITY"
        echo "$CITY: SUCCESS" >> "$BENCHMARK_DIR/status.txt"
    else
        echo -e "${RED}✗${NC} Downstream tasks failed for $CITY"
        echo "$CITY: EVAL_FAILED" >> "$BENCHMARK_DIR/status.txt"
        continue
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    echo ""
    echo -e "${GREEN}✓✓✓${NC} $CITY completed in ${HOURS}h ${MINUTES}m"
    echo "$CITY: ${HOURS}h ${MINUTES}m" >> "$BENCHMARK_DIR/durations.txt"
    echo ""

    # 复制结果文件到benchmark目录
    RESULT_DIR="../VecCity-main/veccity/cache/$EXP_ID/evaluate_cache"
    if [ -d "$RESULT_DIR" ]; then
        cp "$RESULT_DIR"/*.csv "$BENCHMARK_DIR/" 2>/dev/null || true
        cp "$RESULT_DIR"/*.npy "$BENCHMARK_DIR/${CITY}_embedding.npy" 2>/dev/null || true
    fi

    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
done

# ========================================================================
# 生成最终报告
# ========================================================================
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         BENCHMARK COMPLETE                                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
cat "$BENCHMARK_DIR/status.txt" 2>/dev/null || echo "No status file found"
echo ""
echo -e "${YELLOW}Durations:${NC}"
cat "$BENCHMARK_DIR/durations.txt" 2>/dev/null || echo "No duration file found"
echo ""
echo -e "${GREEN}All results saved to:${NC} $BENCHMARK_DIR"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Check results in: $BENCHMARK_DIR/"
echo "  2. Compare with HRNR baseline using Table 4 metrics"
echo "  3. Analyze embeddings and downstream task performance"
echo ""
