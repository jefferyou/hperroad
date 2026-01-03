#!/bin/bash
# HRNR_Hyperbolic 快速启动脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "HRNR_Hyperbolic Quick Start"
echo "=================================="

# 设置默认参数
DATASET=${DATASET:-"xa"}
GPU_ID=${GPU_ID:-0}
MODE=${MODE:-"single"}

# 显示帮助信息
show_help() {
    echo "Usage: ./quick_start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              显示帮助信息"
    echo "  -m, --mode MODE         实验模式: single, multi_seed, ablation, comparison, tune"
    echo "  -d, --dataset DATASET   数据集名称 (默认: xa)"
    echo "  -g, --gpu GPU_ID        GPU编号 (默认: 0)"
    echo "  -s, --seed SEED         随机种子 (默认: 0)"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh --mode single --dataset xa"
    echo "  ./quick_start.sh --mode multi_seed --dataset bj"
    echo "  ./quick_start.sh --mode ablation"
    echo "  ./quick_start.sh --mode tune"
    echo ""
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 创建结果和图片目录
mkdir -p results
mkdir -p figures

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Dataset: $DATASET"
echo "  GPU ID: $GPU_ID"
echo "=================================="

# 根据模式执行不同的实验
case $MODE in
    single)
        echo "Running single experiment..."
        python run_hrnr_hyperbolic.py \
            --mode single \
            --dataset $DATASET \
            --gpu_id $GPU_ID \
            --seed ${SEED:-0}
        ;;

    multi_seed)
        echo "Running multi-seed experiments..."
        python run_hrnr_hyperbolic.py \
            --mode multi_seed \
            --dataset $DATASET \
            --gpu_id $GPU_ID \
            --num_runs 5
        ;;

    ablation)
        echo "Running ablation study..."
        python run_hrnr_hyperbolic.py \
            --mode ablation \
            --dataset $DATASET \
            --gpu_id $GPU_ID
        ;;

    comparison)
        echo "Running model comparison..."
        python run_hrnr_hyperbolic.py \
            --mode comparison \
            --dataset $DATASET \
            --gpu_id $GPU_ID
        ;;

    tune)
        echo "Running hyperparameter tuning..."
        python hyperparameter_tuning.py \
            --method random \
            --max_trials 50 \
            --dataset $DATASET \
            --gpu_id $GPU_ID \
            --metric auc
        ;;

    full)
        echo "Running full experiment pipeline..."
        echo ""
        echo "[1/4] Single experiment..."
        python run_hrnr_hyperbolic.py \
            --mode single \
            --dataset $DATASET \
            --gpu_id $GPU_ID

        echo ""
        echo "[2/4] Multi-seed experiments..."
        python run_hrnr_hyperbolic.py \
            --mode multi_seed \
            --dataset $DATASET \
            --gpu_id $GPU_ID \
            --num_runs 3

        echo ""
        echo "[3/4] Ablation study..."
        python run_hrnr_hyperbolic.py \
            --mode ablation \
            --dataset $DATASET \
            --gpu_id $GPU_ID

        echo ""
        echo "[4/4] Hyperparameter tuning..."
        python hyperparameter_tuning.py \
            --method random \
            --max_trials 30 \
            --dataset $DATASET \
            --gpu_id $GPU_ID
        ;;

    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Valid modes: single, multi_seed, ablation, comparison, tune, full"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Experiment completed!"
echo "Results saved to: ./results/"
echo "Figures saved to: ./figures/"
echo "=================================="
