#!/bin/bash
# 消融实验启动脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
DATASET="xa"
SEED=0
GPU=true
GPU_ID=0
RESUME=false
SKIP_BASELINE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --no-gpu)
            GPU=false
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET       Dataset name (default: xa)"
            echo "  --seed SEED            Random seed (default: 0)"
            echo "  --gpu-id GPU_ID        GPU device ID (default: 0)"
            echo "  --no-gpu               Disable GPU (default: enabled)"
            echo "  --resume               Resume from previous progress"
            echo "  --skip-baseline        Skip baseline HRNR experiment"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset xa --seed 0"
            echo "  $0 --dataset xa --resume"
            echo "  $0 --dataset xa --gpu-id 1"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 打印配置
echo ""
echo "=========================================="
echo "  ABLATION STUDY CONFIGURATION"
echo "=========================================="
echo "Dataset:        $DATASET"
echo "Seed:           $SEED"
echo "GPU:            $GPU"
if [ "$GPU" = true ]; then
    echo "GPU ID:         $GPU_ID"
fi
echo "Resume:         $RESUME"
echo "Skip Baseline:  $SKIP_BASELINE"
echo "=========================================="
echo ""

# 检查GPU可用性
if [ "$GPU" = true ]; then
    print_info "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i $GPU_ID || {
            print_error "GPU $GPU_ID not found!"
            exit 1
        }
        print_success "GPU $GPU_ID is available"
    else
        print_warning "nvidia-smi not found, cannot verify GPU"
    fi
fi

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查Python脚本是否存在
if [ ! -f "run_complete_ablation.py" ]; then
    print_error "run_complete_ablation.py not found!"
    exit 1
fi

# 构建Python命令
PYTHON_CMD="python run_complete_ablation.py"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD --gpu $GPU"
PYTHON_CMD="$PYTHON_CMD --gpu_id $GPU_ID"

if [ "$RESUME" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --resume"
fi

if [ "$SKIP_BASELINE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_baseline"
fi

# 打印命令
print_info "Running command:"
echo "  $PYTHON_CMD"
echo ""

# 运行实验
print_info "Starting ablation study..."
eval $PYTHON_CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    print_success "Ablation study completed!"
    echo ""

    # 运行结果分析
    RESULTS_FILE="$SCRIPT_DIR/results/ablation/ablation_results_${DATASET}_seed${SEED}.json"

    if [ -f "$RESULTS_FILE" ]; then
        print_info "Analyzing results..."
        python analyze_ablation_results.py --results_file "$RESULTS_FILE"

        if [ $? -eq 0 ]; then
            print_success "Analysis completed!"
            print_info "Results directory: $SCRIPT_DIR/results/ablation/"
        else
            print_warning "Analysis failed, but results are saved"
        fi
    else
        print_warning "Results file not found: $RESULTS_FILE"
    fi
else
    print_error "Ablation study failed!"
    exit 1
fi

echo ""
print_success "All done!"
