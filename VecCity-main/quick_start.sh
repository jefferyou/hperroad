#!/bin/bash

################################################################################
# HRNR_Hyperbolic 快速启动脚本
#
# 这个脚本会自动：
# 1. 检查环境和依赖
# 2. 验证embeddings文件
# 3. 运行实验（5个城市：prt, cd, bj, xa, sf）
# 4. 分析结果
#
# 使用方法:
#   bash quick_start.sh
#   bash quick_start.sh --help
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 辅助函数
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
HRNR_Hyperbolic 快速启动脚本

使用方法:
    bash quick_start.sh [选项]

选项:
    --help              显示此帮助信息
    --device <device>   指定设备 (cpu/gpu, 默认: gpu)
    --datasets <list>   指定数据集列表 (默认: prt cd bj xa sf)
    --seeds <list>      指定随机种子列表 (默认: 31 42 53 64 75)
    --skip-check        跳过环境检查
    --skip-analysis     跳过结果分析
    --exp-id <id>       指定实验ID (默认: 1)

示例:
    # 使用默认配置运行
    bash quick_start.sh

    # 只在两个城市上运行，使用CPU
    bash quick_start.sh --device cpu --datasets "bj cd"

    # 只运行3次
    bash quick_start.sh --seeds "31 42 53"

    # 跳过检查直接运行
    bash quick_start.sh --skip-check
EOF
    exit 0
}

# 解析命令行参数
DEVICE="gpu"
DATASETS="prt cd bj xa sf"
SEEDS="31 42 53 64 75"
SKIP_CHECK=false
SKIP_ANALYSIS=false
EXP_ID="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        *)
            print_error "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 主函数
main() {
    print_header "HRNR_Hyperbolic 实验自动运行脚本"

    echo ""
    print_info "配置信息:"
    echo "  设备: ${DEVICE}"
    echo "  数据集: ${DATASETS}"
    echo "  随机种子: ${SEEDS}"
    echo "  实验ID: ${EXP_ID}"
    echo ""

    # 步骤1: 环境检查
    if [ "$SKIP_CHECK" = false ]; then
        print_header "步骤 1/4: 环境检查"
        check_environment
    else
        print_warning "跳过环境检查"
    fi

    # 步骤2: 验证embeddings
    print_header "步骤 2/4: 验证Embeddings文件"
    check_embeddings

    # 步骤3: 运行实验
    print_header "步骤 3/4: 运行实验"
    run_experiments

    # 步骤4: 分析结果
    if [ "$SKIP_ANALYSIS" = false ]; then
        print_header "步骤 4/4: 分析结果"
        analyze_results
    else
        print_warning "跳过结果分析"
    fi

    # 完成
    print_header "实验完成"
    print_success "所有步骤已完成！"
    echo ""
    print_info "结果保存在: ${RESULT_DIR}"
    echo ""
}

# 环境检查
check_environment() {
    echo ""
    print_info "检查Python环境..."

    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "Python未安装"
        exit 1
    fi
    print_success "Python: $(python --version)"

    # 检查必要的Python包
    print_info "检查Python包..."

    REQUIRED_PACKAGES="torch numpy pandas"
    for package in $REQUIRED_PACKAGES; do
        if python -c "import $package" 2>/dev/null; then
            print_success "  $package: 已安装"
        else
            print_error "  $package: 未安装"
            print_warning "请运行: pip install $package"
            exit 1
        fi
    done

    # 检查GPU（如果使用GPU）
    if [ "$DEVICE" = "gpu" ] || [ "$DEVICE" = "cuda" ]; then
        print_info "检查GPU..."
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
            GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
            print_success "  GPU可用: $GPU_NAME (x$GPU_COUNT)"
        else
            print_warning "  GPU不可用，将使用CPU"
            DEVICE="cpu"
        fi
    fi

    # 检查必要的脚本
    print_info "检查脚本文件..."
    REQUIRED_SCRIPTS="run_hrnr_experiments.py run_downstream_only.py analyze_results.py"
    for script in $REQUIRED_SCRIPTS; do
        if [ -f "$script" ]; then
            print_success "  $script: 存在"
        else
            print_error "  $script: 不存在"
            exit 1
        fi
    done

    echo ""
    print_success "环境检查完成"
}

# 检查embeddings文件
check_embeddings() {
    echo ""
    print_info "检查embeddings文件..."

    CACHE_DIR="./veccity/cache/${EXP_ID}/evaluate_cache"

    if [ ! -d "$CACHE_DIR" ]; then
        print_error "缓存目录不存在: $CACHE_DIR"
        print_warning "请先运行模型训练生成embeddings"
        exit 1
    fi

    MISSING_EMBEDDINGS=()

    for dataset in $DATASETS; do
        EMBEDDING_FILE="${CACHE_DIR}/road_embedding_HRNR_Hyperbolic_${dataset}_128.npy"

        if [ -f "$EMBEDDING_FILE" ]; then
            FILE_SIZE=$(ls -lh "$EMBEDDING_FILE" | awk '{print $5}')
            print_success "  ${dataset}: ${EMBEDDING_FILE} (${FILE_SIZE})"
        else
            print_warning "  ${dataset}: 未找到 ${EMBEDDING_FILE}"
            MISSING_EMBEDDINGS+=("$dataset")
        fi
    done

    if [ ${#MISSING_EMBEDDINGS[@]} -gt 0 ]; then
        echo ""
        print_warning "以下数据集的embeddings未找到:"
        for dataset in "${MISSING_EMBEDDINGS[@]}"; do
            echo "  - $dataset"
        done

        echo ""
        read -p "是否继续运行已有embeddings的数据集? (y/n) " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "已取消"
            exit 0
        fi

        # 更新DATASETS，只包含有embeddings的数据集
        NEW_DATASETS=""
        for dataset in $DATASETS; do
            if [[ ! " ${MISSING_EMBEDDINGS[@]} " =~ " ${dataset} " ]]; then
                NEW_DATASETS="$NEW_DATASETS $dataset"
            fi
        done
        DATASETS=$(echo $NEW_DATASETS | xargs)  # 去除首尾空格
        print_info "将运行的数据集: $DATASETS"
    fi

    echo ""
    print_success "Embeddings检查完成"
}

# 运行实验
run_experiments() {
    echo ""
    print_info "开始运行实验..."

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULT_DIR="./experiment_results/${TIMESTAMP}"

    # 转换数据集和种子列表为数组
    DATASETS_ARRAY=($DATASETS)
    SEEDS_ARRAY=($SEEDS)

    TOTAL_EXPERIMENTS=$((${#DATASETS_ARRAY[@]} * ${#SEEDS_ARRAY[@]}))

    echo ""
    print_info "实验配置:"
    echo "  数据集数量: ${#DATASETS_ARRAY[@]}"
    echo "  每个数据集运行次数: ${#SEEDS_ARRAY[@]}"
    echo "  总实验数: ${TOTAL_EXPERIMENTS}"
    echo "  预计时间: $(($TOTAL_EXPERIMENTS * 15 / 60)) - $(($TOTAL_EXPERIMENTS * 30 / 60)) 小时"
    echo ""

    read -p "是否开始运行? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi

    # 运行Python实验脚本
    python run_hrnr_experiments.py \
        --task segment \
        --model HRNR_Hyperbolic \
        --datasets $DATASETS \
        --seeds $SEEDS \
        --output_dim 128 \
        --device $DEVICE \
        --exp_id $EXP_ID \
        --output_dir "$RESULT_DIR"

    if [ $? -eq 0 ]; then
        print_success "实验运行完成"
        export RESULT_DIR  # 导出给后续步骤使用
    else
        print_error "实验运行失败"
        exit 1
    fi
}

# 分析结果
analyze_results() {
    echo ""
    print_info "开始分析结果..."

    if [ -z "$RESULT_DIR" ]; then
        print_error "结果目录未设置"
        exit 1
    fi

    # 生成多种格式的报告
    print_info "生成CSV报告..."
    python analyze_results.py "$RESULT_DIR" --format csv --output "${RESULT_DIR}/summary.csv"

    print_info "生成Markdown报告..."
    python analyze_results.py "$RESULT_DIR" --format markdown --output "${RESULT_DIR}/summary.md"

    print_info "生成LaTeX表格..."
    python analyze_results.py "$RESULT_DIR" --format latex --output "${RESULT_DIR}/table.tex"

    print_info "生成JSON报告..."
    python analyze_results.py "$RESULT_DIR" --format json --output "${RESULT_DIR}/summary.json"

    echo ""
    print_success "结果分析完成"
    echo ""
    print_info "生成的文件:"
    echo "  - CSV: ${RESULT_DIR}/summary.csv"
    echo "  - Markdown: ${RESULT_DIR}/summary.md"
    echo "  - LaTeX: ${RESULT_DIR}/table.tex"
    echo "  - JSON: ${RESULT_DIR}/summary.json"
}

# 运行主函数
main
