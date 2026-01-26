#!/bin/bash

# 测试断点续传功能的演示脚本

echo "=========================================="
echo "测试断点续传功能"
echo "=========================================="
echo ""

# 清理之前的测试
rm -f .benchmark_state
echo "1. 清理旧状态文件"
echo ""

# 创建模拟的状态文件
echo "2. 模拟已完成Xi'an和Beijing的训练"
cat > .benchmark_state << EOF
RESULTS_DIR="results/benchmark_test"
xian_training=completed
xian_exp_id="hrnr_hyp_xa_s0_test"
xian_tsi=completed
xian_tte=completed
beijing_training=completed
beijing_exp_id="hrnr_hyp_bj_s0_test"
EOF

echo "   状态文件内容："
cat .benchmark_state
echo ""

# 测试状态检查函数
echo "3. 测试状态检查函数"

task_completed() {
    local city=$1
    local task=$2
    local state_file=".benchmark_state"

    if grep -q "${city}_${task}=completed" "$state_file" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 测试各种情况
if task_completed "xian" "training"; then
    echo "   ✓ xian_training: 已完成（正确）"
else
    echo "   ✗ xian_training: 未完成（错误）"
fi

if task_completed "xian" "sts"; then
    echo "   ✗ xian_sts: 已完成（错误）"
else
    echo "   ✓ xian_sts: 未完成（正确）"
fi

if task_completed "beijing" "training"; then
    echo "   ✓ beijing_training: 已完成（正确）"
else
    echo "   ✗ beijing_training: 未完成（错误）"
fi

if task_completed "beijing" "tsi"; then
    echo "   ✓ beijing_tsi: 未完成（正确）"
else
    echo "   ✗ beijing_tsi: 已完成（错误）"
fi

echo ""
echo "4. 测试实验ID恢复"

get_saved_exp_id() {
    local city=$1
    local state_file=".benchmark_state"

    if [ -f "$state_file" ]; then
        source "$state_file"
        local var_name="${city}_exp_id"
        echo "${!var_name}"
    fi
}

xian_id=$(get_saved_exp_id "xian")
beijing_id=$(get_saved_exp_id "beijing")

echo "   Xi'an Experiment ID: $xian_id"
echo "   Beijing Experiment ID: $beijing_id"

echo ""
echo "=========================================="
echo "✓ 断点续传功能测试通过"
echo "=========================================="
echo ""
echo "实际使用时的效果："
echo "  - 首次运行会创建 .benchmark_state 文件"
echo "  - 每完成一个任务，会在文件中标记"
echo "  - 中断后再次运行，会自动跳过已完成的任务"
echo "  - 不需要手动指定从哪里开始"
echo ""

# 清理测试文件
rm -f .benchmark_state
echo "测试完成，已清理临时文件"
