#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证GPU配置是否正确传递到整个流程

检查：
1. 配置文件中的GPU设置
2. PyTorch GPU可用性
3. 实验脚本的参数传递
"""

import sys
import os
import json
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


def check_pytorch_gpu():
    """检查PyTorch GPU支持"""
    print("\n" + "="*80)
    print("1. PyTorch GPU Support")
    print("="*80)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA is available, {gpu_count} GPU(s) detected")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # 测试GPU操作
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.matmul(x, x)
                print_success("GPU tensor operations work correctly")
                return True
            except Exception as e:
                print_error(f"GPU tensor operations failed: {e}")
                return False
        else:
            print_error("CUDA not available")
            print_info("Check NVIDIA drivers and CUDA installation")
            return False

    except ImportError:
        print_error("PyTorch not installed")
        return False


def check_config_files():
    """检查配置文件中的GPU设置"""
    print("\n" + "="*80)
    print("2. Configuration Files GPU Settings")
    print("="*80)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_dir = project_root / 'VecCity-main' / 'veccity' / 'config' / 'model' / 'segment'

    configs = [
        'HRNR.json',
        'HRNR_Hyperbolic.json'
    ]

    all_ok = True

    for config_name in configs:
        config_path = config_dir / config_name

        print(f"\n{config_name}:")

        if not config_path.exists():
            print_error(f"  Config file not found: {config_path}")
            all_ok = False
            continue

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            gpu_enabled = config.get('gpu', False)
            device = config.get('device', 'cpu')
            gpu_id = config.get('gpu_id', 'not set')

            print(f"  gpu: {gpu_enabled}")
            print(f"  device: {device}")
            print(f"  gpu_id: {gpu_id}")

            if gpu_enabled and device == 'cuda':
                print_success("  GPU configuration is correct")
            else:
                print_error("  GPU configuration is incorrect")
                print_info("  Expected: gpu=true, device='cuda'")
                all_ok = False

        except Exception as e:
            print_error(f"  Failed to read config: {e}")
            all_ok = False

    return all_ok


def check_experiment_scripts():
    """检查实验脚本的GPU参数传递"""
    print("\n" + "="*80)
    print("3. Experiment Scripts GPU Parameter Passing")
    print("="*80)

    script_dir = Path(__file__).parent

    scripts = {
        'run_complete_ablation.py': 'Ablation experiment script',
        'run_hrnr_hyperbolic.py': 'HRNR_Hyperbolic experiment script'
    }

    all_ok = True

    for script_name, desc in scripts.items():
        script_path = script_dir / script_name

        print(f"\n{script_name} - {desc}:")

        if not script_path.exists():
            print_warning(f"  Script not found: {script_path}")
            continue

        try:
            with open(script_path, 'r') as f:
                content = f.read()

            # 检查GPU参数传递
            checks = {
                "'gpu'": "GPU parameter",
                "'gpu_id'": "GPU ID parameter",
                "other_args": "Parameter dictionary"
            }

            for check, desc in checks.items():
                if check in content:
                    print_success(f"  {desc} found")
                else:
                    print_warning(f"  {desc} not found")

        except Exception as e:
            print_error(f"  Failed to read script: {e}")
            all_ok = False

    return all_ok


def generate_gpu_config_summary():
    """生成GPU配置总结"""
    print("\n" + "="*80)
    print("4. GPU Configuration Summary")
    print("="*80)

    summary_file = Path(__file__).parent / 'GPU_CONFIG_SUMMARY.md'

    with open(summary_file, 'w') as f:
        f.write("# GPU Configuration Summary\n\n")
        f.write("## Configuration Files\n\n")
        f.write("### HRNR.json & HRNR_Hyperbolic.json\n\n")
        f.write("```json\n")
        f.write('{\n')
        f.write('  "gpu": true,\n')
        f.write('  "gpu_id": 0,\n')
        f.write('  "device": "cuda"\n')
        f.write('}\n')
        f.write("```\n\n")

        f.write("## Experiment Scripts\n\n")
        f.write("GPU parameters are passed via `other_args`:\n\n")
        f.write("```python\n")
        f.write("other_args = {\n")
        f.write("    'gpu': True,\n")
        f.write("    'gpu_id': 0,\n")
        f.write("    ...\n")
        f.write("}\n")
        f.write("```\n\n")

        f.write("## Verification Steps\n\n")
        f.write("1. Run this script: `python verify_gpu_config.py`\n")
        f.write("2. Check PyTorch GPU: `python -c 'import torch; print(torch.cuda.is_available())'`\n")
        f.write("3. Monitor GPU usage: `nvidia-smi` or `watch -n 1 nvidia-smi`\n")
        f.write("4. Check experiment logs for device placement\n\n")

        f.write("## Expected GPU Usage\n\n")
        f.write("During training, you should see:\n")
        f.write("- GPU memory usage increase\n")
        f.write("- GPU utilization % increase\n")
        f.write("- Faster training compared to CPU\n\n")

        f.write("## Troubleshooting\n\n")
        f.write("If GPU is not being used:\n\n")
        f.write("1. **Check CUDA availability**\n")
        f.write("   ```bash\n")
        f.write("   nvidia-smi\n")
        f.write("   python -c 'import torch; print(torch.cuda.is_available())'\n")
        f.write("   ```\n\n")

        f.write("2. **Verify config files**\n")
        f.write("   - Check `gpu: true` and `device: 'cuda'` in JSON configs\n\n")

        f.write("3. **Check experiment parameters**\n")
        f.write("   ```bash\n")
        f.write("   ./run_ablation.sh --gpu-id 0  # Explicitly set GPU\n")
        f.write("   ```\n\n")

        f.write("4. **Monitor during training**\n")
        f.write("   ```bash\n")
        f.write("   watch -n 1 nvidia-smi\n")
        f.write("   ```\n\n")

    print_success(f"Summary saved to: {summary_file}")
    return True


def main():
    """主函数"""
    print("=" * 80)
    print(" GPU CONFIGURATION VERIFICATION")
    print("=" * 80)

    results = []

    # 运行所有检查
    results.append(("PyTorch GPU", check_pytorch_gpu()))
    results.append(("Config Files", check_config_files()))
    results.append(("Experiment Scripts", check_experiment_scripts()))
    results.append(("Summary Generation", generate_gpu_config_summary()))

    # 总结
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80 + "\n")

    passed = 0
    failed = 0

    for name, result in results:
        if result:
            print_success(f"{name}: PASS")
            passed += 1
        else:
            print_error(f"{name}: FAIL")
            failed += 1

    print(f"\n{'='*80}")
    print(f"Total: {passed} passed, {failed} failed")
    print(f"{'='*80}\n")

    if failed == 0:
        print_success("✓ All GPU configurations are correct!")
        print_info("\nYou can now run experiments with GPU:")
        print("  ./run_ablation.sh --gpu-id 0")
        print("\nMonitor GPU usage:")
        print("  nvidia-smi")
        print("  watch -n 1 nvidia-smi")
        return 0
    else:
        print_error("✗ Some GPU configurations need fixing.")
        print_info("\nPlease fix the issues above and run this script again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
