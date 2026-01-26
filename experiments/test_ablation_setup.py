#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试消融实验环境配置

检查：
1. Python环境
2. VecCity路径
3. GPU可用性
4. 必要的依赖
5. 目录权限
"""

import sys
import os
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

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"\n{'='*80}")
    print("1. Checking Python Version")
    print(f"{'='*80}")

    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 7:
        print_success("Python version is compatible (>= 3.7)")
        return True
    else:
        print_error("Python version too old, need >= 3.7")
        return False

def check_veccity():
    """检查VecCity路径"""
    print(f"\n{'='*80}")
    print("2. Checking VecCity Installation")
    print(f"{'='*80}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    veccity_root = project_root / 'VecCity-main'

    print(f"Expected VecCity path: {veccity_root}")

    if veccity_root.exists():
        print_success(f"VecCity directory found")

        # 检查关键文件
        pipeline_file = veccity_root / 'veccity' / 'pipeline' / '__init__.py'
        if pipeline_file.exists():
            print_success("VecCity pipeline found")
        else:
            print_error(f"VecCity pipeline not found: {pipeline_file}")
            return False

        # 检查HRNR模型
        hrnr_file = veccity_root / 'veccity' / 'upstream' / 'road_representation' / 'HRNR.py'
        hrnr_hyp_file = veccity_root / 'veccity' / 'upstream' / 'road_representation' / 'HRNR_Hyperbolic.py'

        if hrnr_file.exists():
            print_success("HRNR model found")
        else:
            print_error(f"HRNR model not found: {hrnr_file}")

        if hrnr_hyp_file.exists():
            print_success("HRNR_Hyperbolic model found")
        else:
            print_error(f"HRNR_Hyperbolic model not found: {hrnr_hyp_file}")
            return False

        return True
    else:
        print_error(f"VecCity directory not found: {veccity_root}")
        return False

def check_gpu():
    """检查GPU可用性"""
    print(f"\n{'='*80}")
    print("3. Checking GPU Availability")
    print(f"{'='*80}")

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

            return True
        else:
            print_warning("CUDA not available, will use CPU")
            print_info("You can still run experiments, but it will be slower")
            return True

    except ImportError:
        print_error("PyTorch not installed")
        print_info("Install with: pip install torch")
        return False

def check_dependencies():
    """检查必要的依赖"""
    print(f"\n{'='*80}")
    print("4. Checking Dependencies")
    print(f"{'='*80}")

    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'torch': 'torch',
    }

    optional = {
        'matplotlib': 'matplotlib (for plotting)',
    }

    all_ok = True

    print("\nRequired packages:")
    for package, install_name in required.items():
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} - install with: pip install {install_name}")
            all_ok = False

    print("\nOptional packages:")
    for package, desc in optional.items():
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_warning(f"{package} - {desc}")

    return all_ok

def check_directories():
    """检查目录权限"""
    print(f"\n{'='*80}")
    print("5. Checking Directories and Permissions")
    print(f"{'='*80}")

    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results' / 'ablation'

    print(f"Script directory: {script_dir}")
    print(f"Results directory: {results_dir}")

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        print_success("Results directory created/accessible")

        # 测试写入权限
        test_file = results_dir / '.test_write'
        test_file.write_text('test')
        test_file.unlink()
        print_success("Write permission OK")

        return True

    except Exception as e:
        print_error(f"Directory access error: {e}")
        return False

def check_scripts():
    """检查脚本文件"""
    print(f"\n{'='*80}")
    print("6. Checking Script Files")
    print(f"{'='*80}")

    script_dir = Path(__file__).parent

    scripts = {
        'run_complete_ablation.py': 'Main ablation script',
        'analyze_ablation_results.py': 'Results analysis script',
        'run_ablation.sh': 'Bash launcher script',
    }

    all_ok = True

    for script, desc in scripts.items():
        script_path = script_dir / script
        if script_path.exists():
            print_success(f"{script} - {desc}")
        else:
            print_error(f"{script} not found - {desc}")
            all_ok = False

    return all_ok

def test_import_veccity():
    """测试导入VecCity"""
    print(f"\n{'='*80}")
    print("7. Testing VecCity Import")
    print(f"{'='*80}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    veccity_root = project_root / 'VecCity-main'

    # 临时添加到路径
    sys.path.insert(0, str(veccity_root))

    try:
        from veccity.utils import ensure_dir
        print_success("veccity.utils imported successfully")

        from veccity.pipeline import run_model
        print_success("veccity.pipeline imported successfully")

        return True

    except Exception as e:
        print_error(f"Failed to import VecCity: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print(" ABLATION EXPERIMENT ENVIRONMENT TEST")
    print("=" * 80)

    results = []

    # 运行所有检查
    results.append(("Python Version", check_python_version()))
    results.append(("VecCity Installation", check_veccity()))
    results.append(("GPU Availability", check_gpu()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Directories", check_directories()))
    results.append(("Script Files", check_scripts()))
    results.append(("VecCity Import", test_import_veccity()))

    # 总结
    print(f"\n{'='*80}")
    print(" SUMMARY")
    print(f"{'='*80}\n")

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
        print_success("✓ All checks passed! Ready to run ablation experiments.")
        print_info("\nNext steps:")
        print("  1. Run: ./run_ablation.sh")
        print("  2. Or: python run_complete_ablation.py --dataset xa --seed 0")
        return 0
    else:
        print_error("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
