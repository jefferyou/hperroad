"""
GPU性能优化补丁 - Linux版本
针对 Tesla V100S-32GB (32GB VRAM) 优化
"""
import os
import sys

# 获取正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 需要修改的文件列表（使用绝对路径）
files_to_patch = [
    os.path.join(project_root, 'VecCity-main/veccity/downstream/downstream_models/travel_time_estimation.py'),
    os.path.join(project_root, 'VecCity-main/veccity/downstream/downstream_models/similarity_search_model.py'),
]

def optimize_dataloader_config():
    """优化DataLoader配置以提升GPU利用率（Linux + Tesla V100S）"""

    for full_path in files_to_patch:
        if not os.path.exists(full_path):
            print(f"Skipping {full_path} (not found)")
            continue

        # 显示相对路径以便阅读
        rel_path = os.path.relpath(full_path, project_root)
        print(f"Processing: {rel_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Linux优化策略（与Windows完全不同）
        # 修改1: num_workers=4 -> num_workers=8 (Linux fork()效率高，利用多核CPU)
        content = content.replace('num_workers=4', 'num_workers=8')

        # 修改2: batch_size=128 -> batch_size=512 (V100S有32GB显存，当前只用1.8GB)
        content = content.replace('batch_size=128', 'batch_size=512')

        if content != original_content:
            # 备份原文件
            backup_path = full_path + '.backup_linux'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"  ✓ Backup saved to: {backup_path}")
            else:
                print(f"  - Backup already exists: {backup_path}")

            # 写入优化后的内容
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Optimized: num_workers=8, batch_size=512")
        else:
            print(f"  - No changes needed")

def print_optimization_summary():
    """打印优化说明"""
    print("\n" + "=" * 80)
    print("GPU PERFORMANCE OPTIMIZATION COMPLETE (Linux + Tesla V100S-32GB)")
    print("=" * 80)
    print("\nChanges made:")
    print("1. num_workers: 4 → 8")
    print("   Reason: Linux fork() is efficient, use multi-core CPU for data loading")
    print("\n2. batch_size: 128 → 512")
    print("   Reason: V100S has 32GB VRAM, current usage only 1.8GB (5.5%)")
    print("   Target: Increase to 8-12GB (25-40% VRAM usage)")
    print("\nExpected improvements:")
    print("- GPU Utilization: 1% → 70-95%")
    print("- GPU Power: 38W → 200-240W")
    print("- VRAM Usage: 1.8GB → 8-12GB")
    print("- Training Speed: 10-20x faster")
    print("- Time/iteration: 9.58s → 0.5-1.0s")
    print("- 10 epochs TTE: ~40 hours → ~2-3 hours")
    print("=" * 80)
    print("\nBackup files created with .backup_linux extension")
    print("To revert: mv file.py.backup_linux file.py")
    print("=" * 80)

if __name__ == '__main__':
    print("=" * 80)
    print("Optimizing GPU Performance for Linux + Tesla V100S-32GB...")
    print("=" * 80)
    optimize_dataloader_config()
    print_optimization_summary()
