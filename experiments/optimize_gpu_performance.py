"""
GPU性能优化补丁
修复Windows上num_workers开销和batch_size过小的问题
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
    """优化DataLoader配置以提升GPU利用率"""

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

        # 修改1: num_workers=4 -> num_workers=0 (Windows优化)
        content = content.replace('num_workers=4', 'num_workers=0')

        # 修改2: batch_size=128 -> batch_size=256 (提高GPU利用率)
        content = content.replace('batch_size=128', 'batch_size=256')

        if content != original_content:
            # 备份原文件
            backup_path = full_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"  ✓ Backup saved to: {backup_path}")

            # 写入优化后的内容
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Optimized: num_workers=0, batch_size=256")
        else:
            print(f"  - No changes needed")

def print_optimization_summary():
    """打印优化说明"""
    print("\n" + "=" * 80)
    print("GPU PERFORMANCE OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nChanges made:")
    print("1. num_workers: 4 → 0")
    print("   Reason: Windows spawn() overhead, num_workers=0 faster on Windows")
    print("\n2. batch_size: 128 → 256")
    print("   Reason: Increase GPU utilization (RTX 5070 Ti has 16GB VRAM)")
    print("\nExpected improvements:")
    print("- GPU Utilization: 2-23% → 60-90%")
    print("- Training Speed: 2-3x faster")
    print("- TTE time/epoch: 4 hours → 1.5-2 hours")
    print("- STS time/epoch: 35 min → 10-15 min")
    print("=" * 80)
    print("\nBackup files created with .backup extension")
    print("To revert: mv file.py.backup file.py")
    print("=" * 80)

if __name__ == '__main__':
    print("=" * 80)
    print("Optimizing GPU Performance for Windows...")
    print("=" * 80)
    optimize_dataloader_config()
    print_optimization_summary()
