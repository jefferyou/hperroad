#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNR_Hyperbolic 使用示例

演示如何使用实验框架进行各种实验
"""

import os
import sys

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../VecCity-main')))


def example_1_single_experiment():
    """示例1: 运行单次实验"""
    print("=" * 80)
    print("示例1: 运行单次实验")
    print("=" * 80)

    from veccity.pipeline import run_model

    # 配置参数
    config = {
        'hyperbolic_dim': 224,
        'lambda_ce': 0.1,
        'lambda_cc': 0.1,
        'temperature': 0.07,
        'lp_learning_rate': 1e-4,
        'max_epoch': 10,  # 演示用，实际应设置更大
    }

    # 运行模型
    result = run_model(
        task='road_representation',
        model_name='HRNR_Hyperbolic',
        dataset_name='xa',
        saved_model=True,
        train=True,
        other_args=config,
        seed=0,
        gpu_id=0,
        gpu=True,
        exp_id='example_single_exp'
    )

    print(f"\n实验结果: {result}")
    print("=" * 80 + "\n")


def example_2_hyperparameter_search():
    """示例2: 超参数搜索"""
    print("=" * 80)
    print("示例2: 超参数搜索")
    print("=" * 80)

    from hyperparameter_tuning import HyperparameterTuner

    # 基础配置
    base_config = {
        'gpu': True,
        'gpu_id': 0,
        'seed': 0,
        'max_epoch': 10,  # 演示用
    }

    # 搜索空间（简化版）
    search_space = {
        'hyperbolic_dim': [128, 224],
        'lambda_ce': [0.05, 0.1, 0.15],
        'lambda_cc': [0.05, 0.1, 0.15],
        'temperature': [0.05, 0.07, 0.1],
    }

    # 创建调优器
    tuner = HyperparameterTuner(
        task='road_representation',
        model='HRNR_Hyperbolic',
        dataset='xa',
        base_config=base_config,
        search_space=search_space,
        metric='auc',
        mode='max',
        max_trials=5,  # 演示用，实际应设置更大
        method='random'
    )

    # 运行调优
    results = tuner.run()

    print(f"\n最佳配置: {results['best_trial']['hyperparams']}")
    print(f"最佳AUC: {results['best_score']:.4f}")
    print("=" * 80 + "\n")


def example_3_visualization():
    """示例3: 结果可视化"""
    print("=" * 80)
    print("示例3: 结果可视化")
    print("=" * 80)

    from visualization_tools import ExperimentVisualizer

    visualizer = ExperimentVisualizer()

    # 假设已经运行了实验并生成了结果文件
    # visualizer.plot_training_curves('path/to/log.log')
    # visualizer.plot_hyperparameter_importance('results/hypertuning_*.json')
    # visualizer.plot_ablation_study('results/*_ablation_study.json')
    # visualizer.plot_model_comparison('results/hrnr_comparison_*.json')

    print("可视化工具已准备就绪")
    print("请使用实际的结果文件路径调用可视化函数")
    print("=" * 80 + "\n")


def example_4_batch_experiments():
    """示例4: 批量实验"""
    print("=" * 80)
    print("示例4: 批量实验（演示代码）")
    print("=" * 80)

    datasets = ['xa']  # 可以添加更多数据集
    seeds = [0, 1, 2]

    print("批量实验配置:")
    print(f"  数据集: {datasets}")
    print(f"  随机种子: {seeds}")
    print(f"  总实验数: {len(datasets) * len(seeds)}")

    # 实际运行时取消注释
    # from veccity.pipeline import run_model
    #
    # for dataset in datasets:
    #     for seed in seeds:
    #         print(f"\nRunning: dataset={dataset}, seed={seed}")
    #         result = run_model(
    #             task='road_representation',
    #             model_name='HRNR_Hyperbolic',
    #             dataset_name=dataset,
    #             seed=seed,
    #             exp_id=f'batch_{dataset}_s{seed}'
    #         )

    print("\n注意: 批量实验代码已注释，取消注释后运行")
    print("=" * 80 + "\n")


def example_5_custom_config():
    """示例5: 使用自定义配置"""
    print("=" * 80)
    print("示例5: 使用自定义配置")
    print("=" * 80)

    import json

    # 创建自定义配置
    custom_config = {
        'hyperbolic_dim': 256,
        'lambda_ce': 0.15,
        'lambda_cc': 0.12,
        'temperature': 0.08,
        'lp_learning_rate': 8e-5,
        'max_epoch': 150,
        'dropout': 0.6,
        'alpha': 0.2,
        'hidden_dims': 256,
        'struct_cmt_num': 300,
        'fnc_cmt_num': 30
    }

    # 保存配置
    config_file = './custom_config_example.json'
    with open(config_file, 'w') as f:
        json.dump(custom_config, f, indent=2)

    print(f"自定义配置已保存到: {config_file}")
    print("配置内容:")
    print(json.dumps(custom_config, indent=2))

    # 使用配置运行实验
    # from veccity.pipeline import run_model
    # result = run_model(
    #     task='road_representation',
    #     model_name='HRNR_Hyperbolic',
    #     dataset_name='xa',
    #     config_file=config_file
    # )

    print("\n注意: 运行代码已注释，取消注释后使用自定义配置")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("HRNR_Hyperbolic 使用示例集")
    print("=" * 80 + "\n")

    examples = {
        '1': ('单次实验', example_1_single_experiment),
        '2': ('超参数搜索', example_2_hyperparameter_search),
        '3': ('结果可视化', example_3_visualization),
        '4': ('批量实验', example_4_batch_experiments),
        '5': ('自定义配置', example_5_custom_config),
    }

    print("可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. 运行所有示例（非计算密集型）")
    print("  q. 退出")

    while True:
        choice = input("\n请选择示例编号 (或 'q' 退出): ").strip()

        if choice == 'q':
            print("退出")
            break
        elif choice == '0':
            # 只运行非计算密集型示例
            example_3_visualization()
            example_4_batch_experiments()
            example_5_custom_config()
            break
        elif choice in examples:
            _, func = examples[choice]
            try:
                func()
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"无效选择: {choice}")

    print("\n" + "=" * 80)
    print("示例结束")
    print("=" * 80)


if __name__ == '__main__':
    main()
