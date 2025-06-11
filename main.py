#!/usr/bin/env python3
"""
GPT-1.5B JAX 推理性能测试主程序
Ubuntu + 4*RTX3080 + CUDA11.8 + JAX0.6.1
使用图分割方法进行多GPU并行推理
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# 清理可能存在的XLA_FLAGS
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

import jax
import numpy as np
from src.models.gpt_model import GPTConfig, GraphPartitionedGPT
from src.data.dataset_loader import DatasetLoader
from src.inference.benchmark import InferenceBenchmark
from src.utils.gpu_utils import setup_jax_environment, check_gpu_setup, print_gpu_status
from src.utils.results import save_benchmark_results, print_performance_summary


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='GPT-1.5B JAX 推理性能测试')
    parser.add_argument('--dataset-dir', default='datasets', help='数据集目录路径')
    parser.add_argument('--output-dir', default='results', help='结果输出目录')
    parser.add_argument('--config', type=str, help='测试特定配置ID (例如: 0,1,3)')
    parser.add_argument('--max-samples', type=int, default=10, help='每个配置的最大测试样本数')
    parser.add_argument('--show-gpu-info', action='store_true', help='显示详细GPU信息')
    
    args = parser.parse_args()
    
    print("🎯 GPT-1.5B JAX 推理性能测试")
    print("=" * 60)
    print(f"💻 平台: Windows")
    print(f"🐍 Python: 3.10")
    print(f"⚡ CUDA: 11.8")
    
    # 设置JAX环境
    setup_jax_environment()
    
    # 检查GPU环境
    if not check_gpu_setup():
        print("❌ GPU环境检查失败，无法继续")
        return
    
    if args.show_gpu_info:
        print_gpu_status()
    
    # 初始化模型
    print(f"\n🏗️ 初始化GPT-1.5B模型...")
    config = GPTConfig()
    model = GraphPartitionedGPT(config)
    
    # 加载数据集
    print(f"\n📂 加载数据集从: {args.dataset_dir}")
    dataset_loader = DatasetLoader(args.dataset_dir)
    datasets = dataset_loader.get_valid_datasets()
    
    if not datasets:
        print("❌ 未找到有效数据集，请检查数据集文件")
        return
    
    # 显示数据集统计
    dataset_loader.print_dataset_summary()
    
    # 过滤特定配置
    if args.config:
        config_ids = [c.strip() for c in args.config.split(',')]
        filtered_datasets = {k: v for k, v in datasets.items() if k in config_ids}
        if filtered_datasets:
            datasets = filtered_datasets
            print(f"\n🎯 只测试指定配置: {list(datasets.keys())}")
        else:
            print(f"❌ 指定的配置 {config_ids} 不存在")
            return
    
    # 运行基准测试
    benchmark = InferenceBenchmark(model)
    results = benchmark.run_full_benchmark(datasets, args.max_samples)
    
    # 保存结果
    json_file, report_file = save_benchmark_results(results, args.output_dir)
    
    # 显示最终摘要
    print_performance_summary(results)
    
    print(f"\n📁 结果文件:")
    print(f"   {report_file.name}")
    print(f"   {json_file.name}")


if __name__ == "__main__":
    main()
