"""
结果保存和报告生成工具
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any


def save_benchmark_results(results: Dict, output_dir: str = 'results') -> tuple:
    """保存基准测试结果"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存详细JSON结果
    json_file = output_path / f"gpt15b_benchmark_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成可读性报告
    report_file = output_path / f"performance_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("GPT-1.5B JAX 推理性能测试报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        info = results['benchmark_info']
        f.write(f"测试时间: {info['timestamp']}\n")
        f.write(f"总执行时间: {info['total_execution_time']:.2f}秒\n")
        f.write(f"GPU数量: {info['gpu_count']}\n")
        f.write(f"JAX版本: {info['jax_version']}\n")
        f.write(f"CUDA版本: {info['cuda_version']}\n")
        f.write(f"平台: {info['platform']}\n\n")
        
        # 模型配置
        model_cfg = info['model_config']
        f.write("模型配置:\n")
        f.write(f"  层数: {model_cfg['n_layer']}\n")
        f.write(f"  注意力头数: {model_cfg['n_head']}\n")
        f.write(f"  嵌入维度: {model_cfg['n_embd']}\n")
        f.write(f"  词汇表大小: {model_cfg['vocab_size']}\n\n")
        
        # 性能结果
        f.write("性能测试结果:\n")
        f.write("-" * 40 + "\n")
        
        all_throughputs = []
        all_latencies = []
        
        for config_id, result in results['results'].items():
            f.write(f"\n配置 {config_id}:\n")
            f.write(f"  测试样本数: {result['samples_tested']}\n")
            f.write(f"  平均推理时间: {result['avg_inference_time']:.3f}±{result['std_inference_time']:.3f}s\n")
            f.write(f"  平均吞吐量: {result['avg_throughput']:.1f} tokens/s\n")
            f.write(f"  吞吐量范围: {result['min_throughput']:.1f} - {result['max_throughput']:.1f} tokens/s\n")
            f.write(f"  平均每token延迟: {result['avg_latency_per_token']:.4f}s\n")
            f.write(f"  总生成token数: {result['total_tokens_generated']}\n")
            
            all_throughputs.append(result['avg_throughput'])
            all_latencies.append(result['avg_latency_per_token'])
        
        # 总体统计
        if all_throughputs:
            f.write(f"\n总体性能统计:\n")
            f.write(f"  平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s\n")
            f.write(f"  最高吞吐量: {max(all_throughputs):.1f} tokens/s\n")
            f.write(f"  平均延迟: {np.mean(all_latencies):.4f}s/token\n")
            f.write(f"  最低延迟: {min(all_latencies):.4f}s/token\n")
    
    print(f"\n💾 结果已保存:")
    print(f"   详细结果: {json_file}")
    print(f"   性能报告: {report_file}")
    
    return json_file, report_file


def print_performance_summary(results: Dict):
    """打印性能摘要"""
    print(f"\n🎉 基准测试完成!")
    print(f"📊 测试配置数: {len(results['results'])}")
    print(f"⏱️ 总执行时间: {results['benchmark_info']['total_execution_time']:.2f}秒")
    
    # 显示性能亮点
    if results['results']:
        all_throughputs = [r['avg_throughput'] for r in results['results'].values()]
        all_latencies = [r['avg_latency_per_token'] for r in results['results'].values()]
        
        print(f"\n🏆 性能亮点:")
        print(f"   最高吞吐量: {max(all_throughputs):.1f} tokens/s")
        print(f"   最低延迟: {min(all_latencies):.4f}s/token")
        print(f"   平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s")
