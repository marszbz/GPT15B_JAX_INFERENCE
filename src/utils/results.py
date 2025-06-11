"""
结果保存和报告生成工具
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResultManager:
    """结果管理器 - 处理基准测试结果的保存和分析"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_cache = {}
    
    def save_results(self, results: Dict, prefix: str = 'gpt15b_benchmark') -> tuple:
        """保存基准测试结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存详细JSON结果
        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成可读性报告
        report_file = self.output_dir / f"performance_report_{timestamp}.txt" 
        self._generate_report(results, report_file)
        
        # 缓存结果
        self.results_cache[timestamp] = results
        
        return json_file, report_file
    
    def _generate_report(self, results: Dict, report_file: Path):
        """生成性能报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GPT-1.5B JAX 推理性能测试报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            info = results.get('benchmark_info', {})
            f.write(f"测试时间: {info.get('timestamp', 'N/A')}\n")
            f.write(f"总执行时间: {info.get('total_execution_time', 0):.2f}秒\n")
            f.write(f"GPU数量: {info.get('gpu_count', 0)}\n")
            f.write(f"JAX版本: {info.get('jax_version', 'N/A')}\n")
            f.write(f"CUDA版本: {info.get('cuda_version', 'N/A')}\n")
            f.write(f"平台: {info.get('platform', 'N/A')}\n\n")
            
            # 模型配置
            model_cfg = info.get('model_config', {})
            f.write("模型配置:\n")
            for key, value in model_cfg.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 性能结果
            self._write_performance_results(f, results.get('results', {}))
    
    def _write_performance_results(self, f, results_data: Dict):
        """写入性能结果"""
        f.write("性能测试结果:\n")
        f.write("-" * 40 + "\n")
        
        all_throughputs = []
        all_latencies = []
        
        for config_id, result in results_data.items():
            f.write(f"\n配置 {config_id}:\n")
            f.write(f"  测试样本数: {result.get('samples_tested', 0)}\n")
            f.write(f"  平均推理时间: {result.get('avg_inference_time', 0):.3f}±{result.get('std_inference_time', 0):.3f}s\n")
            f.write(f"  平均吞吐量: {result.get('avg_throughput', 0):.1f} tokens/s\n")
            f.write(f"  吞吐量范围: {result.get('min_throughput', 0):.1f} - {result.get('max_throughput', 0):.1f} tokens/s\n")
            f.write(f"  平均每token延迟: {result.get('avg_latency_per_token', 0):.4f}s\n")
            f.write(f"  总生成token数: {result.get('total_tokens_generated', 0)}\n")
            
            all_throughputs.append(result.get('avg_throughput', 0))
            all_latencies.append(result.get('avg_latency_per_token', 0))
        
        # 总体统计
        if all_throughputs:
            f.write(f"\n总体性能统计:\n")
            f.write(f"  平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s\n")
            f.write(f"  最高吞吐量: {max(all_throughputs):.1f} tokens/s\n")
            f.write(f"  平均延迟: {np.mean(all_latencies):.4f}s/token\n")
            f.write(f"  最低延迟: {min(all_latencies):.4f}s/token\n")
    
    def load_results(self, timestamp: str) -> Optional[Dict]:
        """加载指定时间戳的结果"""
        if timestamp in self.results_cache:
            return self.results_cache[timestamp]
        
        # 尝试从文件加载
        json_file = self.output_dir / f"gpt15b_benchmark_{timestamp}.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                self.results_cache[timestamp] = results
                return results
        
        return None
    
    def list_available_results(self) -> List[str]:
        """列出所有可用的结果文件"""
        json_files = list(self.output_dir.glob("gpt15b_benchmark_*.json"))
        timestamps = []
        for file in json_files:
            # 提取时间戳
            parts = file.stem.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:])
                timestamps.append(timestamp)
        return sorted(timestamps)
    
    def compare_results(self, timestamp1: str, timestamp2: str) -> Dict:
        """比较两次测试结果"""
        result1 = self.load_results(timestamp1)
        result2 = self.load_results(timestamp2)
        
        if not result1 or not result2:
            return {}
        
        comparison = {
            'timestamp1': timestamp1,
            'timestamp2': timestamp2,
            'improvements': {},
            'regressions': {}
        }
        
        # 比较各配置的性能
        for config_id in result1.get('results', {}):
            if config_id in result2.get('results', {}):
                throughput1 = result1['results'][config_id].get('avg_throughput', 0)
                throughput2 = result2['results'][config_id].get('avg_throughput', 0)
                
                improvement = (throughput2 - throughput1) / throughput1 * 100 if throughput1 > 0 else 0
                
                if improvement > 5:  # 5%以上改进
                    comparison['improvements'][config_id] = improvement
                elif improvement < -5:  # 5%以上退化
                    comparison['regressions'][config_id] = improvement
        
        return comparison


def save_benchmark_results(results: Dict, output_dir: str = 'results') -> tuple:
    """保存基准测试结果（向后兼容的函数）"""
    manager = ResultManager(output_dir)
    return manager.save_results(results)


def print_performance_summary(results: Dict):
    """打印性能摘要"""
    print(f"\n🎉 基准测试完成!")
    print(f"📊 测试配置数: {len(results.get('results', {}))}")
    print(f"⏱️ 总执行时间: {results.get('benchmark_info', {}).get('total_execution_time', 0):.2f}秒")
    
    # 显示性能亮点
    results_data = results.get('results', {})
    if results_data:
        all_throughputs = [r.get('avg_throughput', 0) for r in results_data.values()]
        all_latencies = [r.get('avg_latency_per_token', 0) for r in results_data.values()]
        
        if all_throughputs:
            print(f"\n🏆 性能亮点:")
            print(f"   最高吞吐量: {max(all_throughputs):.1f} tokens/s")
            print(f"   最低延迟: {min(all_latencies):.4f}s/token")
            print(f"   平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s")
