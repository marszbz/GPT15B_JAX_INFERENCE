"""
推理性能基准测试模块
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.gpt_model import GraphPartitionedGPT
from src.data.dataset_loader import SimpleTokenizer


class InferenceBenchmark:
    """推理性能基准测试"""
    
    def __init__(self, model: GraphPartitionedGPT):
        self.model = model
        self.tokenizer = SimpleTokenizer()
        
    def benchmark_single_sample(self, sample: Dict) -> Dict:
        """测试单个样本的推理性能"""
        prompt = sample['prompt']
        generation_length = sample.get('generation_length', 32)
        sample_id = sample.get('id', 'unknown')
        
        # 编码输入
        prompt_tokens = self.tokenizer.encode(prompt, max_length=512)
        input_ids = jnp.array([prompt_tokens])  # 添加batch维度
        
        # 预热（确保JIT编译完成）
        _ = self.model.generate_text(input_ids, max_new_tokens=8)
        
        # 正式推理计时
        start_time = time.time()
        output_ids = self.model.generate_text(input_ids, max_new_tokens=generation_length)
        
        # 确保计算完成
        jax.block_until_ready(output_ids)
        end_time = time.time()
        
        # 计算性能指标
        inference_time = end_time - start_time
        input_length = len([t for t in prompt_tokens if t != 0])  # 去除padding
        total_tokens = output_ids.shape[1]
        generated_tokens = total_tokens - input_length
        
        throughput = generated_tokens / inference_time if inference_time > 0 else 0
        
        return {
            'sample_id': sample_id,
            'input_length': input_length,
            'generated_tokens': generated_tokens,
            'total_tokens': total_tokens,
            'inference_time': inference_time,
            'throughput_tokens_per_sec': throughput,
            'latency_per_token': inference_time / generated_tokens if generated_tokens > 0 else 0
        }
    
    def benchmark_config(self, dataset: List[Dict], config_id: str, max_samples: int = 10) -> Dict:
        """测试特定配置的性能"""
        if not dataset:
            return {}
        
        print(f"\n🧪 测试配置 {config_id}")
        print(f"   数据集大小: {len(dataset)} 个样本")
        print(f"   测试样本数: {min(max_samples, len(dataset))}")
        
        results = []
        test_samples = min(max_samples, len(dataset))
        
        for i in range(test_samples):
            sample = dataset[i]
            
            try:
                result = self.benchmark_single_sample(sample)
                results.append(result)
                
                # 每3个样本显示一次进度
                if (i + 1) % 3 == 0:
                    recent_results = results[-3:]
                    avg_time = np.mean([r['inference_time'] for r in recent_results])
                    avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in recent_results])
                    print(f"   进度 {i+1}/{test_samples}: 平均延迟 {avg_time:.3f}s, "
                          f"平均吞吐量 {avg_throughput:.1f} tokens/s")
                
            except Exception as e:
                print(f"⚠️ 样本 {i} 测试失败: {e}")
                continue
        
        # 计算统计信息
        if not results:
            return {}
        
        inference_times = [r['inference_time'] for r in results]
        throughputs = [r['throughput_tokens_per_sec'] for r in results]
        latencies = [r['latency_per_token'] for r in results]
        
        summary = {
            'config_id': config_id,
            'samples_tested': len(results),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs),
            'avg_latency_per_token': np.mean(latencies),
            'total_tokens_generated': sum(r['generated_tokens'] for r in results),
            'total_time': sum(inference_times),
            'detailed_results': results
        }
        
        print(f"✅ 配置 {config_id} 测试完成:")
        print(f"   平均推理时间: {summary['avg_inference_time']:.3f}±{summary['std_inference_time']:.3f}s")
        print(f"   平均吞吐量: {summary['avg_throughput']:.1f} tokens/s")
        print(f"   平均每token延迟: {summary['avg_latency_per_token']:.4f}s")
        
        return summary
    
    def run_full_benchmark(self, datasets: Dict[str, List[Dict]], max_samples_per_config: int = 10) -> Dict:
        """运行完整的基准测试"""
        print("\n🚀 开始GPT-1.5B JAX推理性能测试")
        print("=" * 60)
        print(f"📊 数据集配置数: {len(datasets)}")
        print(f"🔧 GPU数量: {len(jax.devices())}")
        print(f"📦 JAX版本: {jax.__version__}")
        print(f"🏗️ 模型规模: {self.model.config.n_layer}层, {self.model.config.n_head}头, {self.model.config.n_embd}维")
        
        all_results = {}
        total_start_time = time.time()
        
        # 逐个测试配置
        for config_id, dataset in datasets.items():
            config_result = self.benchmark_config(dataset, config_id, max_samples_per_config)
            if config_result:
                all_results[config_id] = config_result
        
        total_time = time.time() - total_start_time
        
        # 生成综合报告
        benchmark_summary = {
            'benchmark_info': {
                'total_execution_time': total_time,
                'configs_tested': len(all_results),
                'gpu_count': len(jax.devices()),
                'jax_version': jax.__version__,
                'model_config': {
                    'n_layer': self.model.config.n_layer,
                    'n_head': self.model.config.n_head,
                    'n_embd': self.model.config.n_embd,
                    'vocab_size': self.model.config.vocab_size
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'platform': 'Windows',
                'cuda_version': '11.8'
            },
            'results': all_results
        }
        
        return benchmark_summary
