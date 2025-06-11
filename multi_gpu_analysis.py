#!/usr/bin/env python3
"""
多GPU JAX并行化分析脚本
专门用于分析和优化GPT-1.5B JAX推理系统的多GPU并行化策略
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import partial

print("🚀 Multi-GPU JAX Parallelization Analysis")
print("=" * 60)

# 设置JAX环境（必须在导入JAX之前）
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# 清理可能存在的XLA_FLAGS
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

# 导入JAX相关包
try:
    import jax
    import jax.numpy as jnp
    from jax import random, pmap, devices, device_count
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    import flax.linen as nn
    from flax import jax_utils
    import numpy as np
    
    print(f"✅ JAX {jax.__version__} 加载成功")
    print(f"🖥️ 检测到 {len(jax.devices())} 个设备")
    for i, device in enumerate(jax.devices()):
        print(f"   设备 {i}: {device}")
        
    if jax.devices()[0].platform == 'gpu':
        print("✅ CUDA支持已启用")
    else:
        print("⚠️ 未检测到CUDA支持")
        
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)

@dataclass
class ProgressiveGPTConfig:
    """渐进式GPT配置"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1
    use_bias: bool = True
    
    def get_param_count(self) -> int:
        """估算参数量"""
        embed_params = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        layer_params = (4 * self.n_embd * self.n_embd + 
                       8 * self.n_embd * self.n_embd + 
                       4 * self.n_embd)
        total_params = embed_params + self.n_layer * layer_params + self.vocab_size * self.n_embd
        return total_params

class ProgressiveMultiHeadAttention(nn.Module):
    """渐进式多头注意力"""
    config: ProgressiveGPTConfig
    
    @nn.compact
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # QKV投影
        qkv = nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias)(x)
        qkv = qkv.reshape(B, T, 3, self.config.n_head, C // self.config.n_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, nh, T, hs)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个的形状: (B, nh, T, hs)
        
        # 注意力权重 - 正确的转置维度
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(k.shape[-1]))
        
        # 应用因果掩码
        if mask is not None:
            att = jnp.where(mask, att, -jnp.inf)
        
        att = jax.nn.softmax(att, axis=-1)
        
        # 应用注意力到值
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        return nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)(y)

def analyze_gpu_setup():
    """分析当前GPU设置"""
    print("📊 Section 1: GPU Setup Analysis")
    print("-" * 40)
    
    devices = jax.devices()
    print(f"GPU Analysis:")
    print(f"  Total devices: {len(devices)}")
    if devices:
        print(f"  Device platform: {devices[0].platform}")
    
    # 获取设备内存信息
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        print(f"    Platform: {device.platform}")
        print(f"    Device kind: {device.device_kind}")
    
    return devices

def create_device_mesh(devices):
    """创建设备mesh用于并行化"""
    print("\n🔧 Section 2: Device Mesh Creation")
    print("-" * 40)
    
    if len(devices) == 1:
        print("⚠️ Only 1 device available, no parallelization possible")
        return None
    
    try:
        # 尝试创建2D mesh: (data_parallel, model_parallel)
        if len(devices) == 4:
            # 4个GPU: 2x2 mesh
            mesh_shape = (2, 2)
            mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            print(f"✅ Created 2x2 mesh with axes ('data', 'model')")
        elif len(devices) == 2:
            mesh_shape = (2, 1)
            mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            print(f"✅ Created 2x1 mesh with axes ('data', 'model')")
        else:
            # 对于其他情况，使用1D mesh
            mesh_devices = np.array(devices).reshape(-1, 1)
            mesh = Mesh(mesh_devices, axis_names=('data',))
            print(f"✅ Created 1D mesh with {len(devices)} devices")
        
        print(f"  Mesh shape: {mesh.shape}")
        print(f"  Mesh axis names: {mesh.axis_names}")
        return mesh
        
    except Exception as e:
        print(f"❌ Failed to create mesh: {e}")
        print(f"   Error details: {str(e)}")
        return None

def test_configurations():
    """测试不同的模型配置"""
    print("\n📋 Section 3: Model Configuration Testing")
    print("-" * 40)
    
    configs = {
        'small': ProgressiveGPTConfig(
            n_embd=512, n_layer=8, n_head=8, n_positions=512
        ),
        'medium': ProgressiveGPTConfig(
            n_embd=768, n_layer=12, n_head=12, n_positions=1024
        ),
        'large': ProgressiveGPTConfig(
            n_embd=1024, n_layer=24, n_head=16, n_positions=1024
        ),
        'xlarge': ProgressiveGPTConfig(
            n_embd=1280, n_layer=36, n_head=20, n_positions=1024
        )
    }
    
    # 打印配置信息
    for name, config in configs.items():
        param_count = config.get_param_count()
        print(f"{name.upper()} Config:")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Heads: {config.n_head}")
        print(f"  Max positions: {config.n_positions}")
        print(f"  Estimated params: {param_count:,} ({param_count/1e6:.1f}M)")
        print(f"  Estimated memory: {param_count * 4 / (1024**3):.2f} GB")
        print()
    
    return configs

def test_attention_mechanism(config):
    """测试注意力机制"""
    print("\n🧪 Section 4: Attention Mechanism Testing")
    print("-" * 40)
    
    def create_test_data(config, batch_size=2, seq_len=32):
        """创建测试数据"""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, config.n_embd))
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        return x, mask
    
    print(f"Testing attention mechanism...")
    
    # 创建模型和测试数据
    attention = ProgressiveMultiHeadAttention(config)
    x, mask = create_test_data(config)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    # 初始化参数
    key = jax.random.PRNGKey(42)
    params = attention.init(key, x, mask)
    
    # 计算参数量
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Attention params: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # 前向传播
    start_time = time.time()
    output = attention.apply(params, x, mask)
    jax.block_until_ready(output)
    forward_time = time.time() - start_time
    
    print(f"  Output shape: {output.shape}")
    print(f"  Forward pass time: {forward_time*1000:.2f}ms")
    
    # 验证输出
    try:
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        assert not jnp.any(jnp.isnan(output)), "Output contains NaN values"
        assert jnp.all(jnp.isfinite(output)), "Output contains infinite values"
        print(f"  ✅ Attention mechanism test passed!")
        return True, params, output
    except Exception as e:
        print(f"  ❌ Attention test failed: {e}")
        return False, None, None

def test_parallel_attention(config, mesh):
    """测试并行注意力"""
    print("\n🏃 Section 5: Parallel Attention Testing")
    print("-" * 40)
    
    if mesh is None:
        print("⚠️ No mesh available, skipping parallel benchmark")
        return None
    
    def create_sharded_attention(config, mesh):
        attention = ProgressiveMultiHeadAttention(config)
        input_sharding = NamedSharding(mesh, PartitionSpec('data', None, None))
        return attention, input_sharding
    
    # 创建分片模型
    attention, input_sharding = create_sharded_attention(config, mesh)
    
    with mesh:
        # 创建更大的测试数据来体现并行化优势
        batch_size = len(jax.devices()) * 4  # 每个设备处理4个batch
        seq_len = 128
        
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, config.n_embd))
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        
        print(f"  Parallel test data shape: {x.shape}")
        print(f"  Total elements: {x.size:,}")
        
        # 分片输入数据
        x_sharded = jax.device_put(x, input_sharding)
        
        # 初始化参数
        params = attention.init(key, x, mask)
        
        # JIT编译的并行前向传播
        @jax.jit
        def parallel_forward(params, x, mask):
            return attention.apply(params, x, mask)
        
        # 预热
        print(f"  Warming up JIT compilation...")
        for _ in range(3):
            output = parallel_forward(params, x_sharded, mask)
            jax.block_until_ready(output)
        
        # 基准测试
        print(f"  Running benchmark iterations...")
        times = []
        
        for i in range(5):
            start_time = time.time()
            output = parallel_forward(params, x_sharded, mask)
            jax.block_until_ready(output)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"    Run {i+1}: {(end_time - start_time)*1000:.2f}ms")
        
        results = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'throughput': (batch_size * seq_len) / np.mean(times),
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_devices': len(jax.devices())
        }
        
        print(f"\n📊 Parallel Attention Results:")
        print(f"  Mean time: {results['mean_time']*1000:.2f}ms ± {results['std_time']*1000:.2f}ms")
        print(f"  Throughput: {results['throughput']:.1f} tokens/s")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Devices used: {results['num_devices']}")
        
        return results

def analyze_memory_usage(config):
    """分析内存使用情况"""
    print("\n💾 Section 6: Memory Usage Analysis")
    print("-" * 40)
    
    # 计算理论内存需求
    param_count = config.get_param_count()
    param_memory_gb = param_count * 4 / (1024**3)  # float32
    
    print(f"Model Memory Requirements:")
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"  Parameter memory (FP32): {param_memory_gb:.2f} GB")
    print(f"  Parameter memory (FP16): {param_memory_gb/2:.2f} GB")
    
    # 计算激活内存 (batch_size=1, seq_len=1024)
    batch_size = 1
    seq_len = 1024
    
    # 注意力机制的内存需求
    attention_matrix_size = batch_size * config.n_head * seq_len * seq_len * 4  # bytes
    qkv_size = batch_size * seq_len * config.n_embd * 3 * 4  # bytes
    output_size = batch_size * seq_len * config.n_embd * 4  # bytes
    
    activation_memory_gb = (attention_matrix_size + qkv_size + output_size) / (1024**3)
    
    print(f"\nActivation Memory (batch=1, seq_len={seq_len}):")
    print(f"  Attention matrices: {attention_matrix_size/(1024**2):.1f} MB")
    print(f"  QKV tensors: {qkv_size/(1024**2):.1f} MB")
    print(f"  Output tensors: {output_size/(1024**2):.1f} MB")
    print(f"  Total activation memory: {activation_memory_gb*1000:.1f} MB")
    
    # GPU内存容量 (RTX 3090 = 24GB)
    gpu_memory_gb = 24
    total_memory_gb = param_memory_gb + activation_memory_gb
    
    print(f"\nGPU Memory Analysis (per RTX 3090):")
    print(f"  Available memory: {gpu_memory_gb} GB")
    print(f"  Required memory: {total_memory_gb:.2f} GB")
    print(f"  Memory utilization: {(total_memory_gb/gpu_memory_gb)*100:.1f}%")
    
    if total_memory_gb > gpu_memory_gb:
        print(f"  ⚠️ Model too large for single GPU!")
        print(f"  💡 Recommendation: Use model parallelism or reduce precision")
    else:
        print(f"  ✅ Model fits in single GPU memory")
    
    # 多GPU内存分析
    num_gpus = len(jax.devices())
    if num_gpus > 1:
        print(f"\nMulti-GPU Memory Analysis ({num_gpus} GPUs):")
        print(f"  Total GPU memory: {gpu_memory_gb * num_gpus} GB")
        print(f"  Memory per GPU (data parallel): {total_memory_gb:.2f} GB")
        print(f"  Memory per GPU (model parallel): {param_memory_gb/num_gpus + activation_memory_gb:.2f} GB")
        
        if param_memory_gb/num_gpus + activation_memory_gb < gpu_memory_gb:
            print(f"  ✅ Model parallelism viable")
        else:
            print(f"  ⚠️ Need more aggressive parallelization")
    
    return {
        'param_memory_gb': param_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_memory_gb': total_memory_gb,
        'gpu_memory_gb': gpu_memory_gb,
        'num_gpus': num_gpus,
        'memory_utilization': (total_memory_gb/gpu_memory_gb)*100
    }

def generate_recommendations(memory_analysis, parallel_results, num_gpus):
    """生成优化建议"""
    print("\n🎯 Section 7: Optimization Recommendations")
    print("-" * 40)
    
    recommendations = []
    
    # 1. 内存优化建议
    print(f"1. Memory Optimization:")
    if memory_analysis['memory_utilization'] > 80:
        print(f"   ⚠️ High memory utilization ({memory_analysis['memory_utilization']:.1f}%)")
        recommendations.extend([
            "Use mixed precision (FP16) to reduce memory by 50%",
            "Implement gradient checkpointing",
            "Use model parallelism to distribute parameters"
        ])
    else:
        print(f"   ✅ Memory utilization OK ({memory_analysis['memory_utilization']:.1f}%)")
        recommendations.append("Consider increasing batch size")
    
    for rec in recommendations[-3:]:
        print(f"     • {rec}")
    
    # 2. 并行化策略建议
    print(f"\n2. Parallelization Strategy:")
    if num_gpus > 1:
        print(f"   Available GPUs: {num_gpus}")
        if memory_analysis['total_memory_gb'] < memory_analysis['gpu_memory_gb']:
            print(f"   💡 Recommended: Data Parallelism")
            recommendations.extend([
                "Use data parallelism for better scaling",
                "Implement pmap for efficient batch processing"
            ])
        else:
            print(f"   💡 Recommended: Model Parallelism")
            recommendations.extend([
                "Shard attention heads across GPUs",
                "Partition MLP layers across devices"
            ])
    else:
        print(f"   Single GPU setup - focus on memory optimization")
    
    # 3. 具体实现建议
    print(f"\n3. Implementation Priorities:")
    priorities = [
        "HIGH: Fix mesh creation issues for multi-GPU support",
        "HIGH: Implement proper sharding for attention layers",
        "MEDIUM: Add mixed precision support (FP16/BF16)",
        "MEDIUM: Optimize memory allocation patterns",
        "LOW: Fine-tune XLA compilation flags"
    ]
    
    for priority in priorities:
        print(f"     • {priority}")
    
    return recommendations

def save_results(all_results):
    """保存分析结果"""
    print("\n💾 Section 8: Saving Results")
    print("-" * 40)
    
    results_file = Path("multi_gpu_analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Analysis results saved to: {results_file}")
    return results_file

def main():
    """主函数"""
    try:
        # 1. 分析GPU设置
        devices = analyze_gpu_setup()
        
        # 2. 创建设备mesh
        mesh = create_device_mesh(devices)
        
        # 3. 测试配置
        configs = test_configurations()
        test_config = configs['medium']
        print(f"🎯 Using MEDIUM config for testing")
        print(f"  Parameters: {test_config.get_param_count()/1e6:.1f}M")
        
        # 4. 测试注意力机制
        attention_success, params, output = test_attention_mechanism(test_config)
        
        # 5. 测试并行注意力
        parallel_results = None
        if mesh is not None:
            parallel_results = test_parallel_attention(test_config, mesh)
        
        # 6. 内存分析
        memory_analysis = analyze_memory_usage(test_config)
        
        # 为大模型分析
        print("\n" + "="*50)
        print("Large Model Analysis (1.5B parameters)")
        large_memory_analysis = analyze_memory_usage(configs['xlarge'])
        
        # 7. 生成建议
        recommendations = generate_recommendations(memory_analysis, parallel_results, len(jax.devices()))
        
        # 8. 保存结果
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_count": len(jax.devices()),
            "jax_version": jax.__version__,
            "mesh_created": mesh is not None,
            "attention_test_passed": attention_success,
            "memory_analysis": memory_analysis,
            "large_memory_analysis": large_memory_analysis,
            "parallel_results": parallel_results,
            "recommendations": recommendations
        }
        
        results_file = save_results(all_results)
        
        # 9. 最终总结
        print("\n🎯 Multi-GPU Analysis Complete!")
        print("=" * 60)
        print(f"📊 Key Findings:")
        print(f"  • GPU count: {len(jax.devices())}")
        print(f"  • Mesh creation: {'✅ Success' if mesh else '❌ Failed'}")
        print(f"  • Attention test: {'✅ Passed' if attention_success else '❌ Failed'}")
        print(f"  • Memory utilization: {memory_analysis['memory_utilization']:.1f}%")
        if parallel_results:
            print(f"  • Parallel throughput: {parallel_results['throughput']:.1f} tokens/s")
        print(f"  • Total recommendations: {len(recommendations)}")
        print(f"\n📋 Next Steps:")
        print(f"  1. Review detailed results in {results_file}")
        print(f"  2. Implement high-priority recommendations")
        print(f"  3. Test with larger models if mesh issues are resolved")
        
        if not mesh:
            print(f"\n⚠️ CRITICAL: Mesh creation failed!")
            print(f"   This is preventing multi-GPU utilization.")
            print(f"   Priority: Fix mesh creation issues first.")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
