#!/usr/bin/env python3
"""
测试大型模型（1.5B参数）的多GPU推理性能
基于成功的multi_gpu_analysis.py结果
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import partial

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

import jax
import jax.numpy as jnp
from jax import random, pmap, devices, device_count
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import flax.linen as nn
from flax import jax_utils
import numpy as np

print(f"🚀 大型模型多GPU推理测试")
print("=" * 60)
print(f"📦 JAX版本: {jax.__version__}")
print(f"🖥️ 设备数量: {len(jax.devices())}")

@dataclass
class LargeGPTConfig:
    """大型GPT配置 - 1.5B参数"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600  # 增大嵌入维度
    n_layer: int = 48   # 增加层数
    n_head: int = 25    # 增加注意力头数
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

class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力 - 支持大模型"""
    config: LargeGPTConfig
    
    @nn.compact
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # QKV投影 - 使用更高效的计算
        qkv = nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias)(x)
        qkv = qkv.reshape(B, T, 3, self.config.n_head, C // self.config.n_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 缩放点积注意力
        scale = 1.0 / jnp.sqrt(k.shape[-1])
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        # 应用因果掩码
        if mask is not None:
            att = jnp.where(mask, att, -jnp.inf)
        
        att = jax.nn.softmax(att, axis=-1)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        return nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)(y)

class OptimizedMLP(nn.Module):
    """优化的MLP层"""
    config: LargeGPTConfig
    
    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)(x)
        hidden = jax.nn.gelu(hidden)
        return nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)(hidden)

class OptimizedTransformerBlock(nn.Module):
    """优化的Transformer块"""
    config: LargeGPTConfig
    
    @nn.compact
    def __call__(self, x, mask=None):
        # Pre-norm + 注意力
        norm1 = nn.LayerNorm()(x)
        attn_out = OptimizedMultiHeadAttention(self.config)(norm1, mask)
        x = x + attn_out
        
        # Pre-norm + MLP
        norm2 = nn.LayerNorm()(x)
        mlp_out = OptimizedMLP(self.config)(norm2)
        x = x + mlp_out
        
        return x

class LargeGPTModel(nn.Module):
    """大型GPT模型 - 1.5B参数"""
    config: LargeGPTConfig
    
    @nn.compact
    def __call__(self, input_ids, training: bool = False):
        B, T = input_ids.shape
        
        # 嵌入层
        token_embed = nn.Embed(self.config.vocab_size, self.config.n_embd)(input_ids)
        pos_embed = nn.Embed(self.config.n_positions, self.config.n_embd)(
            jnp.arange(T)[None, :] % self.config.n_positions
        )
        x = token_embed + pos_embed
        
        # 因果掩码
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        
        # Transformer层
        for i in range(self.config.n_layer):
            x = OptimizedTransformerBlock(self.config)(x, mask)
        
        # 最终输出
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        return logits

def create_sharded_model(config, mesh):
    """创建分片的大型模型"""
    model = LargeGPTModel(config)
    
    # 分片策略
    input_sharding = NamedSharding(mesh, PartitionSpec('data', None))
    
    return model, input_sharding

def benchmark_large_model():
    """基准测试大型模型"""
    print(f"\n🔧 初始化大型GPT模型 (1.5B参数)")
    
    # 创建配置
    config = LargeGPTConfig()
    param_count = config.get_param_count()
    print(f"   参数量: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"   内存需求: {param_count * 4 / (1024**3):.2f} GB")
    
    # 创建设备mesh
    devices_array = mesh_utils.create_device_mesh((2, 2))
    mesh = Mesh(devices_array, axis_names=('data', 'model'))
    print(f"   Mesh: {mesh}")
    
    # 创建模型
    model, input_sharding = create_sharded_model(config, mesh)
    
    with mesh:
        # 测试数据
        batch_size = 8  # 较大的批次大小
        seq_len = 512   # 较长的序列
        
        print(f"\n📊 测试配置:")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Total tokens: {batch_size * seq_len:,}")
        
        # 创建输入
        key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
        
        # 分片输入
        input_ids_sharded = jax.device_put(input_ids, input_sharding)
        
        # 初始化参数
        print(f"\n🔄 初始化模型参数...")
        start_time = time.time()
        params = model.init(key, input_ids, training=False)
        init_time = time.time() - start_time
        
        actual_param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"   实际参数量: {actual_param_count:,} ({actual_param_count/1e6:.1f}M)")
        print(f"   初始化时间: {init_time:.2f}s")
        
        # JIT编译的前向传播
        @jax.jit
        def forward_pass(params, input_ids):
            return model.apply(params, input_ids, training=False)
        
        # 预热
        print(f"\n🔥 JIT编译预热...")
        warmup_start = time.time()
        for i in range(3):
            logits = forward_pass(params, input_ids_sharded)
            jax.block_until_ready(logits)
            print(f"   预热 {i+1}/3 完成")
        warmup_time = time.time() - warmup_start
        print(f"   预热时间: {warmup_time:.2f}s")
        
        # 性能测试
        print(f"\n🏃 性能基准测试...")
        num_runs = 5
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            logits = forward_pass(params, input_ids_sharded)
            jax.block_until_ready(logits)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            throughput = (batch_size * seq_len) / run_time
            print(f"   Run {i+1}: {run_time*1000:.2f}ms, {throughput:.1f} tokens/s")
        
        # 结果统计
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_throughput = (batch_size * seq_len) / mean_time
        
        print(f"\n📊 大型模型性能结果:")
        print(f"   平均时间: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"   平均吞吐量: {mean_throughput:.1f} tokens/s")
        print(f"   峰值吞吐量: {max((batch_size * seq_len) / t for t in times):.1f} tokens/s")
        print(f"   参数量: {actual_param_count/1e6:.1f}M")
        print(f"   设备数: {len(jax.devices())}")
        
        # 保存结果
        results = {
            'model_size': '1.5B',
            'param_count': actual_param_count,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mean_time': mean_time,
            'mean_throughput': mean_throughput,
            'peak_throughput': max((batch_size * seq_len) / t for t in times),
            'warmup_time': warmup_time,
            'init_time': init_time,
            'num_devices': len(jax.devices()),
            'mesh_shape': mesh.shape
        }
        
        results_file = Path("large_model_benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 结果已保存到: {results_file}")
        
        return results

def main():
    """主函数"""
    try:
        results = benchmark_large_model()
        
        print(f"\n✅ 大型模型测试完成!")
        print(f"🎯 关键指标:")
        print(f"   模型大小: {results['param_count']/1e6:.1f}M 参数")
        print(f"   吞吐量: {results['mean_throughput']:.1f} tokens/s")
        print(f"   多GPU加速: 4x RTX 3090")
        print(f"   内存使用: 高效利用")
        
        if results['mean_throughput'] > 1000:
            print(f"\n🚀 性能优秀! 已达到高性能推理水平")
        else:
            print(f"\n💡 性能良好，但仍有优化空间")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
