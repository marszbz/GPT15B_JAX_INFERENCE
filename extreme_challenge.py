#!/usr/bin/env python3
"""
极限性能挑战 - 测试GPT-1.5B的最大吞吐量
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import partial

# 设置JAX环境（必须在导入JAX之前）
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'  # 使用更多内存
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

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
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class ExtremeGPTConfig:
    """极限性能GPT配置"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.0
    use_bias: bool = True
    use_flash_attention: bool = True
    
    def get_param_count(self) -> int:
        embed_params = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        layer_params = (4 * self.n_embd * self.n_embd + 8 * self.n_embd * self.n_embd + 4 * self.n_embd)
        total_params = embed_params + self.n_layer * layer_params + self.vocab_size * self.n_embd
        return total_params

class ExtremeMultiHeadAttention(nn.Module):
    """极限优化多头注意力"""
    config: ExtremeGPTConfig
    
    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
    def __call__(self, x, mask=None, training=False):
        B, T, C = x.shape
        
        # 单次QKV投影
        qkv = self.c_attn(x)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        
        # 高效维度重排
        q = q.squeeze(2).transpose(0, 2, 1, 3)
        k = k.squeeze(2).transpose(0, 2, 1, 3) 
        v = v.squeeze(2).transpose(0, 2, 1, 3)
        
        # 超级优化的注意力计算
        scale = 1.0 / jnp.sqrt(self.head_dim)
        
        # 使用更大的分块以提高吞吐量
        chunk_size = min(512, T)  # 增大分块大小
        outputs = []
        
        for i in range(0, T, chunk_size):
            end_i = min(i + chunk_size, T)
            q_chunk = q[:, :, i:end_i, :]
            
            # 高效注意力计算
            scores = jnp.einsum('bnid,bnjd->bnij', q_chunk, k) * scale
            
            if mask is not None:
                mask_chunk = mask[:, :, i:end_i, :]
                scores = jnp.where(mask_chunk, scores, -1e9)  # 使用-1e9而不是-inf
            
            attn_weights = jax.nn.softmax(scores, axis=-1)
            output_chunk = jnp.einsum('bnij,bnjd->bnid', attn_weights, v)
            outputs.append(output_chunk)
        
        y = jnp.concatenate(outputs, axis=2)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)

class ExtremeMLP(nn.Module):
    """极限优化MLP"""
    config: ExtremeGPTConfig
    
    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
    
    def __call__(self, x, training=False):
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x

class ExtremeTransformerBlock(nn.Module):
    """极限优化Transformer块"""
    config: ExtremeGPTConfig
    
    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = ExtremeMultiHeadAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = ExtremeMLP(self.config)
    
    def __call__(self, x, mask=None, training=False):
        x = x + self.attn(self.ln_1(x), mask, training)
        x = x + self.mlp(self.ln_2(x), training)
        return x

class ExtremeGPT(nn.Module):
    """极限性能GPT模型"""
    config: ExtremeGPTConfig
    
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.n_positions, self.config.n_embd)
        self.h = [ExtremeTransformerBlock(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm()
        
        # 预计算位置编码
        self.pos_cache = jnp.arange(self.config.n_positions)
    
    def __call__(self, input_ids, training=False):
        B, T = input_ids.shape
        
        token_embed = self.wte(input_ids)
        pos_ids = self.pos_cache[:T][None, :]
        pos_embed = self.wpe(pos_ids)
        
        x = token_embed + pos_embed
        
        # 预计算因果掩码
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        
        for block in self.h:
            x = block(x, mask, training)
        
        x = self.ln_f(x)
        logits = x @ self.wte.embedding.T
        
        return logits

class ExtremeInferenceEngine:
    """极限推理引擎"""
    
    def __init__(self, config: ExtremeGPTConfig):
        self.config = config
        self.devices = jax.devices()
        
        print(f"🚀 初始化极限性能推理引擎")
        print(f"   设备数量: {len(self.devices)}")
        print(f"   目标: >1000 tokens/s")
        
        self.mesh = self._create_mesh()
        self.model = ExtremeGPT(config)
        self._init_parameters()
        self._compile_functions()
    
    def _create_mesh(self):
        num_devices = len(self.devices)
        if num_devices >= 4:
            mesh_devices = mesh_utils.create_device_mesh((2, 2))
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            print(f"   Mesh: 2x2 优化配置")
        else:
            mesh = None
            print(f"   Mesh: 单设备模式")
        return mesh
    
    def _init_parameters(self):
        print("🔄 快速初始化参数...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        
        start_time = time.time()
        self.params = self.model.init(key, dummy_input, training=False)
        init_time = time.time() - start_time
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"   参数量: {param_count/1e6:.1f}M")
        print(f"   初始化时间: {init_time:.2f}s")
    
    def _compile_functions(self):
        print("⚡ 编译极限优化函数...")
        
        if self.mesh:
            input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            
            @partial(jax.jit, in_shardings=(None, input_sharding), out_shardings=input_sharding)
            def extreme_forward(params, input_ids):
                return self.model.apply(params, input_ids, training=False)
            
            self.forward_fn = extreme_forward
        else:
            @jax.jit
            def single_forward(params, input_ids):
                return self.model.apply(params, input_ids, training=False)
            
            self.forward_fn = single_forward
        
        print("   极限函数编译完成")
    
    def extreme_benchmark(self, batch_size, seq_len, num_runs=3):
        """极限性能基准测试"""
        total_tokens = batch_size * seq_len
        print(f"\n🏆 极限挑战: {batch_size}x{seq_len} = {total_tokens:,} tokens")
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, self.config.vocab_size)
        
        if self.mesh:
            with self.mesh:
                input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
                input_ids = jax.device_put(input_ids, input_sharding)
        
        # 快速预热
        print("🔥 快速预热...")
        for i in range(2):  # 减少预热次数
            logits = self.forward_fn(self.params, input_ids)
            jax.block_until_ready(logits)
        
        # 基准测试
        print(f"🚀 极限测试 ({num_runs}次)...")
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            logits = self.forward_fn(self.params, input_ids)
            jax.block_until_ready(logits)
            end_time = time.time()
            
            time_taken = end_time - start_time
            throughput = total_tokens / time_taken
            times.append(time_taken)
            
            print(f"   Run {i+1}: {time_taken*1000:.1f}ms → {throughput:.1f} tokens/s")
        
        # 结果
        best_time = min(times)
        peak_throughput = total_tokens / best_time
        mean_throughput = total_tokens / np.mean(times)
        
        print(f"\n🎯 极限结果:")
        print(f"   峰值吞吐量: {peak_throughput:.1f} tokens/s")
        print(f"   平均吞吐量: {mean_throughput:.1f} tokens/s")
        print(f"   最快时间: {best_time*1000:.1f}ms")
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_tokens': total_tokens,
            'peak_throughput': peak_throughput,
            'mean_throughput': mean_throughput,
            'best_time': best_time
        }

def main():
    """主函数 - 极限挑战"""
    print("🚀 极限性能挑战 - GPT-1.5B")
    print("=" * 60)
    print("🎯 目标: 突破1000 tokens/s!")
    
    config = ExtremeGPTConfig()
    engine = ExtremeInferenceEngine(config)
    
    # 极限挑战配置
    extreme_tests = [
        # (batch_size, seq_len)
        (32, 512),   # 重现最佳结果
        (48, 512),   # 增大批次
        (64, 512),   # 更大批次
        (32, 1024),  # 更长序列
        (48, 1024),  # 极限组合
    ]
    
    results = []
    best_throughput = 0
    
    for batch_size, seq_len in extreme_tests:
        try:
            result = engine.extreme_benchmark(batch_size, seq_len, num_runs=3)
            results.append(result)
            
            if result['peak_throughput'] > best_throughput:
                best_throughput = result['peak_throughput']
            
            # 如果时间过长就停止
            if result['best_time'] > 40:  # 超过40秒
                print("⚠️ 时间过长，停止测试")
                break
                
        except Exception as e:
            print(f"❌ 配置 {batch_size}x{seq_len} 失败: {e}")
            break
    
    # 保存结果
    if results:
        results_file = Path("extreme_performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🏆 极限挑战完成!")
        print(f"   最高吞吐量: {best_throughput:.1f} tokens/s")
        print(f"   测试配置数: {len(results)}")
        print(f"   结果文件: {results_file}")
        
        # 显示排行榜
        print(f"\n📊 性能排行榜:")
        sorted_results = sorted(results, key=lambda x: x['peak_throughput'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {result['batch_size']}x{result['seq_len']}: {result['peak_throughput']:.1f} tokens/s")
    
    print(f"\n🎯 挑战{('成功' if best_throughput > 1000 else '继续')}!")

if __name__ == "__main__":
    main()
