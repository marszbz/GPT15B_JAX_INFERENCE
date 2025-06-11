#!/usr/bin/env python3
"""
终极优化GPT-1.5B JAX推理系统
基于成功的大型模型测试，进一步优化性能
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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # 增加内存使用
os.environ['JAX_ENABLE_X64'] = 'false'  # 保持FP32精度
# 优化XLA编译
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
class UltimateGPTConfig:
    """终极优化GPT配置"""
    vocab_size: int = 50257
    n_positions: int = 2048  # 增加上下文长度
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.0  # 推理时关闭dropout
    use_bias: bool = True
    
    # 新增优化参数
    use_flash_attention: bool = True
    use_mixed_precision: bool = False  # 暂时保持FP32
    batch_size_multiplier: int = 2  # 批次大小倍数
    
    def get_param_count(self) -> int:
        """计算参数量"""
        embed_params = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        layer_params = (4 * self.n_embd * self.n_embd + 8 * self.n_embd * self.n_embd + 4 * self.n_embd)
        total_params = embed_params + self.n_layer * layer_params + self.vocab_size * self.n_embd
        return total_params

class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力机制"""
    config: UltimateGPTConfig
    
    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
    def __call__(self, x, mask=None, training=False):
        B, T, C = x.shape
        
        # 单次QKV投影，然后分割
        qkv = self.c_attn(x)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        
        # 重排维度: (B, T, 1, nh, hd) -> (B, nh, T, hd)
        q = q.squeeze(2).transpose(0, 2, 1, 3)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)
        
        # 优化的注意力计算
        if self.config.use_flash_attention:
            # 简化的flash attention风格计算
            scale = 1.0 / jnp.sqrt(self.head_dim)
            
            # 分块计算注意力以节省内存
            chunk_size = min(256, T)
            outputs = []
            
            for i in range(0, T, chunk_size):
                end_i = min(i + chunk_size, T)
                q_chunk = q[:, :, i:end_i, :]
                
                # 计算注意力分数
                scores = jnp.einsum('bnid,bnjd->bnij', q_chunk, k) * scale
                
                # 应用因果掩码
                if mask is not None:
                    mask_chunk = mask[:, :, i:end_i, :]
                    scores = jnp.where(mask_chunk, scores, -jnp.inf)
                
                # 注意力权重和输出
                attn_weights = jax.nn.softmax(scores, axis=-1)
                output_chunk = jnp.einsum('bnij,bnjd->bnid', attn_weights, v)
                outputs.append(output_chunk)
            
            y = jnp.concatenate(outputs, axis=2)
        else:
            # 标准注意力计算
            att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(self.head_dim))
            if mask is not None:
                att = jnp.where(mask, att, -jnp.inf)
            att = jax.nn.softmax(att, axis=-1)
            y = att @ v
        
        # 重新整理输出
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)

class OptimizedMLP(nn.Module):
    """优化的MLP"""
    config: UltimateGPTConfig
    
    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
    
    def __call__(self, x, training=False):
        # 使用GELU激活函数
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)  # 近似GELU更快
        x = self.c_proj(x)
        return x

class OptimizedTransformerBlock(nn.Module):
    """优化的Transformer块"""
    config: UltimateGPTConfig
    
    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = OptimizedMultiHeadAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = OptimizedMLP(self.config)
    
    def __call__(self, x, mask=None, training=False):
        # Pre-norm架构
        x = x + self.attn(self.ln_1(x), mask, training)
        x = x + self.mlp(self.ln_2(x), training)
        return x

class UltimateGPT(nn.Module):
    """终极优化GPT模型"""
    config: UltimateGPTConfig
    
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.n_positions, self.config.n_embd)
        self.h = [OptimizedTransformerBlock(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm()
        
        # 预计算位置编码
        self.pos_cache = jnp.arange(self.config.n_positions)
    
    def __call__(self, input_ids, training=False):
        B, T = input_ids.shape
        
        # Token和位置嵌入
        token_embed = self.wte(input_ids)
        pos_ids = self.pos_cache[:T][None, :]
        pos_embed = self.wpe(pos_ids)
        
        x = token_embed + pos_embed
        
        # 预计算因果掩码
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        
        # Transformer层
        for block in self.h:
            x = block(x, mask, training)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头（重用词嵌入权重）
        logits = x @ self.wte.embedding.T
        
        return logits

class UltimateInferenceEngine:
    """终极推理引擎"""
    
    def __init__(self, config: UltimateGPTConfig):
        self.config = config
        self.devices = jax.devices()
        
        print(f"🚀 初始化终极GPT推理引擎")
        print(f"   设备数量: {len(self.devices)}")
        print(f"   配置: {config.n_layer}层, {config.n_embd}维度, {config.n_head}头")
        
        # 创建设备mesh
        self.mesh = self._create_mesh()
        
        # 初始化模型
        self.model = UltimateGPT(config)
        self._init_parameters()
        
        # 编译优化函数
        self._compile_functions()
    
    def _create_mesh(self):
        """创建优化的设备mesh"""
        num_devices = len(self.devices)
        if num_devices >= 4:
            # 2x2 mesh for optimal parallelization
            mesh_devices = mesh_utils.create_device_mesh((2, 2))
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            print(f"   Mesh: 2x2 ('data', 'model')")
        elif num_devices == 2:
            mesh_devices = mesh_utils.create_device_mesh((2, 1))
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            print(f"   Mesh: 2x1 ('data', 'model')")
        else:
            mesh = None
            print(f"   Mesh: 单设备模式")
        
        return mesh
    
    def _init_parameters(self):
        """初始化模型参数"""
        print("🔄 初始化模型参数...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        
        start_time = time.time()
        self.params = self.model.init(key, dummy_input, training=False)
        init_time = time.time() - start_time
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        param_memory = param_count * 4 / (1024**3)
        
        print(f"   参数量: {param_count:,} ({param_count/1e6:.1f}M)")
        print(f"   内存使用: {param_memory:.2f} GB")
        print(f"   初始化时间: {init_time:.2f}s")
    
    def _compile_functions(self):
        """编译优化函数"""
        print("⚡ 编译优化函数...")
        
        if self.mesh:
            # 分片推理函数
            input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            
            @partial(jax.jit, in_shardings=(None, input_sharding), out_shardings=input_sharding)
            def sharded_forward(params, input_ids):
                return self.model.apply(params, input_ids, training=False)
            
            self.forward_fn = sharded_forward
            print("   分片推理函数已编译")
        else:
            # 单设备推理函数
            @jax.jit
            def single_forward(params, input_ids):
                return self.model.apply(params, input_ids, training=False)
            
            self.forward_fn = single_forward
            print("   单设备推理函数已编译")
    
    def benchmark_ultimate_performance(self, batch_size=None, seq_len=512, num_runs=10):
        """终极性能基准测试"""
        if batch_size is None:
            batch_size = len(self.devices) * self.config.batch_size_multiplier
        
        print(f"\n🏆 终极性能基准测试")
        print(f"   批次大小: {batch_size}")
        print(f"   序列长度: {seq_len}")
        print(f"   总tokens: {batch_size * seq_len:,}")
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(key, (batch_size, seq_len), 0, self.config.vocab_size)
        
        if self.mesh:
            with self.mesh:
                input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
                input_ids = jax.device_put(input_ids, input_sharding)
        
        # 预热
        print("🔥 预热编译...")
        for i in range(3):
            logits = self.forward_fn(self.params, input_ids)
            jax.block_until_ready(logits)
            print(f"   预热 {i+1}/3 完成")
        
        # 基准测试
        print(f"🚀 运行 {num_runs} 次基准测试...")
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            logits = self.forward_fn(self.params, input_ids)
            jax.block_until_ready(logits)
            end_time = time.time()
            
            time_taken = end_time - start_time
            throughput = (batch_size * seq_len) / time_taken
            times.append(time_taken)
            
            print(f"   Run {i+1}: {time_taken*1000:.2f}ms, {throughput:.1f} tokens/s")
        
        # 计算统计信息
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_throughput = (batch_size * seq_len) / mean_time
        peak_throughput = (batch_size * seq_len) / min(times)
        
        results = {
            'config': {
                'n_embd': self.config.n_embd,
                'n_layer': self.config.n_layer,
                'n_head': self.config.n_head,
                'params': self.config.get_param_count()
            },
            'performance': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'total_tokens': batch_size * seq_len,
                'mean_time': mean_time,
                'std_time': std_time,
                'mean_throughput': mean_throughput,
                'peak_throughput': peak_throughput,
                'devices': len(self.devices)
            },
            'times': times
        }
        
        print(f"\n🎯 终极性能结果:")
        print(f"   平均时间: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"   平均吞吐量: {mean_throughput:.1f} tokens/s")
        print(f"   峰值吞吐量: {peak_throughput:.1f} tokens/s")
        print(f"   参数量: {self.config.get_param_count()/1e6:.1f}M")
        print(f"   设备利用: {len(self.devices)}x RTX 3090")
        
        return results

def main():
    """主函数"""
    print("🚀 终极优化GPT-1.5B JAX推理系统")
    print("=" * 60)
    print(f"📦 JAX版本: {jax.__version__}")
    print(f"🖥️ 设备数量: {len(jax.devices())}")
    
    # 创建终极配置
    config = UltimateGPTConfig()
    print(f"\n📋 终极配置:")
    print(f"   参数量: {config.get_param_count()/1e6:.1f}M")
    print(f"   上下文长度: {config.n_positions}")
    print(f"   Flash注意力: {config.use_flash_attention}")
    
    try:
        # 创建推理引擎
        engine = UltimateInferenceEngine(config)
        
        # 运行多种批次大小的测试
        batch_sizes = [4, 8, 16, 32]
        all_results = []
        
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"测试批次大小: {batch_size}")
            
            try:
                results = engine.benchmark_ultimate_performance(
                    batch_size=batch_size, 
                    seq_len=512, 
                    num_runs=5
                )
                all_results.append(results)
                
                # 如果内存不足，停止增加批次大小
                if results['performance']['mean_time'] > 30:  # 超过30秒就停止
                    print("⚠️ 时间过长，停止增加批次大小")
                    break
                    
            except Exception as e:
                print(f"❌ 批次大小 {batch_size} 测试失败: {e}")
                break
        
        # 保存所有结果
        if all_results:
            results_file = Path("ultimate_performance_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            print(f"\n💾 结果已保存到: {results_file}")
            
            # 找出最佳性能
            best_result = max(all_results, key=lambda x: x['performance']['peak_throughput'])
            best_throughput = best_result['performance']['peak_throughput']
            best_batch = best_result['performance']['batch_size']
            
            print(f"\n🏆 最佳性能:")
            print(f"   峰值吞吐量: {best_throughput:.1f} tokens/s")
            print(f"   最佳批次大小: {best_batch}")
            print(f"   模型规模: {config.get_param_count()/1e6:.1f}M参数")
            print(f"   多GPU加速: {len(jax.devices())}x RTX 3090")
            
        print(f"\n✅ 终极优化测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
