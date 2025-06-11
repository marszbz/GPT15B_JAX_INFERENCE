#!/usr/bin/env python3
"""
渐进式GPT-1.5B JAX推理测试
逐步增加模型规模，避免死锁问题
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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# 清理可能存在的XLA_FLAGS
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

# 导入JAX相关包
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import numpy as np
    print(f"✅ JAX {jax.__version__} 加载成功")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)


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
        # 嵌入层: vocab_size * n_embd + n_positions * n_embd
        embed_params = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        
        # 每个transformer层的参数
        # 注意力: 4 * n_embd^2 (qkv + output projection)
        # MLP: 2 * n_embd * (4 * n_embd) = 8 * n_embd^2
        # LayerNorm: 2 * n_embd * 2 = 4 * n_embd
        layer_params = (4 * self.n_embd * self.n_embd + 
                       8 * self.n_embd * self.n_embd + 
                       4 * self.n_embd)
        
        # 总参数 = 嵌入 + 层数 * 每层参数 + 最终LM头
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
        # k.shape = (B, nh, T, hs), 我们想转置最后两个维度 T 和 hs
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


class ProgressiveMLP(nn.Module):
    """渐进式MLP"""
    config: ProgressiveGPTConfig
    
    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)(x)
        hidden = jax.nn.gelu(hidden)
        return nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)(hidden)


class ProgressiveTransformerBlock(nn.Module):
    """渐进式Transformer块"""
    config: ProgressiveGPTConfig
    
    @nn.compact
    def __call__(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_out = ProgressiveMultiHeadAttention(self.config)(nn.LayerNorm()(x), mask)
        x = x + attn_out
        
        # MLP + 残差连接
        mlp_out = ProgressiveMLP(self.config)(nn.LayerNorm()(x))
        x = x + mlp_out
        
        return x


class ProgressiveGPT(nn.Module):
    """渐进式GPT模型"""
    config: ProgressiveGPTConfig
    
    @nn.compact
    def __call__(self, input_ids, training: bool = False):
        B, T = input_ids.shape
        
        # Token嵌入
        token_embed = nn.Embed(self.config.vocab_size, self.config.n_embd)(input_ids)
        
        # 位置嵌入
        pos_embed = nn.Embed(self.config.n_positions, self.config.n_embd)(
            jnp.arange(T)[None, :] % self.config.n_positions
        )
        
        # 组合嵌入
        x = token_embed + pos_embed
        
        # 创建因果掩码
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        mask = mask.astype(jnp.bool_)
        
        # Transformer层
        for i in range(self.config.n_layer):
            x = ProgressiveTransformerBlock(self.config)(x, mask)
        
        # 最终层归一化
        x = nn.LayerNorm()(x)
        
        # 语言模型头
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        return logits


class ProgressiveInferenceEngine:
    """渐进式推理引擎"""
    
    def __init__(self, config: ProgressiveGPTConfig):
        self.config = config
        self.devices = jax.devices()
        
        print(f"🔧 初始化渐进式GPT推理引擎")
        print(f"   设备数量: {len(self.devices)}")
        for i, device in enumerate(self.devices):
            print(f"   设备 {i}: {device}")
        
        # 估算参数量
        estimated_params = config.get_param_count()
        print(f"   估算参数量: {estimated_params:,} ({estimated_params/1e6:.1f}M)")
        
        # 初始化模型
        self.model = ProgressiveGPT(config)
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        print("🔄 初始化模型参数...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
        
        # 初始化参数
        start_time = time.time()
        self.params = self.model.init(key, dummy_input, training=False)
        init_time = time.time() - start_time
        
        # 计算实际参数量
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"📊 实际参数量: {param_count:,} ({param_count/1e6:.1f}M)")
        print(f"⏱️ 参数初始化时间: {init_time:.2f}s")
        
        # 内存使用估算
        param_memory = param_count * 4 / (1024**3)  # 假设float32
        print(f"💾 参数内存使用: {param_memory:.2f} GB")
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, params, input_ids):
        """JIT编译的前向传播"""
        return self.model.apply(params, input_ids, training=False)
    
    def warmup(self, input_ids: jnp.ndarray) -> float:
        """预热JIT编译"""
        print("🔥 预热JIT编译...")
        start_time = time.time()
        
        # 执行几次前向传播以完成JIT编译
        for _ in range(3):
            logits = self.forward_pass(self.params, input_ids)
            jax.block_until_ready(logits)
        
        warmup_time = time.time() - start_time
        print(f"⏱️ JIT编译时间: {warmup_time:.2f}s")
        return warmup_time
    
    def benchmark_forward_pass(self, input_ids: jnp.ndarray, num_runs: int = 10) -> Dict[str, float]:
        """基准测试前向传播"""
        print(f"🏃 运行前向传播基准测试 ({num_runs}次)...")
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            logits = self.forward_pass(self.params, input_ids)
            jax.block_until_ready(logits)
            end_time = time.time()
            
            times.append(end_time - start_time)
            if (i + 1) % 5 == 0:
                print(f"  完成 {i + 1}/{num_runs} 次")
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'times': times
        }
    
    def benchmark_generation(self, input_ids: jnp.ndarray, max_new_tokens: int = 32) -> Dict[str, Any]:
        """基准测试文本生成"""
        print(f"📝 运行文本生成基准测试 (生成{max_new_tokens}个tokens)...")
        
        current_ids = input_ids
        generation_times = []
        
        start_time = time.time()
        
        for step in range(max_new_tokens):
            # 防止序列过长
            if current_ids.shape[1] >= self.config.n_positions:
                current_ids = current_ids[:, -(self.config.n_positions-1):]
            
            # 生成下一个token
            step_start = time.time()
            logits = self.forward_pass(self.params, current_ids)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
            jax.block_until_ready(current_ids)
            step_time = time.time() - step_start
            
            generation_times.append(step_time)
            
            if (step + 1) % 10 == 0:
                print(f"  生成 {step + 1}/{max_new_tokens} tokens")
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'tokens_generated': max_new_tokens,
            'throughput': max_new_tokens / total_time,
            'mean_token_time': np.mean(generation_times),
            'std_token_time': np.std(generation_times),
            'final_sequence_length': current_ids.shape[1],
            'output_ids': current_ids
        }


def get_test_configurations() -> List[ProgressiveGPTConfig]:
    """获取测试配置列表"""
    configs = [
        # 小型模型 (~100M参数)
        ProgressiveGPTConfig(
            n_embd=768, n_layer=12, n_head=12, n_positions=512
        ),
        # 中型模型 (~300M参数)
        ProgressiveGPTConfig(
            n_embd=1024, n_layer=24, n_head=16, n_positions=1024
        ),
        # 大型模型 (~800M参数)
        ProgressiveGPTConfig(
            n_embd=1280, n_layer=36, n_head=20, n_positions=1024
        ),
        # 超大型模型 (~1.5B参数)
        ProgressiveGPTConfig(
            n_embd=1600, n_layer=48, n_head=25, n_positions=2048
        ),
    ]
    return configs


def run_progressive_benchmark():
    """运行渐进式基准测试"""
    print("\n🚀 开始渐进式GPT推理测试")
    print("=" * 60)
    
    configs = get_test_configurations()
    results = []
    
    # 准备测试输入
    test_input = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
    
    for i, config in enumerate(configs):
        print(f"\n📋 测试配置 {i+1}/{len(configs)}")
        print(f"   嵌入维度: {config.n_embd}")
        print(f"   层数: {config.n_layer}")
        print(f"   注意力头数: {config.n_head}")
        print(f"   最大位置: {config.n_positions}")
        print("-" * 40)
        
        try:
            # 创建推理引擎
            engine = ProgressiveInferenceEngine(config)
            
            # 预热
            warmup_time = engine.warmup(test_input)
            
            # 前向传播基准测试
            forward_results = engine.benchmark_forward_pass(test_input, num_runs=5)
            
            # 文本生成基准测试
            generation_results = engine.benchmark_generation(test_input, max_new_tokens=16)
            
            # 收集结果
            result = {
                'config': {
                    'n_embd': config.n_embd,
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_positions': config.n_positions,
                    'estimated_params': config.get_param_count()
                },
                'warmup_time': warmup_time,
                'forward_pass': forward_results,
                'generation': generation_results
            }
            results.append(result)
            
            # 打印结果摘要
            print(f"\n📊 测试结果摘要:")
            print(f"   预热时间: {warmup_time:.2f}s")
            print(f"   前向传播: {forward_results['mean_time']*1000:.1f}ms ± {forward_results['std_time']*1000:.1f}ms")
            print(f"   生成吞吐量: {generation_results['throughput']:.1f} tokens/s")
            print(f"   每token时间: {generation_results['mean_token_time']*1000:.1f}ms")
            
        except Exception as e:
            print(f"❌ 配置 {i+1} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            break  # 如果当前配置失败，停止测试更大的配置
    
    # 保存结果
    if results:
        results_file = Path("progressive_benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 结果已保存到: {results_file}")
        
        # 打印最终摘要
        print(f"\n🎯 渐进式测试完成!")
        print(f"   成功测试配置数: {len(results)}")
        print(f"   最大参数量: {max(r['config']['estimated_params'] for r in results)/1e6:.1f}M")
    
    return results


def main():
    """主函数"""
    print("🎯 渐进式GPT-1.5B JAX推理测试系统")
    print("=" * 60)
    print(f"📦 JAX版本: {jax.__version__}")
    print(f"🖥️ 设备数量: {len(jax.devices())}")
    
    for i, device in enumerate(jax.devices()):
        print(f"   设备 {i}: {device}")
    
    try:
        results = run_progressive_benchmark()
        if results:
            print("\n✅ 渐进式测试完成！")
            print("💡 查看 progressive_benchmark_results.json 获取详细结果")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
