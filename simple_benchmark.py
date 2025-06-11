#!/usr/bin/env python3
"""
简化版GPT-1.5B JAX推理测试 - 避免复杂的图分割问题
"""

import os
import sys
import time
from pathlib import Path

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

from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import partial


@dataclass
class SimpleGPTConfig:
    """简化的GPT配置"""
    vocab_size: int = 50257
    n_positions: int = 512  # 减小序列长度
    n_embd: int = 768       # 减小嵌入维度
    n_layer: int = 12       # 减少层数
    n_head: int = 12        # 减少注意力头数
    dropout: float = 0.1
    use_bias: bool = True


class SimpleGPT(nn.Module):
    """简化的GPT模型"""
    config: SimpleGPTConfig
    
    @nn.compact
    def __call__(self, input_ids, training: bool = False):
        B, T = input_ids.shape
        
        # Token嵌入
        token_embed = nn.Embed(self.config.vocab_size, self.config.n_embd)(input_ids)
        
        # 位置嵌入
        pos_embed = nn.Embed(self.config.n_positions, self.config.n_embd)(
            jnp.arange(T)[None, :]
        )
        
        # 组合嵌入
        x = token_embed + pos_embed
        
        # 简化的Transformer层
        for i in range(self.config.n_layer):
            # 自注意力
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.config.n_head,
                qkv_features=self.config.n_embd
            )
            x_norm = nn.LayerNorm()(x)
            attn_out = attn(x_norm, x_norm, mask=jnp.tril(jnp.ones((T, T))))
            x = x + attn_out
            
            # MLP
            mlp_out = nn.Dense(4 * self.config.n_embd)(nn.LayerNorm()(x))
            mlp_out = jax.nn.gelu(mlp_out)
            mlp_out = nn.Dense(self.config.n_embd)(mlp_out)
            x = x + mlp_out
        
        # 最终层归一化
        x = nn.LayerNorm()(x)
        
        # 语言模型头
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        return logits


class SimpleInferenceEngine:
    """简化的推理引擎"""
    
    def __init__(self, config: SimpleGPTConfig):
        self.config = config
        self.devices = jax.devices()
        
        print(f"🔧 初始化简化GPT推理引擎")
        print(f"   设备数量: {len(self.devices)}")
        for i, device in enumerate(self.devices):
            print(f"   设备 {i}: {device}")
        
        # 初始化模型
        self.model = SimpleGPT(config)
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        print("🔄 初始化模型参数...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        
        # 初始化参数
        self.params = self.model.init(key, dummy_input, training=False)
        
        # 计算参数量
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"📊 模型参数量: {param_count:,} ({param_count/1e6:.1f}M)")
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, params, input_ids):
        """JIT编译的前向传播"""
        return self.model.apply(params, input_ids, training=False)
    
    def generate_text(self, input_ids: jnp.ndarray, max_new_tokens: int = 16) -> jnp.ndarray:
        """自回归文本生成"""
        current_ids = input_ids
        
        for step in range(max_new_tokens):
            # 防止序列过长
            if current_ids.shape[1] >= self.config.n_positions:
                current_ids = current_ids[:, -(self.config.n_positions-1):]
            
            # 生成下一个token
            logits = self.forward_pass(self.params, current_ids)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            
            # 拼接新token
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        return current_ids


def simple_benchmark():
    """简化的基准测试"""
    print("\n🚀 开始简化GPT推理测试")
    print("=" * 50)
    
    # 初始化模型
    config = SimpleGPTConfig()
    engine = SimpleInferenceEngine(config)
    
    # 准备测试数据
    test_inputs = [
        jnp.array([[1, 2, 3, 4, 5]]),
        jnp.array([[10, 20, 30, 40, 50]]),
        jnp.array([[100, 200, 300, 400, 500]]),
    ]
    
    print(f"\n🧪 运行推理测试...")
    print("-" * 30)
    
    total_time = 0
    for i, input_ids in enumerate(test_inputs):
        print(f"测试 {i+1}: 输入长度 {input_ids.shape[1]}")
        
        # 预热
        _ = engine.generate_text(input_ids, max_new_tokens=4)
        
        # 计时推理
        start_time = time.time()
        output = engine.generate_text(input_ids, max_new_tokens=8)
        jax.block_until_ready(output)  # 确保计算完成
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        tokens_generated = output.shape[1] - input_ids.shape[1]
        throughput = tokens_generated / inference_time if inference_time > 0 else 0
        
        print(f"  推理时间: {inference_time:.3f}s")
        print(f"  生成token数: {tokens_generated}")
        print(f"  吞吐量: {throughput:.1f} tokens/s")
        print(f"  输出序列长度: {output.shape[1]}")
        print()
    
    print("📊 测试总结:")
    print(f"  总测试时间: {total_time:.3f}s")
    print(f"  平均推理时间: {total_time/len(test_inputs):.3f}s")
    print(f"  设备数量: {len(jax.devices())}")
    
    return True


def main():
    """主函数"""
    print("🎯 简化版GPT-1.5B JAX推理测试")
    print("=" * 60)
    print(f"📦 JAX版本: {jax.__version__}")
    print(f"🖥️ 设备数量: {len(jax.devices())}")
    
    try:
        success = simple_benchmark()
        if success:
            print("\n✅ 测试完成！")
            print("💡 如果简化版本工作正常，可以尝试完整版本的基准测试")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
