"""
GPT-1.5B模型定义 - 使用JAX/Flax实现
支持图分割和多GPU并行推理
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import partial


@dataclass
class GPTConfig:
    """GPT-1.5B模型配置"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.1
    use_bias: bool = True


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T, C = x.shape
        head_dim = C // self.config.n_head
        
        # QKV投影
        qkv = nn.Dense(3 * C, use_bias=self.config.use_bias)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # 重塑为多头格式
        q = q.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        
        # 缩放点积注意力
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # 因果掩码
        mask = jnp.tril(jnp.ones((T, T)))
        attn_weights = jnp.where(mask, attn_weights, -1e10)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Dropout
        if training:
            attn_weights = nn.Dropout(rate=self.config.dropout)(
                attn_weights, deterministic=not training
            )
        
        # 应用注意力权重
        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        out = nn.Dense(C, use_bias=self.config.use_bias)(out)
        if training:
            out = nn.Dropout(rate=self.config.dropout)(out, deterministic=not training)
        
        return out


class MLP(nn.Module):
    """前馈神经网络"""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # 第一层：升维到4*n_embd
        x = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)(x)
        x = jax.nn.gelu(x)
        
        # 第二层：降维回n_embd
        x = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)(x)
        
        if training:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not training)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer解码器块"""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # 自注意力 + 残差连接 + 层归一化
        attn_out = MultiHeadAttention(self.config)(x, training)
        x = nn.LayerNorm()(x + attn_out)
        
        # MLP + 残差连接 + 层归一化
        mlp_out = MLP(self.config)(x, training)
        x = nn.LayerNorm()(x + mlp_out)
        
        return x


class GPTModel(nn.Module):
    """GPT-1.5B主模型"""
    config: GPTConfig
    
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
        
        if training:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not training)
        
        # 48个Transformer块
        for _ in range(self.config.n_layer):
            x = TransformerBlock(self.config)(x, training)
        
        # 最终层归一化
        x = nn.LayerNorm()(x)
        
        # 语言模型头
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        
        return logits


class GraphPartitionedGPT:
    """图分割GPT推理引擎"""
    
    def __init__(self, config: GPTConfig):
        self.config = config
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        print(f"🔧 初始化图分割GPT推理引擎")
        print(f"   GPU数量: {self.num_devices}")
        for i, device in enumerate(self.devices):
            print(f"   GPU {i}: {device}")
        
        # 初始化模型
        self.model = GPTModel(config)
        self._init_model_parameters()
        
        # 设置多GPU分片
        if self.num_devices > 1:
            self._setup_graph_partitioning()
        else:
            print("ℹ️ 单GPU模式")
            self.sharded_params = self.params
    
    def _init_model_parameters(self):
        """初始化模型参数"""
        print("🔄 初始化GPT-1.5B模型参数...")
        
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        
        # 初始化参数
        self.params = self.model.init(key, dummy_input, training=False)
        
        # 计算参数量
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"📊 模型参数量: {param_count:,} ({param_count/1e9:.2f}B)")
    
    def _setup_graph_partitioning(self):
        """设置图分割和参数分片"""
        print("🕸️ 设置图分割和多GPU并行...")
        
        # 创建设备网格
        self.mesh = jax.make_mesh((self.num_devices,), ('model',))
        
        # 定义分片策略
        def get_partition_spec(param):
            """为不同参数定义分片策略"""
            if param.ndim >= 2:
                # 大的权重矩阵按第一个维度分片
                if param.shape[0] >= 512:
                    return jax.sharding.PartitionSpec('model', None)
                elif param.shape[1] >= 512:
                    return jax.sharding.PartitionSpec(None, 'model')
                else:
                    return jax.sharding.PartitionSpec()
            else:
                # 1D参数不分片
                return jax.sharding.PartitionSpec()
        
        # 应用分片规范
        self.param_spec = jax.tree_util.tree_map(get_partition_spec, self.params)
        
        # 创建分片并分布参数
        sharding = jax.sharding.NamedSharding(self.mesh, self.param_spec)
        self.sharded_params = jax.device_put(self.params, sharding)
        
        print(f"✅ 图分割完成，参数已分布到 {self.num_devices} 个GPU")
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, params, input_ids):
        """JIT编译的前向传播"""
        return self.model.apply(params, input_ids, training=False)
    
    @partial(jax.jit, static_argnums=(0,))
    def generate_next_token(self, params, input_ids):
        """生成下一个token（JIT编译）"""
        logits = self.forward_pass(params, input_ids)
        # 贪婪解码：选择概率最高的token
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        return next_token
    
    def generate_text(self, input_ids: jnp.ndarray, max_new_tokens: int = 32) -> jnp.ndarray:
        """自回归文本生成"""
        current_ids = input_ids
        params = self.sharded_params
        
        for step in range(max_new_tokens):
            # 防止序列过长
            if current_ids.shape[1] >= self.config.n_positions:
                # 截断保留最新的部分
                current_ids = current_ids[:, -(self.config.n_positions-1):]
            
            # 生成下一个token
            next_token = self.generate_next_token(params, current_ids)
            
            # 拼接新token
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        return current_ids
