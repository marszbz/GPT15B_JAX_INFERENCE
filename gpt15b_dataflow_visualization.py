#!/usr/bin/env python3
"""
GPT-1.5B推理数据流图可视化
展示完整的推理过程，包括多GPU分片和通信
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

try:
    import jax
    import jax.numpy as jnp
    from jax import random, devices
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    import flax.linen as nn
    import numpy as np
    print(f"✅ JAX {jax.__version__} 数据流可视化模式")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class GPT15BConfig:
    """GPT-1.5B配置"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.0
    use_bias: bool = True

class GPTDataFlowVisualizer:
    """GPT-1.5B数据流可视化器"""
    
    def __init__(self):
        self.devices = jax.devices()
        self.config = GPT15BConfig()
        self.mesh = None
        
    def visualize_complete_dataflow(self):
        """完整的GPT-1.5B数据流可视化"""
        print("🎨 GPT-1.5B推理数据流图可视化")
        print("="*100)
        
        # 步骤1：输入处理
        self.step1_input_processing()
        
        # 步骤2：嵌入层
        self.step2_embedding_layer()
        
        # 步骤3：Transformer层序列
        self.step3_transformer_layers()
        
        # 步骤4：输出处理
        self.step4_output_processing()
        
        # 步骤5：多GPU分片可视化
        self.step5_multi_gpu_sharding()
        
        # 步骤6：实际推理演示
        self.step6_real_inference_demo()
    
    def step1_input_processing(self):
        """步骤1：输入处理数据流"""
        print("\n🔤 步骤1：输入处理数据流")
        print("="*80)
        
        print("📝 输入数据流图：")
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输入处理阶段                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  原始文本                                                                   │
│  ┌─────────────────┐                                                        │
│  │ "Hello world!"  │                                                        │
│  └─────────────────┘                                                        │
│           │                                                                 │
│           ▼ Tokenization                                                    │
│  ┌─────────────────┐                                                        │
│  │ [15496, 995, 0] │ ← Token IDs                                           │
│  └─────────────────┘                                                        │
│           │                                                                 │
│           ▼ Padding & Batching                                              │
│  ┌─────────────────────────────────────┐                                    │
│  │ Input IDs: [B, seq_len]             │                                    │
│  │ Shape: (32, 512)                    │ ← Batch处理                        │
│  │ ┌─────┬─────┬─────┬─────┬─────┐     │                                    │
│  │ │15496│ 995 │  0  │ ... │ ... │     │                                    │
│  │ │15496│ 995 │  0  │ ... │ ... │     │                                    │
│  │ │ ... │ ... │ ... │ ... │ ... │     │                                    │
│  │ └─────┴─────┴─────┴─────┴─────┘     │                                    │
│  └─────────────────────────────────────┘                                    │
│           │                                                                 │
│           ▼ Attention Mask Generation                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │ Attention Mask: [B, seq_len, seq_len] │                                 │
│  │ Shape: (32, 512, 512)               │ ← 因果掩码                         │
│  │ ┌─────────────────────────────────┐ │                                    │
│  │ │ 1 0 0 0 0 ... (下三角矩阵)     │ │                                    │
│  │ │ 1 1 0 0 0 ...                   │ │                                    │
│  │ │ 1 1 1 0 0 ...                   │ │                                    │
│  │ │ 1 1 1 1 0 ...                   │ │                                    │
│  │ │ ...                             │ │                                    │
│  │ └─────────────────────────────────┘ │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        print("🔍 数据流详解：")
        print("   1. 文本标记化：将原始文本转换为token ID序列")
        print("   2. 批次打包：组合多个序列形成批次")
        print("   3. 填充对齐：确保序列长度一致")
        print("   4. 掩码生成：创建因果注意力掩码")
        
        print("\n📊 数据形状变化：")
        print("   原始文本 → Token序列: [seq_len]")
        print("   批次打包 → 输入矩阵: [batch_size, seq_len]")
        print("   掩码矩阵 → 注意力掩码: [batch_size, seq_len, seq_len]")
    
    def step2_embedding_layer(self):
        """步骤2：嵌入层数据流"""
        print("\n🔢 步骤2：嵌入层数据流")
        print("="*80)
        
        print("📊 嵌入层数据流图：")
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           嵌入层数据流                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入Token IDs                                                              │
│  ┌─────────────────────────┐                                                │
│  │ [32, 512] int32         │ ← 批次×序列长度                                │
│  │ ┌─────┬─────┬─────┐     │                                                │
│  │ │15496│ 995 │  0  │ ... │                                                │
│  │ │15496│ 995 │  0  │ ... │                                                │
│  │ │ ... │ ... │ ... │ ... │                                                │
│  │ └─────┴─────┴─────┘     │                                                │
│  └─────────────────────────┘                                                │
│           │                     │                                           │
│           │                     │                                           │
│  ┌────────▼──────────┐   ┌──────▼──────────┐                               │
│  │                   │   │                 │                               │
│  │  Token Embedding  │   │ Position Embedding │                           │
│  │                   │   │                 │                               │
│  │ ┌───────────────┐ │   │ ┌─────────────┐ │                               │
│  │ │Vocab: 50257   │ │   │ │Pos: 2048    │ │                               │
│  │ │Dim:   1600    │ │   │ │Dim: 1600    │ │                               │
│  │ │               │ │   │ │             │ │                               │
│  │ │Weight Matrix  │ │   │ │Learned Pos  │ │                               │
│  │ │[50257, 1600]  │ │   │ │[2048, 1600] │ │                               │
│  │ └───────────────┘ │   │ └─────────────┘ │                               │
│  └───────────────────┘   └─────────────────┘                               │
│           │                     │                                           │
│           ▼                     ▼                                           │
│  ┌─────────────────────────────────────┐                                    │
│  │       Token Embeddings              │                                    │
│  │       [32, 512, 1600]               │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼ Element-wise Addition                                 │
│  ┌─────────────────────────────────────┐                                    │
│  │     Combined Embeddings             │                                    │
│  │     [32, 512, 1600]                 │ ← 最终嵌入表示                     │
│  │ ┌─────────────────────────────────┐ │                                    │
│  │ │每个token的1600维向量表示        │ │                                    │
│  │ │包含语义信息和位置信息           │ │                                    │
│  │ └─────────────────────────────────┘ │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        print("🔍 嵌入层详解：")
        print("   1. Token嵌入：将token ID映射到高维向量空间")
        print("   2. 位置嵌入：为每个位置学习可训练的位置向量")
        print("   3. 向量相加：逐元素相加获得最终的输入表示")
        
        print("\n📊 参数统计：")
        token_emb_params = self.config.vocab_size * self.config.n_embd
        pos_emb_params = self.config.n_positions * self.config.n_embd
        total_emb_params = token_emb_params + pos_emb_params
        
        print(f"   Token嵌入参数: {token_emb_params:,} ({token_emb_params/1e6:.1f}M)")
        print(f"   位置嵌入参数: {pos_emb_params:,} ({pos_emb_params/1e6:.1f}M)")
        print(f"   嵌入层总参数: {total_emb_params:,} ({total_emb_params/1e6:.1f}M)")
    
    def step3_transformer_layers(self):
        """步骤3：Transformer层数据流"""
        print("\n🧠 步骤3：Transformer层数据流")
        print("="*80)
        
        print("📊 单个Transformer块数据流：")
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Transformer Block 数据流 (1/48)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: [32, 512, 1600]                                                      │
│  ┌─────────────────────────────────────┐                                    │
│  │ X = Combined Embeddings             │                                    │
│  │     [Batch, Seq, Hidden]            │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│                     │─────────────────────┐ Residual Connection           │
│                     │                     │                               │
│                     ▼                     │                               │
│  ┌─────────────────────────────────────┐  │                               │
│  │         Layer Norm 1                │  │                               │
│  │         [32, 512, 1600]             │  │                               │
│  └─────────────────────────────────────┘  │                               │
│                     │                     │                               │
│                     ▼                     │                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Multi-Head Attention                              │   │
│  │                                                                     │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                       │   │
│  │  │     Q     │  │     K     │  │     V     │                       │   │
│  │  │[32,512,   │  │[32,512,   │  │[32,512,   │                       │   │
│  │  │   1600]   │  │   1600]   │  │   1600]   │ ← Linear投影            │   │
│  │  └───────────┘  └───────────┘  └───────────┘                       │   │
│  │         │             │             │                               │   │
│  │         ▼             ▼             ▼                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │    Reshape to Multi-Head Format     │                           │   │
│  │  │    [32, 25, 512, 64]               │ ← 25个注意力头              │   │
│  │  └─────────────────────────────────────┘                           │   │
│  │                     │                                               │   │
│  │                     ▼                                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │         Scaled Dot-Product           │                           │   │
│  │  │         Attention                   │                           │   │
│  │  │   Attention = softmax(QK^T/√d_k)V   │                           │   │
│  │  └─────────────────────────────────────┘                           │   │
│  │                     │                                               │   │
│  │                     ▼                                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │      Concatenate Heads              │                           │   │
│  │  │      [32, 512, 1600]               │                           │   │
│  │  └─────────────────────────────────────┘                           │   │
│  │                     │                                               │   │
│  │                     ▼                                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │       Output Projection             │                           │   │
│  │  │       [32, 512, 1600]               │                           │   │
│  │  └─────────────────────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                     │                                                     │
│                     ▼                                                     │
│                ┌─────────┐ ◄──────────────────────────────────────────────┘
│                │   ADD   │ ← 残差连接
│                └─────────┘
│                     │
│                     ▼
│  ┌─────────────────────────────────────┐
│  │          Layer Norm 2               │
│  │          [32, 512, 1600]            │
│  └─────────────────────────────────────┘
│                     │
│                     ▼
│                     │─────────────────────┐ Residual Connection
│                     │                     │
│                     ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Feed Forward Network                             │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │        Linear Layer 1               │                           │   │
│  │  │        [32, 512, 6400]             │ ← 4x隐藏维度扩展            │   │
│  │  └─────────────────────────────────────┘                           │   │
│  │                     │                                               │   │
│  │                     ▼                                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │         GELU Activation             │                           │   │
│  │  │         [32, 512, 6400]            │                           │   │
│  │  └─────────────────────────────────────┘                           │   │
│  │                     │                                               │   │
│  │                     ▼                                               │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │        Linear Layer 2               │                           │   │
│  │  │        [32, 512, 1600]             │ ← 恢复到隐藏维度            │   │
│  │  └─────────────────────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                     │                                                     │
│                     ▼                                                     │
│                ┌─────────┐ ◄──────────────────────────────────────────────┘
│                │   ADD   │ ← 残差连接
│                └─────────┘
│                     │
│                     ▼
│  ┌─────────────────────────────────────┐
│  │    输出到下一层: [32, 512, 1600]     │ ← 传递给下一个Transformer块
│  └─────────────────────────────────────┘
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        print("\n🔄 48层Transformer序列：")
        print("""
Layer 1  → Layer 2  → Layer 3  → ... → Layer 48
  ↓          ↓          ↓                 ↓
[B,S,H]   [B,S,H]   [B,S,H]   ...    [B,S,H]

其中: B=32(batch), S=512(sequence), H=1600(hidden)
        """)
        
        print("\n📊 单层参数统计：")
        # 注意力层参数
        qkv_params = 3 * self.config.n_embd * self.config.n_embd
        attn_out_params = self.config.n_embd * self.config.n_embd
        attn_total = qkv_params + attn_out_params
        
        # MLP层参数
        mlp_up_params = self.config.n_embd * 4 * self.config.n_embd
        mlp_down_params = 4 * self.config.n_embd * self.config.n_embd
        mlp_total = mlp_up_params + mlp_down_params
        
        # LayerNorm参数
        ln_params = 2 * 2 * self.config.n_embd  # 2个LayerNorm，每个有scale和bias
        
        layer_total = attn_total + mlp_total + ln_params
        
        print(f"   注意力层参数: {attn_total:,} ({attn_total/1e6:.1f}M)")
        print(f"   MLP层参数: {mlp_total:,} ({mlp_total/1e6:.1f}M)")
        print(f"   LayerNorm参数: {ln_params:,} ({ln_params/1e3:.1f}K)")
        print(f"   单层总参数: {layer_total:,} ({layer_total/1e6:.1f}M)")
        print(f"   48层总参数: {layer_total * 48:,} ({layer_total * 48/1e6:.1f}M)")
    
    def step4_output_processing(self):
        """步骤4：输出处理数据流"""
        print("\n📤 步骤4：输出处理数据流")
        print("="*80)
        
        print("📊 输出层数据流图：")
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出处理阶段                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  来自最后一层Transformer                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │   Hidden States                     │                                    │
│  │   [32, 512, 1600]                   │ ← 最终的隐藏表示                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │      Final Layer Norm               │                                    │
│  │      [32, 512, 1600]                │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Language Model Head                               │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────┐                           │   │
│  │  │      Linear Projection              │                           │   │
│  │  │      [32, 512, 50257]              │ ← 投影到词汇表大小          │   │
│  │  │                                     │                           │   │
│  │  │  ┌───────────────────────────────┐  │                           │   │
│  │  │  │    Weight Matrix              │  │                           │   │
│  │  │  │    [1600, 50257]              │  │ ← 与token embedding共享    │   │
│  │  │  │                               │  │                           │   │
│  │  │  │ 每个隐藏维度对应              │  │                           │   │
│  │  │  │ 整个词汇表的概率分布          │  │                           │   │
│  │  │  └───────────────────────────────┘  │                           │   │
│  │  └─────────────────────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │         Logits                      │                                    │
│  │         [32, 512, 50257]            │ ← 原始分数                          │
│  │ ┌─────────────────────────────────┐ │                                    │
│  │ │每个位置对所有vocab的未归一化分数│ │                                    │
│  │ └─────────────────────────────────┘ │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │       Softmax (可选)                │                                    │
│  │       [32, 512, 50257]              │                                    │
│  │ ┌─────────────────────────────────┐ │                                    │
│  │ │   概率分布 (sum=1 for each pos) │ │                                    │
│  │ │   P(token|context)              │ │                                    │
│  │ └─────────────────────────────────┘ │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │    Token Sampling/Selection         │                                    │
│  │                                     │                                    │
│  │ ┌─────────────────────────────────┐ │                                    │
│  │ │ • Greedy: argmax(logits)        │ │                                    │
│  │ │ • Top-k: 选择top-k候选           │ │                                    │
│  │ │ • Top-p: 核采样                 │ │                                    │
│  │ │ • Temperature: 控制随机性       │ │                                    │
│  │ └─────────────────────────────────┘ │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │      Next Token IDs                 │                                    │
│  │      [32, 1]                        │ ← 每个序列的下一个token            │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        print("🔍 输出处理详解：")
        print("   1. 最终LayerNorm：标准化最后的隐藏状态")
        print("   2. 语言模型头：将隐藏状态投影到词汇表空间")
        print("   3. Logits生成：为每个token计算未归一化分数")
        print("   4. Token采样：根据策略选择下一个token")
        
        print("\n📊 输出层参数统计：")
        lm_head_params = self.config.n_embd * self.config.vocab_size
        print(f"   LM Head参数: {lm_head_params:,} ({lm_head_params/1e6:.1f}M)")
        print("   注意：通常与token embedding权重共享")
    
    def step5_multi_gpu_sharding(self):
        """步骤5：多GPU分片数据流"""
        print("\n🔀 步骤5：多GPU分片数据流")
        print("="*80)
        
        if len(self.devices) < 4:
            print("⚠️ 需要4个GPU来展示完整的分片策略")
            return
        
        print("📊 4GPU分片策略数据流：")
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4GPU 2x2 分片策略                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  设备网格布局：                                                             │
│  ┌─────────────┬─────────────┐                                              │
│  │    GPU 0    │    GPU 1    │ ← data轴=0                                  │
│  │   (0,0)     │   (0,1)     │                                             │
│  ├─────────────┼─────────────┤                                              │
│  │    GPU 2    │    GPU 3    │ ← data轴=1                                  │
│  │   (1,0)     │   (1,1)     │                                             │
│  └─────────────┴─────────────┘                                              │
│       ↑             ↑                                                       │
│    model=0       model=1                                                    │
│                                                                             │
│  输入数据分片：[32, 512] → 按batch维度分片                                  │
│  ┌─────────────────────────────────────┐                                    │
│  │ GPU 0: [16, 512] │ GPU 1: [16, 512] │ ← 前16个样本 | 后16个样本           │
│  │ GPU 2: [16, 512] │ GPU 3: [16, 512] │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
│  嵌入权重分片：[50257, 1600] → 按vocab维度分片                              │
│  ┌─────────────────────────────────────┐                                    │
│  │ GPU 0: [25128,1600] │ GPU 1: [25129,1600] │ ← vocab前半 | vocab后半     │
│  │ GPU 2: [25128,1600] │ GPU 3: [25129,1600] │                             │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
│  注意力权重分片：QKV投影 [1600, 4800] → 按head维度分片                     │
│  ┌─────────────────────────────────────┐                                    │
│  │ GPU 0: [1600,2400] │ GPU 1: [1600,2400] │ ← 前12.5头 | 后12.5头         │
│  │ GPU 2: [1600,2400] │ GPU 3: [1600,2400] │                               │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
│  MLP权重分片：[1600, 6400] → 按隐藏维度分片                                 │
│  ┌─────────────────────────────────────┐                                    │
│  │ GPU 0: [1600,3200] │ GPU 1: [1600,3200] │ ← 前半隐藏层 | 后半隐藏层      │
│  │ GPU 2: [1600,3200] │ GPU 3: [1600,3200] │                               │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
│  通信模式：                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. All-Gather: 收集attention头结果                                 │   │
│  │    GPU0,1,2,3 → 交换头计算结果                                      │   │
│  │                                                                     │   │
│  │ 2. All-Reduce: MLP层结果聚合                                       │   │
│  │    GPU0,1,2,3 → 求和隐藏层计算                                      │   │
│  │                                                                     │   │
│  │ 3. Reduce-Scatter: 输出logits分布                                  │   │
│  │    GPU0,1,2,3 → 重新分片结果                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        print("🔍 分片策略详解：")
        print("   1. 数据并行：batch维度分片，每GPU处理部分样本")
        print("   2. 模型并行：参数按不同维度分片到不同GPU")
        print("   3. 混合并行：结合数据和模型并行策略")
        print("   4. 自动通信：JAX自动插入必要的通信操作")
        
        # 创建实际的mesh演示
        self._create_actual_mesh_demo()
    
    def _create_actual_mesh_demo(self):
        """创建实际的mesh演示"""
        devices_array = np.array(self.devices[:4]).reshape(2, 2)
        self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
        
        print("\n🕸️ 实际设备网格：")
        print(f"   网格形状: {dict(self.mesh.shape)}")
        print(f"   轴名称: {self.mesh.axis_names}")
        print(f"   设备分布:")
        
        for i in range(2):
            for j in range(2):
                device = self.mesh.devices[i, j]
                print(f"     位置[{i},{j}]: {device}")
    
    def step6_real_inference_demo(self):
        """步骤6：实际推理演示"""
        print("\n🚀 步骤6：实际推理演示")
        print("="*80)
        
        if not self.mesh:
            print("⚠️ 需要创建mesh才能进行实际演示")
            return
        
        print("🎬 执行实际的分片推理...")
        
        # 模拟GPT推理的关键计算
        def simple_gpt_step(x, wte, wpe, attn_w, mlp_w):
            """简化的GPT前向步骤"""
            # 1. 嵌入
            batch_size, seq_len = x.shape
            token_emb = wte[x]  # [batch, seq, embd]
            pos_ids = jnp.arange(seq_len)[None, :]  # [1, seq]
            pos_emb = wpe[pos_ids]  # [1, seq, embd]
            hidden = token_emb + pos_emb
            
            # 2. 简化的注意力
            hidden = jnp.dot(hidden, attn_w)
            hidden = jax.nn.gelu(hidden)
            
            # 3. 简化的MLP
            hidden = jnp.dot(hidden, mlp_w)
            
            # 4. 输出logits
            logits = jnp.dot(hidden, wte.T)  # 权重共享
            
            return logits
        
        # JIT编译
        jit_gpt_step = jax.jit(simple_gpt_step)
        
        with self.mesh:
            # 创建测试数据
            key = jax.random.PRNGKey(42)
            batch_size, seq_len = 8, 64
            vocab_size, hidden_dim = 1000, 512  # 简化尺寸
            
            # 输入数据
            input_ids = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
            
            # 模型权重（简化）
            wte = jax.random.normal(key, (vocab_size, hidden_dim)) * 0.02
            wpe = jax.random.normal(key, (seq_len, hidden_dim)) * 0.02
            attn_w = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
            mlp_w = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
            
            # 定义分片策略
            input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            wte_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            wpe_sharding = NamedSharding(self.mesh, PartitionSpec(None, None))
            attn_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            mlp_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            
            # 应用分片
            input_ids_sharded = jax.device_put(input_ids, input_sharding)
            wte_sharded = jax.device_put(wte, wte_sharding)
            wpe_sharded = jax.device_put(wpe, wpe_sharding)
            attn_w_sharded = jax.device_put(attn_w, attn_sharding)
            mlp_w_sharded = jax.device_put(mlp_w, mlp_sharding)
            
            print("📊 分片信息：")
            print(f"   输入: {input_ids.shape} → PartitionSpec('data', None)")
            print(f"   Token嵌入: {wte.shape} → PartitionSpec('model', None)")
            print(f"   位置嵌入: {wpe.shape} → PartitionSpec(None, None)")
            print(f"   注意力权重: {attn_w.shape} → PartitionSpec(None, 'model')")
            print(f"   MLP权重: {mlp_w.shape} → PartitionSpec('model', None)")
            
            # 预热
            print("\n🔥 JIT编译预热...")
            for i in range(3):
                logits = jit_gpt_step(
                    input_ids_sharded, wte_sharded, wpe_sharded, 
                    attn_w_sharded, mlp_w_sharded
                )
                jax.block_until_ready(logits)
                print(f"   预热 {i+1}/3 完成")
            
            # 性能测试
            print("\n⚡ 推理性能测试...")
            times = []
            for i in range(5):
                start_time = time.time()
                logits = jit_gpt_step(
                    input_ids_sharded, wte_sharded, wpe_sharded,
                    attn_w_sharded, mlp_w_sharded
                )
                jax.block_until_ready(logits)
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                print(f"   测试 {i+1}: {elapsed*1000:.2f}ms")
            
            avg_time = np.mean(times)
            tokens_processed = batch_size * seq_len
            throughput = tokens_processed / avg_time
            
            print(f"\n🎯 推理结果：")
            print(f"   输入形状: {input_ids.shape}")
            print(f"   输出形状: {logits.shape}")
            print(f"   平均时间: {avg_time*1000:.2f}ms")
            print(f"   处理token数: {tokens_processed}")
            print(f"   吞吐量: {throughput:.1f} tokens/s")
            print(f"   设备利用: {len(self.devices)}个GPU并行")
            
            # 保存结果
            results = {
                'input_shape': list(input_ids.shape),
                'output_shape': list(logits.shape),
                'avg_time_ms': avg_time * 1000,
                'throughput_tokens_per_sec': throughput,
                'devices_used': len(self.devices),
                'mesh_shape': dict(self.mesh.shape)
            }
            
            results_file = Path("gpt15b_dataflow_demo_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 结果已保存: {results_file}")
    
    def generate_summary(self):
        """生成数据流总结"""
        print("\n🎯 GPT-1.5B数据流总结")
        print("="*80)
        
        print("📊 完整数据流路径：")
        print("""
文本输入 → Tokenization → 
    ↓
Token IDs [B, S] → Token Embedding [B, S, H] → 
    ↓
Position Embedding [S, H] → Combined Embedding [B, S, H] →
    ↓
Transformer Layer 1 → ... → Transformer Layer 48 →
    ↓
Final LayerNorm [B, S, H] → LM Head [B, S, V] →
    ↓
Logits [B, S, V] → Token Sampling → Next Token IDs [B, 1]

其中: B=Batch, S=Sequence, H=Hidden(1600), V=Vocab(50257)
        """)
        
        print("\n🚀 分布式加速策略：")
        print("   ✅ 数据并行：batch维度分片到多GPU")
        print("   ✅ 模型并行：参数矩阵分片到多GPU") 
        print("   ✅ 注意力头并行：多头注意力分布计算")
        print("   ✅ 自动通信：JAX处理GPU间数据交换")
        
        print("\n📈 性能优化技术：")
        print("   ✅ JIT编译：静态图编译优化")
        print("   ✅ 操作融合：减少内存访问")
        print("   ✅ 混合精度：可选FP16加速")
        print("   ✅ 权重共享：embedding和LM head共享权重")
        
        # 计算总参数量
        total_params = self._calculate_total_params()
        print(f"\n📊 模型规模统计：")
        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   模型大小: {total_params * 4 / (1024**3):.2f}GB (FP32)")
        print(f"   推荐GPU内存: ≥ {total_params * 4 / (1024**3) * 1.5:.1f}GB")
    
    def _calculate_total_params(self):
        """计算总参数量"""
        # 嵌入层
        token_emb = self.config.vocab_size * self.config.n_embd
        pos_emb = self.config.n_positions * self.config.n_embd
        
        # Transformer层
        qkv_params = 3 * self.config.n_embd * self.config.n_embd
        attn_out_params = self.config.n_embd * self.config.n_embd
        mlp_up_params = self.config.n_embd * 4 * self.config.n_embd
        mlp_down_params = 4 * self.config.n_embd * self.config.n_embd
        ln_params = 2 * 2 * self.config.n_embd  # 2个LayerNorm，每个有scale和bias
        
        layer_params = qkv_params + attn_out_params + mlp_up_params + mlp_down_params + ln_params
        
        # 输出层（通常与token embedding共享，这里不重复计算）
        final_ln_params = 2 * self.config.n_embd
        
        total = token_emb + pos_emb + layer_params * self.config.n_layer + final_ln_params
        return total

def main():
    """主函数"""
    visualizer = GPTDataFlowVisualizer()
    
    print("🎨 开始GPT-1.5B推理数据流可视化...")
    
    # 执行完整的数据流可视化
    visualizer.visualize_complete_dataflow()
    
    # 生成总结
    visualizer.generate_summary()
    
    print("\n🎉 GPT-1.5B数据流可视化完成！")
    print("   现在您可以清楚地看到整个推理过程的数据流向。")

if __name__ == "__main__":
    main()
