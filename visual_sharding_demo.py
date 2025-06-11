#!/usr/bin/env python3
"""
JAX自动分片和图划分可视化演示
详细展示分片机制的工作原理
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
    print(f"✅ JAX {jax.__version__} 可视化演示模式")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

class ShardingVisualizer:
    """分片可视化器"""
    
    def __init__(self):
        self.devices = jax.devices()
        self.mesh = None
        
    def step1_understand_tensors(self):
        """第1步：理解张量和分片的基本概念"""
        print(f"\n" + "="*60)
        print(f"🎯 第1步：理解张量分片的基本概念")
        print(f"="*60)
        
        # 创建一个简单的矩阵
        key = jax.random.PRNGKey(42)
        matrix = jax.random.normal(key, (8, 8))
        
        print(f"📊 原始矩阵 (8x8):")
        print(f"   形状: {matrix.shape}")
        print(f"   数据类型: {matrix.dtype}")
        print(f"   总元素数: {matrix.size}")
        print(f"   内存大小: {matrix.nbytes} bytes")
        
        # 显示矩阵内容（简化版）
        print(f"\n📋 矩阵内容（前4x4）:")
        for i in range(4):
            row_str = "   "
            for j in range(4):
                row_str += f"{matrix[i,j]:.2f} "
            print(row_str + "...")
        print("   ...")
        
        print(f"\n💡 分片概念:")
        print(f"   分片 = 将大矩阵分割成小块，分布到不同GPU")
        print(f"   目的 = 突破单GPU内存限制，实现并行计算")
        print(f"   方式 = 按行分片、按列分片、按块分片")
        
        return matrix
    
    def step2_create_mesh_visualization(self):
        """第2步：可视化设备网格创建"""
        print(f"\n" + "="*60)
        print(f"🔧 第2步：设备网格创建过程")
        print(f"="*60)
        
        print(f"📱 可用设备:")
        for i, device in enumerate(self.devices):
            print(f"   GPU {i}: {device}")
        
        # 创建2x2网格
        if len(self.devices) >= 4:
            devices_array = np.array(self.devices[:4]).reshape(2, 2)
            self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
            
            print(f"\n🕸️ 创建2x2设备网格:")
            print(f"   ┌─────────┬─────────┐")
            print(f"   │ GPU 0   │ GPU 1   │  ← data轴 第0行")
            print(f"   │ {self.mesh.devices[0,0]} │ {self.mesh.devices[0,1]} │")
            print(f"   ├─────────┼─────────┤")
            print(f"   │ GPU 2   │ GPU 3   │  ← data轴 第1行")
            print(f"   │ {self.mesh.devices[1,0]} │ {self.mesh.devices[1,1]} │")
            print(f"   └─────────┴─────────┘")
            print(f"      ↑         ↑")
            print(f"   model轴    model轴")
            print(f"    第0列      第1列")
            
            print(f"\n📐 网格配置:")
            print(f"   形状: {dict(self.mesh.shape)}")
            print(f"   轴名称: {self.mesh.axis_names}")
            print(f"   data轴: 用于数据并行 (batch维度分片)")
            print(f"   model轴: 用于模型并行 (参数维度分片)")
            
        elif len(self.devices) == 1:
            devices_array = np.array(self.devices).reshape(1, 1)
            self.mesh = Mesh(devices_array, axis_names=('data',))
            print(f"\n🕸️ 单设备网格:")
            print(f"   ┌─────────┐")
            print(f"   │ GPU 0   │")
            print(f"   │ {self.mesh.devices[0]} │")
            print(f"   └─────────┘")
            
        return self.mesh
    
    def step3_partition_spec_explanation(self):
        """第3步：详解PartitionSpec分片规范"""
        print(f"\n" + "="*60)
        print(f"📋 第3步：PartitionSpec分片规范详解")
        print(f"="*60)
        
        if not self.mesh:
            print("⚠️ 需要先创建mesh")
            return
        
        print(f"🔍 PartitionSpec语法解析:")
        print(f"   PartitionSpec(axis0, axis1, axis2, ...)")
        print(f"   - None: 不分片，复制到所有设备")
        print(f"   - 'data': 沿data轴分片")
        print(f"   - 'model': 沿model轴分片")
        
        # 演示不同的分片策略
        examples = [
            {
                'name': '不分片',
                'spec': PartitionSpec(),
                'description': '张量复制到所有设备',
                'use_case': 'LayerNorm参数、bias等小参数'
            },
            {
                'name': 'batch分片',
                'spec': PartitionSpec('data', None),
                'description': '第0维(batch)沿data轴分片',
                'use_case': '输入数据、激活值'
            },
            {
                'name': '参数分片',
                'spec': PartitionSpec('model', None),
                'description': '第0维沿model轴分片',
                'use_case': '嵌入层权重、输出层权重'
            },
            {
                'name': '注意力头分片',
                'spec': PartitionSpec(None, 'model'),
                'description': '第1维沿model轴分片',
                'use_case': 'QKV投影权重'
            },
            {
                'name': '二维分片',
                'spec': PartitionSpec('data', 'model'),
                'description': '第0维沿data轴，第1维沿model轴分片',
                'use_case': '大型矩阵的块分片'
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n   {i}. {example['name']}:")
            print(f"      规范: {example['spec']}")
            print(f"      含义: {example['description']}")
            print(f"      用途: {example['use_case']}")
        
        return examples
    
    def step4_visual_sharding_demo(self):
        """第4步：可视化分片演示"""
        print(f"\n" + "="*60)
        print(f"🎬 第4步：实际分片过程可视化")
        print(f"="*60)
        
        if not self.mesh:
            print("⚠️ 需要先创建mesh")
            return
        
        # 创建测试矩阵
        key = jax.random.PRNGKey(42)
        
        # 演示1：batch分片
        print(f"\n📊 演示1：Batch分片")
        print(f"-" * 30)
        
        batch_data = jax.random.normal(key, (8, 4))  # 8个batch，4个特征
        print(f"原始数据形状: {batch_data.shape}")
        
        with self.mesh:
            # batch维度分片
            batch_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            sharded_batch = jax.device_put(batch_data, batch_sharding)
            
            print(f"分片策略: PartitionSpec('data', None)")
            print(f"分片结果:")
            
            if len(self.devices) >= 4:
                print(f"   GPU 0,2: batch 0-3 (shape: 4x4)")
                print(f"   GPU 1,3: batch 4-7 (shape: 4x4)")
                print(f"   │ batch0 batch1 batch2 batch3 │ → GPU 0,2")
                print(f"   │ batch4 batch5 batch6 batch7 │ → GPU 1,3")
            
        # 演示2：参数分片
        print(f"\n📊 演示2：参数分片")
        print(f"-" * 30)
        
        weight_matrix = jax.random.normal(key, (8, 8))
        print(f"权重矩阵形状: {weight_matrix.shape}")
        
        with self.mesh:
            # 第0维分片
            weight_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            sharded_weight = jax.device_put(weight_matrix, weight_sharding)
            
            print(f"分片策略: PartitionSpec('model', None)")
            print(f"分片结果:")
            
            if len(self.devices) >= 4:
                print(f"   GPU 0,1: 行 0-3 (shape: 4x8)")
                print(f"   GPU 2,3: 行 4-7 (shape: 4x8)")
                print(f"   ┌─ 行0-3 ─┐ → GPU 0,1")
                print(f"   │ weight  │")
                print(f"   ├─ 行4-7 ─┤ → GPU 2,3")
                print(f"   │ weight  │")
                print(f"   └─────────┘")
        
        # 演示3：注意力头分片
        print(f"\n📊 演示3：注意力头分片")
        print(f"-" * 30)
        
        qkv_weight = jax.random.normal(key, (8, 12))  # 8个输入，12个输出(3*4头)
        print(f"QKV权重形状: {qkv_weight.shape}")
        
        with self.mesh:
            # 第1维分片（注意力头）
            qkv_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            sharded_qkv = jax.device_put(qkv_weight, qkv_sharding)
            
            print(f"分片策略: PartitionSpec(None, 'model')")
            print(f"分片结果:")
            
            if len(self.devices) >= 4:
                print(f"   GPU 0,2: 输出 0-5 (Q,K的一半)")
                print(f"   GPU 1,3: 输出 6-11 (K,V的一半)")
                print(f"   ┌─Q头─┬─K头─┐ ┌─K头─┬─V头─┐")
                print(f"   │ 0-5 │ GPU0,2 │ 6-11│ GPU1,3 │")
                print(f"   └─────┴─────┘ └─────┴─────┘")
        
        return sharded_batch, sharded_weight, sharded_qkv
    
    def step5_automatic_sharding_mechanism(self):
        """第5步：自动分片机制深度解析"""
        print(f"\n" + "="*60)
        print(f"🤖 第5步：JAX自动分片机制解析")
        print(f"="*60)
        
        print(f"🧠 JAX自动分片的工作原理:")
        print(f"\n1️⃣ 编译时分析:")
        print(f"   - 扫描计算图中的所有操作")
        print(f"   - 分析张量的形状和使用模式")
        print(f"   - 根据PartitionSpec推断最优分片")
        
        print(f"\n2️⃣ 分片推断:")
        print(f"   - 输入分片 → 自动推断输出分片")
        print(f"   - 考虑操作语义（矩阵乘法、元素wise等）")
        print(f"   - 最小化设备间通信")
        
        print(f"\n3️⃣ 通信插入:")
        print(f"   - 检测分片不匹配的地方")
        print(f"   - 自动插入reshape/通信操作")
        print(f"   - 优化通信模式（AllReduce、AllGather等）")
        
        print(f"\n4️⃣ 代码生成:")
        print(f"   - 生成每个设备的本地计算代码")
        print(f"   - 插入必要的同步点")
        print(f"   - 优化内存布局")
        
        # 实际演示自动分片推断
        if self.mesh:
            self._demonstrate_automatic_inference()
    
    def _demonstrate_automatic_inference(self):
        """演示自动推断过程"""
        print(f"\n🔍 自动推断演示:")
        print(f"-" * 30)
        
        key = jax.random.PRNGKey(42)
        
        with self.mesh:
            # 定义一个简单的计算
            def matrix_multiply(x, w):
                return jnp.dot(x, w)
            
            # 创建输入
            x = jax.random.normal(key, (4, 8))  # batch=4, features=8
            w = jax.random.normal(key, (8, 16))  # features=8, hidden=16
            
            # 定义输入分片
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            
            # 分片输入
            x_sharded = jax.device_put(x, x_sharding)
            w_sharded = jax.device_put(w, w_sharding)
            
            print(f"输入张量:")
            print(f"   x: {x.shape} → 分片策略: ('data', None)")
            print(f"   w: {w.shape} → 分片策略: (None, 'model')")
            
            # 执行计算
            result = matrix_multiply(x_sharded, w_sharded)
            
            print(f"\n计算: y = x @ w")
            print(f"输出张量:")
            print(f"   y: {result.shape}")
            print(f"   自动推断的分片: ('data', 'model')")
            
            print(f"\n🎯 推断逻辑:")
            print(f"   x(4,8) @ w(8,16) = y(4,16)")
            print(f"   x分片: (data, None)")
            print(f"   w分片: (None, model)")
            print(f"   ↓ 矩阵乘法规则")
            print(f"   y分片: (data, model) ← 自动推断!")
            
            print(f"\n📡 通信分析:")
            print(f"   - x的第1维与w的第0维收缩 → 需要AllReduce")
            print(f"   - 结果保持x的第0维分片(data)")
            print(f"   - 结果保持w的第1维分片(model)")
    
    def step6_graph_partitioning_visualization(self):
        """第6步：图划分可视化"""
        print(f"\n" + "="*60)
        print(f"🕸️ 第6步：计算图划分过程")
        print(f"="*60)
        
        print(f"📊 计算图划分的层次:")
        print(f"\n1️⃣ 操作级划分:")
        print(f"   原始图: A → B → C → D")
        print(f"   划分后: ")
        print(f"   GPU0: A₀ → B₀ → C₀ → D₀")
        print(f"   GPU1: A₁ → B₁ → C₁ → D₁")
        print(f"   GPU2: A₂ → B₂ → C₂ → D₂")
        print(f"   GPU3: A₃ → B₃ → C₃ → D₃")
        
        print(f"\n2️⃣ 数据流划分:")
        print(f"   输入数据 → 分片到各GPU")
        print(f"   计算过程 → 并行执行")
        print(f"   中间结果 → 根据需要通信")
        print(f"   最终输出 → 聚合或保持分片")
        
        print(f"\n3️⃣ 通信模式:")
        
        # ASCII艺术展示通信模式
        print(f"   AllReduce (全归约):")
        print(f"   GPU0 ←→ GPU1")
        print(f"    ↕    ↗↙ ↕")
        print(f"   GPU2 ←→ GPU3")
        
        print(f"\n   AllGather (全收集):")
        print(f"   GPU0 → 收集所有片段")
        print(f"   GPU1 → 收集所有片段")
        print(f"   GPU2 → 收集所有片段")
        print(f"   GPU3 → 收集所有片段")
        
        print(f"\n   Scatter (分散):")
        print(f"   主GPU → 分发到各GPU")
        print(f"   GPU0 ← 片段0")
        print(f"   GPU1 ← 片段1")
        print(f"   GPU2 ← 片段2")
        print(f"   GPU3 ← 片段3")
        
        # 实际演示图划分
        if self.mesh:
            self._demonstrate_graph_partitioning()
    
    def _demonstrate_graph_partitioning(self):
        """演示实际的图划分"""
        print(f"\n🎬 实际图划分演示:")
        print(f"-" * 30)
        
        # 定义一个多步骤计算
        def multi_step_computation(x, w1, w2):
            """多步骤计算：线性 → 激活 → 线性"""
            h1 = jnp.dot(x, w1)  # 第一个线性层
            h2 = jax.nn.relu(h1)  # 激活函数
            y = jnp.dot(h2, w2)   # 第二个线性层
            return y
        
        # JIT编译以观察图划分
        jit_computation = jax.jit(multi_step_computation)
        
        key = jax.random.PRNGKey(42)
        
        with self.mesh:
            # 创建测试数据
            x = jax.random.normal(key, (8, 16))   # batch=8, input=16
            w1 = jax.random.normal(key, (16, 32)) # input=16, hidden=32
            w2 = jax.random.normal(key, (32, 8))  # hidden=32, output=8
            
            # 定义分片策略
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w1_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            w2_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            
            # 应用分片
            x_sharded = jax.device_put(x, x_sharding)
            w1_sharded = jax.device_put(w1, w1_sharding)
            w2_sharded = jax.device_put(w2, w2_sharding)
            
            print(f"计算图结构:")
            print(f"   x(8,16) → Linear1 → ReLU → Linear2 → y(8,8)")
            print(f"   ↓分片")
            print(f"   x: ('data', None)")
            print(f"   w1: (None, 'model')")
            print(f"   w2: ('model', None)")
            
            # 执行计算
            result = jit_computation(x_sharded, w1_sharded, w2_sharded)
            
            print(f"\n图划分结果:")
            print(f"   步骤1: x @ w1 → h1('data', 'model')")
            print(f"   步骤2: ReLU(h1) → h2('data', 'model')")
            print(f"   步骤3: h2 @ w2 → y('data', None)")
            
            print(f"\n通信开销:")
            print(f"   步骤1→2: 无通信 (分片兼容)")
            print(f"   步骤2→3: AllReduce (model维度归约)")
            print(f"   总通信: 最小化!")
    
    def step7_performance_implications(self):
        """第7步：性能影响分析"""
        print(f"\n" + "="*60)
        print(f"⚡ 第7步：分片策略对性能的影响")
        print(f"="*60)
        
        print(f"📈 性能因素分析:")
        
        print(f"\n1️⃣ 计算并行度:")
        print(f"   数据并行: 线性加速 (理想情况)")
        print(f"   模型并行: 受通信限制")
        print(f"   混合并行: 平衡计算和通信")
        
        print(f"\n2️⃣ 内存效率:")
        print(f"   不分片: 内存复制，计算并行")
        print(f"   分片: 内存节省，可能增加通信")
        print(f"   优化目标: 最大化内存利用率")
        
        print(f"\n3️⃣ 通信开销:")
        print(f"   AllReduce: O(数据量)")
        print(f"   AllGather: O(数据量 × 设备数)")
        print(f"   点对点: O(数据量/设备数)")
        
        print(f"\n4️⃣ 负载均衡:")
        print(f"   理想: 每个设备计算量相等")
        print(f"   现实: 可能存在不平衡")
        print(f"   解决: 动态分片调整")
        
        # 实际性能测试
        if self.mesh:
            self._performance_comparison()
    
    def _performance_comparison(self):
        """性能对比测试"""
        print(f"\n🏁 性能对比测试:")
        print(f"-" * 30)
        
        key = jax.random.PRNGKey(42)
        
        # 测试不同分片策略的性能
        def simple_computation(x, w):
            return jnp.dot(x, w)
        
        jit_comp = jax.jit(simple_computation)
        
        # 创建测试数据
        x = jax.random.normal(key, (64, 512))
        w = jax.random.normal(key, (512, 2048))
        
        with self.mesh:
            # 策略1: 不分片
            x_replicated = jax.device_put(x, NamedSharding(self.mesh, PartitionSpec()))
            w_replicated = jax.device_put(w, NamedSharding(self.mesh, PartitionSpec()))
            
            # 策略2: batch分片
            x_data_sharded = jax.device_put(x, NamedSharding(self.mesh, PartitionSpec('data', None)))
            w_replicated2 = jax.device_put(w, NamedSharding(self.mesh, PartitionSpec()))
            
            # 策略3: 模型分片
            x_replicated3 = jax.device_put(x, NamedSharding(self.mesh, PartitionSpec()))
            w_model_sharded = jax.device_put(w, NamedSharding(self.mesh, PartitionSpec(None, 'model')))
            
            print(f"测试配置: x{x.shape} @ w{w.shape}")
            
            # 预热
            for strategy_name, x_test, w_test in [
                ("复制策略", x_replicated, w_replicated),
                ("数据分片", x_data_sharded, w_replicated2),
                ("模型分片", x_replicated3, w_model_sharded)
            ]:
                for _ in range(3):
                    result = jit_comp(x_test, w_test)
                    jax.block_until_ready(result)
                
                # 计时
                times = []
                for _ in range(5):
                    start = time.time()
                    result = jit_comp(x_test, w_test)
                    jax.block_until_ready(result)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000
                print(f"   {strategy_name}: {avg_time:.2f}ms")
    
    def comprehensive_demo(self):
        """完整演示流程"""
        print(f"🎯 JAX自动分片和图划分完整演示")
        print(f"="*60)
        print(f"📚 本演示将详细展示:")
        print(f"   1. 张量分片的基本概念")
        print(f"   2. 设备网格的创建过程")
        print(f"   3. PartitionSpec分片规范")
        print(f"   4. 实际分片过程可视化")
        print(f"   5. 自动分片机制解析")
        print(f"   6. 计算图划分过程")
        print(f"   7. 性能影响分析")
        
        # 逐步执行演示
        matrix = self.step1_understand_tensors()
        mesh = self.step2_create_mesh_visualization()
        examples = self.step3_partition_spec_explanation()
        sharded_tensors = self.step4_visual_sharding_demo()
        self.step5_automatic_sharding_mechanism()
        self.step6_graph_partitioning_visualization()
        self.step7_performance_implications()
        
        # 总结
        print(f"\n" + "="*60)
        print(f"🎓 总结：JAX分片机制的核心要点")
        print(f"="*60)
        print(f"✅ 关键概念:")
        print(f"   • 设备网格: 将GPU组织为逻辑网格")
        print(f"   • PartitionSpec: 指定张量如何分片")
        print(f"   • 自动推断: JAX自动推断最优分片")
        print(f"   • 通信优化: 最小化设备间数据传输")
        
        print(f"\n🔧 技术优势:")
        print(f"   • 透明性: 用户只需指定分片策略")
        print(f"   • 自动化: 编译器处理所有细节")
        print(f"   • 高效性: 智能通信优化")
        print(f"   • 灵活性: 支持多种并行模式")
        
        print(f"\n💡 最佳实践:")
        print(f"   • 根据模型大小选择分片策略")
        print(f"   • 平衡计算和通信开销")
        print(f"   • 利用混合并行获得最佳性能")
        print(f"   • 测试不同策略找到最优配置")

def main():
    """主函数"""
    visualizer = ShardingVisualizer()
    visualizer.comprehensive_demo()

if __name__ == "__main__":
    main()
