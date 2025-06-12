#!/usr/bin/env python3
"""
JAX分布式推理策略完整教程
详细解释JAX在多GPU环境下的分布式推理机制
"""

import os
import sys
import time
import numpy as np
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
    print("✅ JAX {} 分布式推理教程".format(jax.__version__))
except ImportError as e:
    print("❌ JAX导入失败: {}".format(e))
    sys.exit(1)

class JAXDistributedInferenceTutorial:
    """JAX分布式推理策略教程"""
    
    def __init__(self):
        self.devices = jax.devices()
        self.mesh = None
        
    def explain_distributed_concepts(self):
        """解释分布式推理的基本概念"""
        print("🎓 JAX分布式推理基础概念")
        print("="*80)
        
        print("💡 什么是分布式推理？")
        print("   分布式推理是将大型模型的推理计算分散到多个GPU上")
        print("   目标：突破单GPU内存限制，提高推理速度")
        
        print("\n🔍 JAX分布式推理的核心组件：")
        print("   1. 设备网格(Device Mesh): 将GPU组织为逻辑网格")
        print("   2. 分片策略(Sharding): 决定数据如何分布到各GPU")
        print("   3. 自动并行化: JAX自动处理计算和通信")
        print("   4. JIT编译: 静态图编译优化性能")
        
        print("\n📊 分布式推理的优势：")
        print("   ✅ 内存扩展: 可以运行超大模型")
        print("   ✅ 速度提升: 并行计算加速推理")
        print("   ✅ 自动化: JAX处理复杂的分布式细节")
        print("   ✅ 透明性: 代码几乎无需修改")
    
    def demonstrate_device_mesh_creation(self):
        """演示设备网格创建"""
        print("\n🕸️ 设备网格(Device Mesh)详解")
        print("="*80)
        
        print("📱 当前GPU配置：")
        for i, device in enumerate(self.devices):
            print("   GPU {}: {}".format(i, device))
        
        print("\n🔧 设备网格的作用：")
        print("   • 将物理GPU组织为逻辑网格")
        print("   • 定义数据和模型的分布轴")
        print("   • 支持多维并行策略")
        
        if len(self.devices) >= 4:
            # 创建2x2网格
            devices_array = np.array(self.devices[:4]).reshape(2, 2)
            self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
            
            print("\n✅ 创建2x2设备网格：")
            print("   网格形状: {}".format(dict(self.mesh.shape)))
            print("   轴名称: {}".format(self.mesh.axis_names))
            
            print("\n📐 设备分布图：")
            print("   ┌─────────┬─────────┐")
            print("   │  GPU0   │  GPU1   │ ← data轴=0")
            print("   │ (data=0)│ (data=0)│")
            print("   │(model=0)│(model=1)│")
            print("   ├─────────┼─────────┤")
            print("   │  GPU2   │  GPU3   │ ← data轴=1")
            print("   │ (data=1)│ (data=1)│")
            print("   │(model=0)│(model=1)│")
            print("   └─────────┴─────────┘")
            print("    model=0   model=1")
            
        elif len(self.devices) == 2:
            devices_array = np.array(self.devices).reshape(2, 1)
            self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
            print("\n✅ 创建2x1设备网格")
            
        else:
            print("\n⚠️ GPU数量不足，无法演示多GPU网格")
            return False
            
        return True
    
    def explain_sharding_strategies(self):
        """解释分片策略"""
        print("\n🔀 分片策略(Sharding Strategies)详解")
        print("="*80)
        
        print("🎯 什么是分片？")
        print("   分片是将大张量切分到多个设备上的策略")
        print("   目标：平衡内存使用和计算效率")
        
        print("\n📋 主要分片类型：")
        
        print("\n1️⃣ 数据并行(Data Parallelism):")
        print("   • 模型参数复制到每个GPU")
        print("   • 输入数据按batch维度分片")
        print("   • 适用：模型较小，数据量大")
        print("   • 示例：PartitionSpec('data', None)")
        
        print("\n2️⃣ 模型并行(Model Parallelism):")
        print("   • 模型参数按维度分片")
        print("   • 输入数据保持完整或部分分片")
        print("   • 适用：模型巨大，内存不足")
        print("   • 示例：PartitionSpec(None, 'model')")
        
        print("\n3️⃣ 混合并行(Hybrid Parallelism):")
        print("   • 结合数据并行和模型并行")
        print("   • 灵活的分片组合")
        print("   • 适用：大模型+大数据")
        print("   • 示例：PartitionSpec('data', 'model')")
        
        print("\n4️⃣ 流水线并行(Pipeline Parallelism):")
        print("   • 模型按层分片到不同GPU")
        print("   • 层间串行执行，层内并行")
        print("   • 适用：超深模型")
        
        if self.mesh:
            self._demonstrate_sharding_examples()
    
    def _demonstrate_sharding_examples(self):
        """演示具体的分片示例"""
        print("\n🎬 分片策略实际演示")
        print("-" * 40)
        
        # 创建示例张量
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, hidden_dim = 8, 128, 1024
        
        x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
        print("原始张量形状: {}".format(x.shape))
        
        with self.mesh:
            print("\n📊 不同分片策略对比：")
            
            # 策略1：数据并行
            data_parallel_spec = PartitionSpec('data', None, None)
            data_sharding = NamedSharding(self.mesh, data_parallel_spec)
            x_data_parallel = jax.device_put(x, data_sharding)
            
            print("\n1️⃣ 数据并行分片：")
            print("   分片规范: {}".format(data_parallel_spec))
            print("   含义: batch维度分片，其他维度完整")
            print("   每个GPU: ({}, {}, {})".format(
                batch_size//2, seq_len, hidden_dim))
            
            # 策略2：序列并行
            seq_parallel_spec = PartitionSpec(None, 'model', None)
            seq_sharding = NamedSharding(self.mesh, seq_parallel_spec)
            x_seq_parallel = jax.device_put(x, seq_sharding)
            
            print("\n2️⃣ 序列并行分片：")
            print("   分片规范: {}".format(seq_parallel_spec))
            print("   含义: 序列长度维度分片")
            print("   每个GPU: ({}, {}, {})".format(
                batch_size, seq_len//2, hidden_dim))
            
            # 策略3：隐藏层并行
            hidden_parallel_spec = PartitionSpec(None, None, 'model')
            hidden_sharding = NamedSharding(self.mesh, hidden_parallel_spec)
            x_hidden_parallel = jax.device_put(x, hidden_sharding)
            
            print("\n3️⃣ 隐藏层并行分片：")
            print("   分片规范: {}".format(hidden_parallel_spec))
            print("   含义: 隐藏维度分片")
            print("   每个GPU: ({}, {}, {})".format(
                batch_size, seq_len, hidden_dim//2))
            
            # 策略4：混合并行
            hybrid_spec = PartitionSpec('data', 'model', None)
            hybrid_sharding = NamedSharding(self.mesh, hybrid_spec)
            x_hybrid = jax.device_put(x, hybrid_sharding)
            
            print("\n4️⃣ 混合并行分片：")
            print("   分片规范: {}".format(hybrid_spec))
            print("   含义: batch和序列维度都分片")
            print("   每个GPU: ({}, {}, {})".format(
                batch_size//2, seq_len//2, hidden_dim))
    
    def explain_inference_patterns(self):
        """解释推理模式"""
        print("\n🚀 分布式推理模式详解")
        print("="*80)
        
        print("🔍 推理过程中的关键步骤：")
        
        print("\n1️⃣ 输入预处理：")
        print("   • 将输入数据分片到各GPU")
        print("   • 根据分片策略分布token")
        print("   • 处理attention mask")
        
        print("\n2️⃣ 前向传播：")
        print("   • 每个GPU并行计算局部结果")
        print("   • 根据需要插入通信操作")
        print("   • 自动处理维度匹配")
        
        print("\n3️⃣ 通信同步：")
        print("   • All-Reduce: 归约求和/平均")
        print("   • All-Gather: 收集所有分片")
        print("   • Reduce-Scatter: 归约后重分布")
        
        print("\n4️⃣ 输出聚合：")
        print("   • 合并各GPU的输出")
        print("   • 生成最终预测结果")
        print("   • 处理概率分布")
        
        if self.mesh:
            self._demonstrate_inference_flow()
    
    def _demonstrate_inference_flow(self):
        """演示推理流程"""
        print("\n🎬 实际推理流程演示")
        print("-" * 40)
        
        # 定义一个简单的推理函数
        def simple_inference(x, w1, w2):
            """简化的推理函数：Linear -> ReLU -> Linear"""
            h = jnp.dot(x, w1)  # 第一层线性变换
            h = jax.nn.relu(h)  # 激活函数
            return jnp.dot(h, w2)  # 第二层线性变换
        
        # JIT编译
        jit_inference = jax.jit(simple_inference)
        
        key = jax.random.PRNGKey(42)
        batch_size, input_dim, hidden_dim, output_dim = 8, 512, 1024, 256
        
        # 创建测试数据
        x = jax.random.normal(key, (batch_size, input_dim))
        w1 = jax.random.normal(key, (input_dim, hidden_dim))
        w2 = jax.random.normal(key, (hidden_dim, output_dim))
        
        with self.mesh:
            print("📝 推理函数：")
            print("   def inference(x, w1, w2):")
            print("       h = x @ w1    # 输入变换")
            print("       h = relu(h)   # 激活")
            print("       return h @ w2 # 输出变换")
            
            # 定义分片策略
            x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            w1_sharding = NamedSharding(self.mesh, PartitionSpec(None, 'model'))
            w2_sharding = NamedSharding(self.mesh, PartitionSpec('model', None))
            
            # 应用分片
            x_sharded = jax.device_put(x, x_sharding)
            w1_sharded = jax.device_put(w1, w1_sharding)
            w2_sharded = jax.device_put(w2, w2_sharding)
            
            print("\n🔀 分片配置：")
            print("   x:  {} → 按batch分片".format(x.shape))
            print("   w1: {} → 按hidden分片".format(w1.shape))
            print("   w2: {} → 按hidden分片".format(w2.shape))
            
            # 执行推理
            print("\n⚡ 执行分布式推理...")
            start_time = time.time()
            result = jit_inference(x_sharded, w1_sharded, w2_sharded)
            jax.block_until_ready(result)
            inference_time = time.time() - start_time
            
            print("✅ 推理完成！")
            print("   输入: {}".format(x.shape))
            print("   输出: {}".format(result.shape))
            print("   时间: {:.2f}ms".format(inference_time * 1000))
            print("   设备: {}个GPU并行".format(len(self.devices)))
    
    def explain_communication_patterns(self):
        """解释通信模式"""
        print("\n📡 分布式推理中的通信模式")
        print("="*80)
        
        print("🔄 JAX自动通信机制：")
        print("   JAX会在需要时自动插入通信操作")
        print("   用户无需手动管理通信")
        
        print("\n📋 常见通信操作：")
        
        print("\n1️⃣ All-Reduce (全归约):")
        print("   场景: 需要对所有GPU的结果求和/平均")
        print("   示例: 损失函数计算、梯度聚合")
        print("   ┌─────┐   ┌─────┐")
        print("   │ GPU0│──▶│ SUM │")
        print("   └─────┘   │     │")
        print("   ┌─────┐   │     │   ┌─────┐")
        print("   │ GPU1│──▶│     │──▶│ ALL │")
        print("   └─────┘   │     │   └─────┘")
        print("   ┌─────┐   │     │")
        print("   │ GPU2│──▶│     │")
        print("   └─────┘   └─────┘")
        
        print("\n2️⃣ All-Gather (全收集):")
        print("   场景: 需要收集所有GPU的数据片段")
        print("   示例: 序列长度分片后的重组")
        print("   GPU0: [A] ──┐")
        print("   GPU1: [B] ──┼──▶ [A,B,C,D] (全部)")
        print("   GPU2: [C] ──┘")
        print("   GPU3: [D] ──┘")
        
        print("\n3️⃣ Reduce-Scatter (归约分散):")
        print("   场景: 归约后按新策略分片")
        print("   示例: 注意力计算后的重分布")
        
        print("\n🚀 通信优化技术：")
        print("   • 通信/计算重叠：边算边传")
        print("   • 操作融合：减少通信次数")
        print("   • 带宽优化：高效利用网络")
        print("   • 拓扑感知：基于硬件布局优化")
    
    def demonstrate_performance_analysis(self):
        """演示性能分析"""
        print("\n📈 分布式推理性能分析")
        print("="*80)
        
        print("⚡ 性能影响因素：")
        
        print("\n1️⃣ 并行效率：")
        print("   理想情况: 4个GPU = 4倍速度")
        print("   实际情况: 通信开销导致效率下降")
        print("   目标: 最大化计算/通信比")
        
        print("\n2️⃣ 内存效率：")
        print("   模型并行: 节省内存，增加通信")
        print("   数据并行: 复制模型，节省通信")
        print("   混合策略: 平衡内存和通信")
        
        print("\n3️⃣ 批次大小影响：")
        print("   更大批次: 更好的GPU利用率")
        print("   分片批次: 每GPU处理部分数据")
        print("   权衡: 延迟 vs 吞吐量")
        
        if self.mesh and len(self.devices) >= 2:
            self._performance_comparison()
    
    def _performance_comparison(self):
        """性能对比测试"""
        print("\n🏁 性能对比测试")
        print("-" * 40)
        
        def simple_matmul(x, w):
            return jnp.dot(x, w)
        
        jit_matmul = jax.jit(simple_matmul)
        
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (64, 1024))
        w = jax.random.normal(key, (1024, 2048))
        
        with self.mesh:
            # 测试不同分片策略的性能
            strategies = [
                ("复制策略", PartitionSpec(), PartitionSpec()),
                ("数据分片", PartitionSpec('data', None), PartitionSpec()),
                ("模型分片", PartitionSpec(), PartitionSpec(None, 'model')),
            ]
            
            print("测试配置: {} @ {}".format(x.shape, w.shape))
            print("\n性能对比:")
            
            for name, x_spec, w_spec in strategies:
                try:
                    x_sharding = NamedSharding(self.mesh, x_spec)
                    w_sharding = NamedSharding(self.mesh, w_spec)
                    
                    x_sharded = jax.device_put(x, x_sharding)
                    w_sharded = jax.device_put(w, w_sharding)
                    
                    # 预热
                    for _ in range(3):
                        result = jit_matmul(x_sharded, w_sharded)
                        jax.block_until_ready(result)
                    
                    # 计时
                    times = []
                    for _ in range(5):
                        start = time.time()
                        result = jit_matmul(x_sharded, w_sharded)
                        jax.block_until_ready(result)
                        times.append(time.time() - start)
                    
                    avg_time = np.mean(times) * 1000
                    print("   {}: {:.2f}ms".format(name, avg_time))
                    
                except Exception as e:
                    print("   {}: 失败 - {}".format(name, str(e)))
    
    def explain_practical_considerations(self):
        """解释实际应用考虑"""
        print("\n💡 实际应用考虑")
        print("="*80)
        
        print("🎯 选择分片策略的原则：")
        
        print("\n1️⃣ 根据模型大小：")
        print("   小模型(< 1B): 数据并行优先")
        print("   中模型(1B-10B): 混合并行")
        print("   大模型(> 10B): 模型并行为主")
        
        print("\n2️⃣ 根据硬件配置：")
        print("   高内存GPU: 可用数据并行")
        print("   多GPU系统: 考虑混合并行")
        print("   网络带宽: 影响通信策略")
        
        print("\n3️⃣ 根据应用场景：")
        print("   在线推理: 低延迟优先")
        print("   批处理: 高吞吐量优先")
        print("   实时应用: 稳定性优先")
        
        print("\n🔧 最佳实践：")
        print("   ✅ 从简单策略开始")
        print("   ✅ 测试不同配置")
        print("   ✅ 监控GPU利用率")
        print("   ✅ 考虑内存限制")
        print("   ✅ 优化批次大小")
        
        print("\n⚠️ 常见陷阱：")
        print("   ❌ 过度分片导致通信开销")
        print("   ❌ 负载不均衡")
        print("   ❌ 忽略内存碎片")
        print("   ❌ 未考虑扩展性")
    
    def comprehensive_tutorial(self):
        """完整教程"""
        print("🎓 JAX分布式推理完整教程")
        print("="*80)
        
        # 逐步讲解
        self.explain_distributed_concepts()
        
        if self.demonstrate_device_mesh_creation():
            self.explain_sharding_strategies()
            self.explain_inference_patterns()
            self.explain_communication_patterns()
            self.demonstrate_performance_analysis()
        
        self.explain_practical_considerations()
        
        # 总结
        print("\n🎯 教程总结")
        print("="*60)
        print("✅ 核心概念:")
        print("   • 设备网格: 组织GPU的逻辑结构")
        print("   • 分片策略: 决定数据分布方式")
        print("   • 自动并行: JAX处理分布式细节")
        print("   • 通信优化: 最小化数据传输开销")
        
        print("\n🚀 关键优势:")
        print("   • 内存扩展: 突破单GPU限制")
        print("   • 性能提升: 多GPU并行加速")
        print("   • 易用性: 代码改动最小")
        print("   • 灵活性: 支持多种并行模式")
        
        print("\n💡 成功秘诀:")
        print("   • 理解模型特性选择策略")
        print("   • 平衡计算和通信开销")
        print("   • 充分测试不同配置")
        print("   • 持续监控性能指标")
        
        print("\n🎉 现在您已经掌握了JAX分布式推理的核心策略！")

def main():
    """主函数"""
    tutorial = JAXDistributedInferenceTutorial()
    tutorial.comprehensive_tutorial()

if __name__ == "__main__":
    main()
