#!/usr/bin/env python3
"""
干净的图分割策略实现
展示JAX中的XLA编译器优化和图分割机制
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
# 清理XLA_FLAGS避免错误
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
    print(f"✅ JAX {jax.__version__} 加载成功")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class GraphPartitionConfig:
    """图分割配置"""
    num_devices: int = 4
    data_axis: str = 'data'
    model_axis: str = 'model'

class XLAGraphPartitioner:
    """XLA图分割和优化演示"""
    
    def __init__(self, config: GraphPartitionConfig):
        self.config = config
        self.devices = jax.devices()
        self.mesh = None
        
    def create_device_mesh(self):
        """创建设备网格"""
        print(f"\n🔧 创建设备网格")
        print("-" * 30)
        
        num_devices = len(self.devices)
        print(f"可用设备数量: {num_devices}")
        
        for i, device in enumerate(self.devices):
            print(f"   设备 {i}: {device}")
        
        try:
            if num_devices >= 4:
                # 4个GPU: 2x2网格
                devices_array = np.array(self.devices[:4]).reshape(2, 2)
                self.mesh = Mesh(devices_array, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建2x2网格成功")
                
            elif num_devices == 2:
                # 2个GPU: 2x1网格  
                devices_array = np.array(self.devices).reshape(2, 1)
                self.mesh = Mesh(devices_array, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建2x1网格成功")
                
            elif num_devices == 1:
                # 单GPU
                devices_array = np.array(self.devices).reshape(1, 1)
                self.mesh = Mesh(devices_array, axis_names=('data',))
                self.config.model_axis = None  # 单设备无模型并行
                print(f"✅ 创建单设备网格")
                
            else:
                print(f"⚠️ 不支持的设备数量: {num_devices}")
                return False
            
            # 显示网格信息
            print(f"   网格形状: {dict(self.mesh.shape)}")
            print(f"   轴名称: {self.mesh.axis_names}")
            
            # 显示设备分布
            print(f"   设备分布:")
            if self.mesh.devices.ndim == 2:
                rows, cols = self.mesh.devices.shape
                for i in range(rows):
                    for j in range(cols):
                        device = self.mesh.devices[i, j]
                        print(f"     [{i},{j}]: {device}")
            else:
                for i, device in enumerate(self.mesh.devices.flat):
                    print(f"     [{i}]: {device}")
                    
            return True
            
        except Exception as e:
            print(f"❌ 网格创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demonstrate_xla_optimizations(self):
        """演示XLA编译器优化"""
        print(f"\n⚡ XLA编译器优化演示")
        print("-" * 30)
        
        if not self.mesh:
            print("⚠️ 需要先创建mesh")
            return None
        
        # 定义简单的计算图
        def simple_gpt_layer(x, w1, w2):
            """模拟GPT层的计算"""
            # 矩阵乘法 + 激活函数 + 另一个矩阵乘法
            h = jnp.dot(x, w1)  # 线性变换
            h = jax.nn.gelu(h)   # 激活函数
            return jnp.dot(h, w2)  # 输出投影
        
        # JIT编译
        jit_layer = jax.jit(simple_gpt_layer)
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, hidden_dim = 32, 512, 1600
        
        x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
        w1 = jax.random.normal(key, (hidden_dim, hidden_dim * 4))
        w2 = jax.random.normal(key, (hidden_dim * 4, hidden_dim))
        
        print(f"📊 计算图分析:")
        print(f"   输入: {x.shape}")
        print(f"   权重1: {w1.shape}")  
        print(f"   权重2: {w2.shape}")
        print(f"   操作: MatMul → GELU → MatMul")
        
        # 分片数据到设备
        with self.mesh:
            if self.config.model_axis:
                # 多设备分片
                x_sharding = NamedSharding(self.mesh, PartitionSpec(self.config.data_axis, None, None))
                w1_sharding = NamedSharding(self.mesh, PartitionSpec(None, self.config.model_axis))
                w2_sharding = NamedSharding(self.mesh, PartitionSpec(self.config.model_axis, None))
            else:
                # 单设备
                x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None, None))
                w1_sharding = NamedSharding(self.mesh, PartitionSpec(None, None))
                w2_sharding = NamedSharding(self.mesh, PartitionSpec(None, None))
            
            x_sharded = jax.device_put(x, x_sharding)
            w1_sharded = jax.device_put(w1, w1_sharding)
            w2_sharded = jax.device_put(w2, w2_sharding)
            
            print(f"\n🔧 XLA优化流程:")
            print(f"   1. 图构建: Python → HLO (High Level Operations)")
            print(f"   2. 图优化: 操作融合、内存优化、并行化")
            print(f"   3. 代码生成: HLO → GPU kernels")
            print(f"   4. 执行: 高效的GPU计算")
            
            # JIT预热
            print(f"\n🚀 JIT编译预热...")
            for i in range(3):
                result = jit_layer(x_sharded, w1_sharded, w2_sharded)
                jax.block_until_ready(result)
                print(f"   预热 {i+1}/3 完成")
            
            # 性能测试
            print(f"\n📈 性能基准测试:")
            times = []
            for i in range(5):
                start_time = time.time()
                result = jit_layer(x_sharded, w1_sharded, w2_sharded)
                jax.block_until_ready(result)
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                print(f"   测试 {i+1}: {elapsed*1000:.2f}ms")
            
            # 计算统计
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # 计算吞吐量 (FLOPS)
            flops = batch_size * seq_len * (
                2 * hidden_dim * hidden_dim * 4 +  # 第一个MatMul
                hidden_dim * 4 +                   # GELU (近似)
                2 * hidden_dim * 4 * hidden_dim    # 第二个MatMul
            )
            throughput_gflops = flops / avg_time / 1e9
            
            print(f"\n🎯 XLA优化效果:")
            print(f"   平均时间: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"   吞吐量: {throughput_gflops:.1f} GFLOPS")
            print(f"   输出形状: {result.shape}")
            print(f"   内存效率: 自动分片管理")
            
            return {
                'avg_time_ms': avg_time * 1000,
                'throughput_gflops': throughput_gflops,
                'output_shape': result.shape
            }
    
    def analyze_sharding_strategies(self):
        """分析分片策略"""
        print(f"\n📋 分片策略分析")
        print("-" * 30)
        
        if not self.mesh:
            print("⚠️ 需要先创建mesh")
            return None
        
        # 定义不同组件的分片策略
        strategies = {
            'embedding': {
                'token_emb': PartitionSpec(self.config.model_axis, None),  # 词汇表分片
                'pos_emb': PartitionSpec(None, None),  # 不分片
                'description': '词汇表维度分片，减少单设备内存'
            },
            'attention': {
                'qkv_proj': PartitionSpec(None, self.config.model_axis),  # 头维度分片
                'out_proj': PartitionSpec(self.config.model_axis, None),
                'description': '注意力头并行，提高计算效率'
            },
            'mlp': {
                'up_proj': PartitionSpec(None, self.config.model_axis),    # 隐藏层分片
                'down_proj': PartitionSpec(self.config.model_axis, None),
                'description': 'MLP层分片，平衡计算负载'
            },
            'data': {
                'input_tokens': PartitionSpec(self.config.data_axis, None),  # batch分片
                'attention_mask': PartitionSpec(self.config.data_axis, None, None),
                'description': '数据并行，提高训练效率'
            }
        }
        
        # 处理单设备情况
        if not self.config.model_axis:
            for component in strategies:
                for key, spec in strategies[component].items():
                    if isinstance(spec, PartitionSpec) and key != 'description':
                        # 移除model轴分片
                        new_spec = []
                        for axis in spec:
                            if axis != 'model':
                                new_spec.append(axis)
                            else:
                                new_spec.append(None)
                        strategies[component][key] = PartitionSpec(*new_spec)
        
        print(f"📊 分片策略详情:")
        for component, specs in strategies.items():
            print(f"\n🔍 {component.upper()}:")
            print(f"   策略: {specs['description']}")
            for param, spec in specs.items():
                if param != 'description':
                    print(f"   {param}: {spec}")
        
        return strategies
    
    def estimate_performance(self):
        """性能估计"""
        print(f"\n📈 性能估计分析")
        print("-" * 30)
        
        # RTX 3090规格
        gpu_memory_gb = 24
        gpu_tflops = 35.6
        
        # GPT-1.5B模型规格
        param_count = 1.5e9
        param_memory_gb = param_count * 4 / (1024**3)  # float32
        
        scenarios = {
            '单GPU模式': {
                'devices': 1,
                'memory_per_gpu': param_memory_gb,
                'efficiency': 0.6,
                'communication': 0.0
            },
            '数据并行': {
                'devices': len(self.devices),
                'memory_per_gpu': param_memory_gb,  # 每GPU完整模型
                'efficiency': 0.8,
                'communication': 0.1
            },
            '模型并行': {
                'devices': len(self.devices),
                'memory_per_gpu': param_memory_gb / len(self.devices),
                'efficiency': 0.7,
                'communication': 0.2
            },
            '混合并行': {
                'devices': len(self.devices),
                'memory_per_gpu': param_memory_gb / 2,
                'efficiency': 0.85,
                'communication': 0.15
            }
        }
        
        print(f"💾 内存和性能对比:")
        for scenario, config in scenarios.items():
            memory_util = (config['memory_per_gpu'] / gpu_memory_gb) * 100
            theoretical_tflops = gpu_tflops * config['devices']
            effective_tflops = theoretical_tflops * config['efficiency'] * (1 - config['communication'])
            speedup = effective_tflops / (gpu_tflops * 0.6)
            
            print(f"\n   {scenario}:")
            print(f"     内存/GPU: {config['memory_per_gpu']:.2f}GB ({memory_util:.1f}%)")
            print(f"     理论算力: {theoretical_tflops:.1f} TFLOPS")
            print(f"     有效算力: {effective_tflops:.1f} TFLOPS")
            print(f"     加速比: {speedup:.2f}x")
        
        return scenarios

def main():
    """主函数"""
    print(f"🔍 XLA编译器图优化和分割分析")
    print("=" * 50)
    
    # 创建配置
    config = GraphPartitionConfig(num_devices=len(jax.devices()))
    partitioner = XLAGraphPartitioner(config)
    
    try:
        # 1. 创建设备网格
        if not partitioner.create_device_mesh():
            print("❌ 设备网格创建失败")
            return
        
        # 2. XLA优化演示
        xla_results = partitioner.demonstrate_xla_optimizations()
        
        # 3. 分片策略分析
        sharding_strategies = partitioner.analyze_sharding_strategies()
        
        # 4. 性能估计
        performance_scenarios = partitioner.estimate_performance()
        
        # 5. 保存结果
        results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'device_count': len(partitioner.devices),
            'mesh_shape': dict(partitioner.mesh.shape) if partitioner.mesh else None,
            'xla_performance': xla_results,
            'sharding_strategies': {
                component: {k: str(v) for k, v in specs.items()}
                for component, specs in sharding_strategies.items()
            } if sharding_strategies else None,
            'performance_scenarios': performance_scenarios
        }
        
        results_file = Path("xla_graph_optimization_analysis.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 分析结果已保存: {results_file}")
        
        # 6. 总结
        print(f"\n🎯 XLA图优化总结")
        print("=" * 40)
        print(f"📊 系统配置:")
        print(f"   设备数量: {len(partitioner.devices)}")
        print(f"   网格布局: {dict(partitioner.mesh.shape)}")
        
        if xla_results:
            print(f"   XLA性能: {xla_results['throughput_gflops']:.1f} GFLOPS")
            print(f"   平均延迟: {xla_results['avg_time_ms']:.2f}ms")
        
        print(f"\n💡 XLA核心优化:")
        print(f"   1. 操作融合: 减少内存访问和kernel启动开销")
        print(f"   2. 内存优化: 智能的缓冲区分配和重用")
        print(f"   3. 并行优化: 自动的设备间负载均衡")
        print(f"   4. 图优化: 消除冗余操作和计算")
        
        print(f"\n🔧 图分割技术:")
        print(f"   1. 数据并行: batch维度分片到多设备")
        print(f"   2. 模型并行: 参数矩阵按维度分割")
        print(f"   3. 流水线并行: 层间计算重叠")
        print(f"   4. 混合并行: 多种策略组合使用")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
