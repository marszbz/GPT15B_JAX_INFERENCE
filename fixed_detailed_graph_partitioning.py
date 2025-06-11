#!/usr/bin/env python3
"""
图分割策略详细配置和实现
展示如何在JAX中进行精确的图分割和参数分片
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
# 清理可能存在的有问题的XLA_FLAGS
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
    print(f"✅ JAX {jax.__version__} 图分割模式加载成功")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class GraphPartitionConfig:
    """图分割配置"""
    num_devices: int = 4
    mesh_shape: Tuple[int, int] = (2, 2)
    data_axis: str = 'data'
    model_axis: str = 'model'
    parameter_threshold: int = 512  # 大于此阈值的参数进行分片
    
    # 分片策略配置
    embedding_sharding: str = 'vocab'      # vocab维度分片
    attention_sharding: str = 'heads'      # 注意力头分片
    mlp_sharding: str = 'hidden'           # 隐藏层分片
    output_sharding: str = 'vocab'         # 输出层分片

class DetailedGraphPartitioner:
    """详细的图分割实现"""
    
    def __init__(self, config: GraphPartitionConfig):
        self.config = config
        self.devices = jax.devices()
        if len(self.devices) > config.num_devices:
            self.devices = self.devices[:config.num_devices]
        self.mesh = None
        self.sharding_specs = {}
        
    def create_device_mesh(self):
        """创建设备网格"""
        print(f"\n🔧 创建设备网格")
        print("-" * 30)
        
        print(f"检测到设备数量: {len(self.devices)}")
        for i, device in enumerate(self.devices):
            print(f"   设备 {i}: {device}")
        
        try:
            if len(self.devices) >= 4:
                # 使用实际可用的设备数量
                actual_devices = self.devices[:4]
                print(f"使用设备数量: {len(actual_devices)}")
                
                # 手动创建2x2设备数组
                mesh_devices = np.array(actual_devices).reshape(2, 2)
                self.mesh = Mesh(mesh_devices, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建 (2,2) mesh")
                
            elif len(self.devices) == 2:
                # 2x1网格用于2个GPU
                mesh_devices = np.array(self.devices).reshape(2, 1)
                self.mesh = Mesh(mesh_devices, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建 (2,1) mesh")
                
            elif len(self.devices) == 1:
                # 单设备模式
                mesh_devices = np.array(self.devices).reshape(1, 1)
                self.mesh = Mesh(mesh_devices, axis_names=('data',))
                print(f"✅ 创建 (1,1) mesh (单设备模式)")
                # 更新配置以适应单设备
                self.config.model_axis = None
                
            else:
                print(f"⚠️ 设备数量({len(self.devices)})不支持，无法创建mesh")
                return False
                
            print(f"   网格形状: {self.mesh.shape}")
            print(f"   轴名称: {self.mesh.axis_names}")
            
            # 打印设备分配详情
            print(f"   设备分布:")
            if len(self.mesh.shape) == 2:
                for i in range(self.mesh.shape[0]):
                    for j in range(self.mesh.shape[1]):
                        device = self.mesh.devices[i, j]
                        print(f"     位置[{i},{j}]: {device}")
            else:
                for i, device in enumerate(self.mesh.devices.flat):
                    print(f"     位置[{i}]: {device}")
                
            return True
            
        except Exception as e:
            print(f"❌ 网格创建失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def define_sharding_strategies(self):
        """定义详细的分片策略"""
        print(f"\n📋 定义分片策略")
        print("-" * 30)
        
        if not self.mesh:
            print("⚠️ 未创建mesh，无法定义分片策略")
            return
        
        # 为不同组件定义分片规则
        strategies = {
            # 嵌入层分片策略
            'embedding': {
                'weight': PartitionSpec(self.config.model_axis, None),  # 词汇表维度分片
                'description': '词汇表在model轴分片，降低单设备内存压力'
            },
            
            # 注意力层分片策略
            'attention': {
                'qkv_weight': PartitionSpec(None, self.config.model_axis),  # 注意力头分片
                'qkv_bias': PartitionSpec(self.config.model_axis),
                'output_weight': PartitionSpec(self.config.model_axis, None),
                'output_bias': PartitionSpec(None),  # 复制到所有设备
                'description': '注意力头在model轴分片，实现头并行'
            },
            
            # MLP层分片策略
            'mlp': {
                'dense1_weight': PartitionSpec(None, self.config.model_axis),  # 隐藏层分片
                'dense1_bias': PartitionSpec(self.config.model_axis),
                'dense2_weight': PartitionSpec(self.config.model_axis, None),
                'dense2_bias': PartitionSpec(None),
                'description': 'MLP隐藏层在model轴分片，平衡计算负载'
            },
            
            # LayerNorm分片策略
            'layernorm': {
                'scale': PartitionSpec(None),  # 复制
                'bias': PartitionSpec(None),   # 复制
                'description': 'LayerNorm参数复制到所有设备'
            },
            
            # 输出层分片策略
            'output': {
                'weight': PartitionSpec(self.config.model_axis, None),  # 词汇表分片
                'description': '输出层词汇表维度分片'
            },
            
            # 数据分片策略
            'data': {
                'input_ids': PartitionSpec(self.config.data_axis, None),     # batch分片
                'attention_mask': PartitionSpec(self.config.data_axis, None, None),
                'logits': PartitionSpec(self.config.data_axis, None, self.config.model_axis),
                'description': '输入数据在data轴分片，实现数据并行'
            }
        }
        
        # 处理单设备情况
        if self.config.model_axis is None:
            for component, specs in strategies.items():
                for param_name, spec in specs.items():
                    if param_name != 'description' and isinstance(spec, PartitionSpec):
                        # 将model轴分片改为无分片
                        new_spec_args = []
                        for axis in spec:
                            if axis == 'model':
                                new_spec_args.append(None)
                            else:
                                new_spec_args.append(axis)
                        strategies[component][param_name] = PartitionSpec(*new_spec_args)
        
        self.sharding_specs = strategies
        
        # 打印分片策略
        for component, specs in strategies.items():
            print(f"\n🔍 {component.upper()} 分片策略:")
            print(f"   描述: {specs['description']}")
            for param_name, spec in specs.items():
                if param_name != 'description':
                    print(f"   {param_name}: {spec}")
    
    def demonstrate_xla_optimizations(self):
        """演示XLA编译器优化"""
        print(f"\n⚡ XLA编译器优化演示")
        print("-" * 30)
        
        if not self.mesh:
            print("⚠️ 未创建mesh，无法演示XLA优化")
            return
        
        # 创建简单的计算图
        def simple_computation(x, w):
            """简单的矩阵乘法 + 激活函数"""
            return jax.nn.gelu(jnp.dot(x, w))
        
        # JIT编译
        jit_computation = jax.jit(simple_computation)
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (32, 1600))  # batch_size x hidden_dim
        w = jax.random.normal(key, (1600, 4800))  # hidden_dim x 3*hidden_dim
        
        print(f"📊 计算图分析:")
        print(f"   输入形状: {x.shape}")
        print(f"   权重形状: {w.shape}")
        print(f"   计算类型: 矩阵乘法 + GELU激活")
        
        # 分片数据
        with self.mesh:
            if self.config.model_axis:
                x_sharding = NamedSharding(self.mesh, PartitionSpec(self.config.data_axis, None))
                w_sharding = NamedSharding(self.mesh, PartitionSpec(None, self.config.model_axis))
            else:
                x_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
                w_sharding = NamedSharding(self.mesh, PartitionSpec(None))
            
            x_sharded = jax.device_put(x, x_sharding)
            w_sharded = jax.device_put(w, w_sharding)
            
            print(f"\n🔧 XLA优化过程:")
            print(f"   1. 图构建: 解析Python代码为XLA HLO")
            print(f"   2. 优化器: 应用融合、重排列等优化")
            print(f"   3. 并行化: 根据分片策略分布计算")
            print(f"   4. 代码生成: 生成高效的GPU kernel")
            
            # 预热JIT编译
            print(f"\n🚀 JIT编译预热...")
            for i in range(3):
                result = jit_computation(x_sharded, w_sharded)
                jax.block_until_ready(result)
                print(f"   预热 {i+1}/3 完成")
            
            # 性能测试
            print(f"\n📈 性能测试:")
            times = []
            for i in range(5):
                start_time = time.time()
                result = jit_computation(x_sharded, w_sharded)
                jax.block_until_ready(result)
                end_time = time.time()
                times.append(end_time - start_time)
                print(f"   运行 {i+1}: {(end_time - start_time)*1000:.2f}ms")
            
            avg_time = np.mean(times)
            throughput = (32 * 1600 * 4800) / avg_time / 1e9  # GFLOPS
            
            print(f"\n🎯 XLA优化效果:")
            print(f"   平均执行时间: {avg_time*1000:.2f}ms")
            print(f"   计算吞吐量: {throughput:.2f} GFLOPS")
            print(f"   输出形状: {result.shape}")
            print(f"   内存效率: 参数自动分片到多GPU")
            
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_gflops': throughput,
            'output_shape': result.shape
        }
    
    def analyze_parameter_distribution(self, model_config=None):
        """分析参数分布"""
        print(f"\n📊 参数分布分析")
        print("-" * 30)
        
        # 模拟GPT-1.5B参数
        vocab_size = 50257
        n_embd = 1600
        n_layer = 48
        n_head = 25
        
        # 分析每个设备的参数分布
        device_params = {f'device_{i}': 0 for i in range(len(self.devices))}
        total_params = 0
        
        print(f"📈 参数分布统计:")
        
        # 计算嵌入层参数
        embed_params = vocab_size * n_embd + 2048 * n_embd
        if len(self.devices) > 1 and self.config.model_axis:
            sharded_embed = embed_params // len(self.devices)  # 词汇表分片
        else:
            sharded_embed = embed_params  # 单设备或无分片
            
        for i in range(len(self.devices)):
            if len(self.devices) > 1 and self.config.model_axis:
                device_params[f'device_{i}'] += sharded_embed
            else:
                device_params[f'device_0'] += embed_params
                break
        total_params += embed_params
        print(f"   嵌入层: {embed_params:,} 参数 → 每设备: {sharded_embed:,}")
        
        # 计算Transformer层参数
        layer_param_count = (
            n_embd * 3 * n_embd +  # QKV
            n_embd * n_embd +      # attention output
            n_embd * 4 * n_embd +  # MLP dense1
            4 * n_embd * n_embd +  # MLP dense2
            4 * n_embd             # biases + layernorms
        )
        
        transformer_params = n_layer * layer_param_count
        if len(self.devices) > 1 and self.config.model_axis:
            sharded_transformer = transformer_params // len(self.devices)
        else:
            sharded_transformer = transformer_params
            
        for i in range(len(self.devices)):
            if len(self.devices) > 1 and self.config.model_axis:
                device_params[f'device_{i}'] += sharded_transformer
            else:
                device_params[f'device_0'] += transformer_params
                break
        total_params += transformer_params
        print(f"   Transformer: {transformer_params:,} 参数 → 每设备: {sharded_transformer:,}")
        
        # 计算输出层参数
        output_params = n_embd * vocab_size
        if len(self.devices) > 1 and self.config.model_axis:
            sharded_output = output_params // len(self.devices)
        else:
            sharded_output = output_params
            
        for i in range(len(self.devices)):
            if len(self.devices) > 1 and self.config.model_axis:
                device_params[f'device_{i}'] += sharded_output
            else:
                device_params[f'device_0'] += output_params
                break
        total_params += output_params
        print(f"   输出层: {output_params:,} 参数 → 每设备: {sharded_output:,}")
        
        print(f"\n📊 设备负载平衡:")
        for device, count in device_params.items():
            if count > 0:  # 只显示有参数的设备
                percentage = (count / total_params) * 100
                memory_gb = count * 4 / (1024**3)  # float32
                print(f"   {device}: {count:,} 参数 ({percentage:.1f}%) ≈ {memory_gb:.2f}GB")
        
        print(f"\n📈 总体统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        
        active_devices = sum(1 for count in device_params.values() if count > 0)
        if active_devices > 1:
            print(f"   平均每设备: {total_params//active_devices:,}")
            min_params = min(count for count in device_params.values() if count > 0)
            max_params = max(count for count in device_params.values() if count > 0)
            print(f"   负载均衡度: {(min_params/max_params)*100:.1f}%")
        else:
            print(f"   单设备模式: 所有参数在一个设备上")
        
        return {
            'total_params': total_params,
            'device_distribution': device_params,
            'memory_per_device_gb': max(device_params.values()) * 4 / (1024**3)
        }
    
    def demonstrate_sharding_execution(self):
        """演示分片执行过程"""
        print(f"\n🚀 分片执行演示")
        print("-" * 30)
        
        if not self.mesh:
            print("⚠️ 未创建mesh，无法演示分片")
            return None
            
        # 创建模拟的模型参数
        key = jax.random.PRNGKey(42)
        
        # 模拟参数字典
        params = {
            'embedding': {
                'weight': jax.random.normal(key, (50257, 1600))  # 词汇表 x 嵌入维度
            },
            'attention': {
                'qkv_weight': jax.random.normal(key, (1600, 4800)),  # 3 * n_embd
                'output_weight': jax.random.normal(key, (1600, 1600))
            },
            'mlp': {
                'dense1_weight': jax.random.normal(key, (1600, 6400)),  # 4 * n_embd
                'dense2_weight': jax.random.normal(key, (6400, 1600))
            }
        }
        
        print(f"📦 原始参数形状:")
        for component, comp_params in params.items():
            for param_name, param in comp_params.items():
                print(f"   {component}.{param_name}: {param.shape}")
        
        # 应用分片
        with self.mesh:
            sharded_params = {}
            
            for component, comp_params in params.items():
                sharded_params[component] = {}
                
                for param_name, param in comp_params.items():
                    # 根据组件类型选择分片策略
                    if component == 'embedding' and 'weight' in param_name:
                        if self.config.model_axis:
                            spec = PartitionSpec(self.config.model_axis, None)  # 词汇表分片
                        else:
                            spec = PartitionSpec()  # 单设备不分片
                    elif component == 'attention':
                        if 'qkv' in param_name:
                            if self.config.model_axis:
                                spec = PartitionSpec(None, self.config.model_axis)  # 注意力头分片
                            else:
                                spec = PartitionSpec()
                        else:
                            if self.config.model_axis:
                                spec = PartitionSpec(self.config.model_axis, None)
                            else:
                                spec = PartitionSpec()
                    elif component == 'mlp':
                        if 'dense1' in param_name:
                            if self.config.model_axis:
                                spec = PartitionSpec(None, self.config.model_axis)  # 隐藏层分片
                            else:
                                spec = PartitionSpec()
                        else:
                            if self.config.model_axis:
                                spec = PartitionSpec(self.config.model_axis, None)
                            else:
                                spec = PartitionSpec()
                    else:
                        spec = PartitionSpec()  # 不分片
                    
                    # 创建分片
                    sharding = NamedSharding(self.mesh, spec)
                    sharded_param = jax.device_put(param, sharding)
                    sharded_params[component][param_name] = sharded_param
                    
                    print(f"   ✅ {component}.{param_name}: {spec} → 已分片")
        
        print(f"\n🎯 分片执行完成!")
        return sharded_params

def main():
    """主函数 - 详细图分割分析"""
    print(f"🔍 详细图分割策略分析")
    print("=" * 50)
    
    # 创建配置
    available_devices = len(jax.devices())
    config = GraphPartitionConfig(
        num_devices=available_devices,
        mesh_shape=(2, 2) if available_devices >= 4 else (2, 1)
    )
    
    partitioner = DetailedGraphPartitioner(config)
    
    try:
        # 1. 创建设备网格
        mesh_success = partitioner.create_device_mesh()
        
        if mesh_success:
            # 2. 定义分片策略
            partitioner.define_sharding_strategies()
            
            # 3. 演示XLA优化
            xla_results = partitioner.demonstrate_xla_optimizations()
            
            # 4. 分析参数分布
            param_analysis = partitioner.analyze_parameter_distribution()
            
            # 5. 演示分片执行
            sharded_params = partitioner.demonstrate_sharding_execution()
            
            # 6. 保存结果
            results = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'config': {
                    'num_devices': config.num_devices,
                    'mesh_shape': config.mesh_shape,
                    'data_axis': config.data_axis,
                    'model_axis': config.model_axis
                },
                'mesh_info': {
                    'shape': list(partitioner.mesh.shape),
                    'axis_names': list(partitioner.mesh.axis_names),
                    'device_count': len(partitioner.devices)
                },
                'xla_optimization': xla_results,
                'parameter_analysis': param_analysis,
                'sharding_successful': sharded_params is not None
            }
            
            results_file = Path("detailed_graph_partition_analysis.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 详细分析结果已保存: {results_file}")
            
            # 7. 总结
            print(f"\n🎯 图分割分析总结")
            print("=" * 40)
            print(f"📊 关键指标:")
            print(f"   设备数量: {len(partitioner.devices)}")
            print(f"   网格配置: {partitioner.mesh.shape}")
            print(f"   总参数量: {param_analysis['total_params']/1e9:.2f}B")
            print(f"   每设备内存: {param_analysis['memory_per_device_gb']:.2f}GB")
            
            if xla_results:
                print(f"   XLA吞吐量: {xla_results['throughput_gflops']:.1f} GFLOPS")
                print(f"   平均延迟: {xla_results['avg_time_ms']:.2f}ms")
            
            print(f"\n💡 XLA编译器优化特性:")
            print(f"   1. 自动图融合: 将小操作合并为大kernel")
            print(f"   2. 内存优化: 减少中间结果的内存占用")
            print(f"   3. 并行优化: 根据mesh自动分布计算")
            print(f"   4. 快速数学: 使用近似但更快的数学函数")
            
            print(f"\n🔧 图分割核心技术:")
            print(f"   1. 参数分片: 大权重矩阵分布到多GPU")
            print(f"   2. 数据并行: batch维度分片到不同设备")
            print(f"   3. 模型并行: 注意力头/MLP分片")
            print(f"   4. 混合并行: 数据+模型并行结合")
            
        else:
            print(f"❌ 无法创建设备网格，请检查GPU设备数量")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
