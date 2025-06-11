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
    print(f"✅ JAX {jax.__version__} 图分割模式")
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
        self.devices = jax.devices()[:config.num_devices]
        self.mesh = None
        self.sharding_specs = {}
        
    def create_device_mesh(self):
        """创建设备网格"""
        print(f"\n🔧 创建设备网格")
        print("-" * 30)
        
        try:
            if len(self.devices) >= 4:
                # 2x2网格用于4个GPU
                mesh_devices = mesh_utils.create_device_mesh(self.config.mesh_shape)
                self.mesh = Mesh(mesh_devices, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建 {self.config.mesh_shape} mesh")
            elif len(self.devices) == 2:
                # 2x1网格用于2个GPU
                mesh_devices = mesh_utils.create_device_mesh((2, 1))
                self.mesh = Mesh(mesh_devices, axis_names=(self.config.data_axis, self.config.model_axis))
                print(f"✅ 创建 (2,1) mesh")
            else:
                print(f"⚠️ 设备数量不足，无法创建mesh")
                return False
                
            print(f"   设备网格形状: {self.mesh.shape}")
            print(f"   轴名称: {self.mesh.axis_names}")
            
            # 打印设备分配
            for i, device in enumerate(self.mesh.devices.flat):
                row, col = divmod(i, self.mesh.shape[1])
                print(f"   位置({row},{col}): {device}")
                
            return True
            
        except Exception as e:
            print(f"❌ 网格创建失败: {e}")
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
        
        self.sharding_specs = strategies
        
        # 打印分片策略
        for component, specs in strategies.items():
            print(f"\n🔍 {component.upper()} 分片策略:")
            print(f"   描述: {specs['description']}")
            for param_name, spec in specs.items():
                if param_name != 'description':
                    print(f"   {param_name}: {spec}")
    
    def analyze_parameter_distribution(self, model_config):
        """分析参数分布"""
        print(f"\n📊 参数分布分析")
        print("-" * 30)
        
        # 模拟GPT-1.5B参数
        vocab_size = 50257
        n_embd = 1600
        n_layer = 48
        n_head = 25
        
        params = {
            'embedding': {
                'token_embedding': (vocab_size, n_embd),
                'position_embedding': (2048, n_embd)
            },
            'transformer_blocks': {},
            'output': {
                'lm_head': (n_embd, vocab_size)
            }
        }
        
        # 每个Transformer块的参数
        for layer_idx in range(n_layer):
            layer_params = {
                'attention': {
                    'qkv_weight': (n_embd, 3 * n_embd),
                    'qkv_bias': (3 * n_embd,),
                    'output_weight': (n_embd, n_embd),
                    'output_bias': (n_embd,)
                },
                'mlp': {
                    'dense1_weight': (n_embd, 4 * n_embd),
                    'dense1_bias': (4 * n_embd,),
                    'dense2_weight': (4 * n_embd, n_embd),
                    'dense2_bias': (n_embd,)
                },
                'layernorm1': {
                    'scale': (n_embd,),
                    'bias': (n_embd,)
                },
                'layernorm2': {
                    'scale': (n_embd,),
                    'bias': (n_embd,)
                }
            }
            params['transformer_blocks'][f'layer_{layer_idx}'] = layer_params
        
        # 分析每个设备的参数分布
        device_params = {f'device_{i}': 0 for i in range(len(self.devices))}
        total_params = 0
        
        print(f"📈 参数分布统计:")
        
        # 计算嵌入层参数
        embed_params = vocab_size * n_embd + 2048 * n_embd
        sharded_embed = embed_params // len(self.devices)  # 词汇表分片
        for i in range(len(self.devices)):
            device_params[f'device_{i}'] += sharded_embed
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
        # 注意力头和MLP分片
        sharded_transformer = transformer_params // len(self.devices)
        for i in range(len(self.devices)):
            device_params[f'device_{i}'] += sharded_transformer
        total_params += transformer_params
        print(f"   Transformer: {transformer_params:,} 参数 → 每设备: {sharded_transformer:,}")
        
        # 计算输出层参数
        output_params = n_embd * vocab_size
        sharded_output = output_params // len(self.devices)  # 词汇表分片
        for i in range(len(self.devices)):
            device_params[f'device_{i}'] += sharded_output
        total_params += output_params
        print(f"   输出层: {output_params:,} 参数 → 每设备: {sharded_output:,}")
        
        print(f"\n📊 设备负载平衡:")
        for device, count in device_params.items():
            percentage = (count / total_params) * 100
            memory_gb = count * 4 / (1024**3)  # float32
            print(f"   {device}: {count:,} 参数 ({percentage:.1f}%) ≈ {memory_gb:.2f}GB")
        
        print(f"\n📈 总体统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   平均每设备: {total_params//len(self.devices):,}")
        print(f"   负载均衡度: {(min(device_params.values())/max(device_params.values()))*100:.1f}%")
        
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
                        spec = PartitionSpec(self.config.model_axis, None)  # 词汇表分片
                    elif component == 'attention':
                        if 'qkv' in param_name:
                            spec = PartitionSpec(None, self.config.model_axis)  # 注意力头分片
                        else:
                            spec = PartitionSpec(self.config.model_axis, None)
                    elif component == 'mlp':
                        if 'dense1' in param_name:
                            spec = PartitionSpec(None, self.config.model_axis)  # 隐藏层分片
                        else:
                            spec = PartitionSpec(self.config.model_axis, None)
                    else:
                        spec = PartitionSpec()  # 不分片
                    
                    # 创建分片
                    sharding = NamedSharding(self.mesh, spec)
                    sharded_param = jax.device_put(param, sharding)
                    sharded_params[component][param_name] = sharded_param
                    
                    print(f"   ✅ {component}.{param_name}: {spec} → 已分片")
        
        print(f"\n🎯 分片执行完成!")
        return sharded_params
    
    def create_performance_prediction(self):
        """创建性能预测"""
        print(f"\n📈 性能预测分析")
        print("-" * 30)
        
        # 基于4个RTX 3090的性能预测
        gpu_memory_gb = 24
        gpu_compute_tflops = 35.6  # RTX 3090理论峰值
        
        # 模型配置
        vocab_size = 50257
        n_embd = 1600
        n_layer = 48
        batch_size = 32
        seq_len = 512
        
        # 计算内存需求
        param_memory = 1.5e9 * 4 / (1024**3)  # 1.5B参数，float32
        activation_memory = batch_size * seq_len * n_embd * 4 / (1024**3)
        
        # 单GPU vs 多GPU对比
        scenarios = {
            '单GPU': {
                'devices': 1,
                'param_memory_per_gpu': param_memory,
                'activation_memory_per_gpu': activation_memory,
                'compute_efficiency': 0.6,  # 单GPU效率
                'communication_overhead': 0.0
            },
            '数据并行(4GPU)': {
                'devices': 4,
                'param_memory_per_gpu': param_memory,  # 每个GPU都有完整参数
                'activation_memory_per_gpu': activation_memory / 4,  # 激活值分片
                'compute_efficiency': 0.8,  # 数据并行效率
                'communication_overhead': 0.1  # AllReduce通信
            },
            '模型并行(4GPU)': {
                'devices': 4,
                'param_memory_per_gpu': param_memory / 4,  # 参数分片
                'activation_memory_per_gpu': activation_memory,  # 完整激活值
                'compute_efficiency': 0.7,  # 模型并行效率（通信开销较大）
                'communication_overhead': 0.2  # 参数通信
            },
            '混合并行(4GPU)': {
                'devices': 4,
                'param_memory_per_gpu': param_memory / 2,  # 2x2混合
                'activation_memory_per_gpu': activation_memory / 2,
                'compute_efficiency': 0.85,  # 最优效率
                'communication_overhead': 0.15  # 平衡的通信
            }
        }
        
        print(f"💾 内存使用预测:")
        for scenario, config in scenarios.items():
            total_memory = config['param_memory_per_gpu'] + config['activation_memory_per_gpu']
            memory_utilization = (total_memory / gpu_memory_gb) * 100
            
            print(f"\n   {scenario}:")
            print(f"     参数内存: {config['param_memory_per_gpu']:.2f}GB/GPU")
            print(f"     激活内存: {config['activation_memory_per_gpu']:.2f}GB/GPU")
            print(f"     总内存: {total_memory:.2f}GB/GPU")
            print(f"     内存利用率: {memory_utilization:.1f}%")
            
            # 性能预测
            theoretical_tflops = gpu_compute_tflops * config['devices']
            effective_tflops = theoretical_tflops * config['compute_efficiency'] * (1 - config['communication_overhead'])
            speedup = effective_tflops / (gpu_compute_tflops * 0.6)  # 相对于单GPU
            
            print(f"     理论算力: {theoretical_tflops:.1f} TFLOPS")
            print(f"     有效算力: {effective_tflops:.1f} TFLOPS")
            print(f"     相对加速: {speedup:.2f}x")
        
        return scenarios

def main():
    """主函数 - 详细图分割分析"""
    print(f"🔍 详细图分割策略分析")
    print("=" * 50)
    
    # 创建配置
    config = GraphPartitionConfig(
        num_devices=len(jax.devices()),
        mesh_shape=(2, 2) if len(jax.devices()) >= 4 else (2, 1)
    )
    
    partitioner = DetailedGraphPartitioner(config)
    
    try:
        # 1. 创建设备网格
        mesh_success = partitioner.create_device_mesh()
        
        if mesh_success:
            # 2. 定义分片策略
            partitioner.define_sharding_strategies()
            
            # 3. 分析参数分布
            param_analysis = partitioner.analyze_parameter_distribution(config)
            
            # 4. 演示分片执行
            sharded_params = partitioner.demonstrate_sharding_execution()
            
            # 5. 性能预测
            performance_prediction = partitioner.create_performance_prediction()
            
            # 6. 保存结果
            results = {
                'config': config.__dict__,
                'mesh_info': {
                    'shape': partitioner.mesh.shape,
                    'axis_names': partitioner.mesh.axis_names,
                    'device_count': len(partitioner.devices)
                },
                'parameter_analysis': param_analysis,
                'performance_prediction': performance_prediction,
                'sharding_specs': {
                    component: {k: str(v) for k, v in specs.items() if k != 'description'}
                    for component, specs in partitioner.sharding_specs.items()
                }
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
            
            print(f"\n💡 关键优势:")
            print(f"   1. 精确的参数分片减少单设备内存压力")
            print(f"   2. 注意力头并行提高计算效率")
            print(f"   3. 混合并行策略平衡内存和计算")
            print(f"   4. 负载均衡确保设备利用率")
            
        else:
            print(f"❌ 无法创建设备网格，请检查GPU设备数量")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
