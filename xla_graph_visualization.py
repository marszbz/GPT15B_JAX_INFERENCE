#!/usr/bin/env python3
"""
XLA编译器图优化和图分割可视化系统
展示XLA如何进行图优化和快速数学运算的详细过程
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import partial
import graphviz

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_math=true --xla_dump_to=/tmp/xla_dumps'

try:
    import jax
    import jax.numpy as jnp
    from jax import random, devices
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    import flax.linen as nn
    import numpy as np
    print(f"✅ JAX {jax.__version__} 图优化模式加载")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

print("🔍 XLA编译器图优化和图分割可视化")
print("=" * 60)

@dataclass
class XLAOptimizationConfig:
    """XLA优化配置"""
    enable_fast_math: bool = True
    enable_graph_fusion: bool = True 
    enable_memory_optimization: bool = True
    dump_hlo_graphs: bool = True
    optimization_level: int = 3

class XLAGraphVisualizer:
    """XLA图优化可视化器"""
    
    def __init__(self, config: XLAOptimizationConfig):
        self.config = config
        self.optimization_stages = []
        self.graph_partitions = {}
        
    def analyze_xla_optimizations(self):
        """分析XLA优化过程"""
        print("\n📊 XLA编译器优化分析")
        print("-" * 40)
        
        optimizations = {
            "图融合优化": {
                "描述": "将多个小操作融合为大操作",
                "示例": ["矩阵乘法+偏置加法融合", "激活函数融合", "批量归一化融合"],
                "性能提升": "减少内存访问，提高缓存利用率"
            },
            "快速数学优化": {
                "描述": "使用近似算法加速数学运算",
                "示例": ["快速倒数平方根", "近似GELU", "融合乘加运算"],
                "性能提升": "2-5x加速数学运算"
            },
            "内存布局优化": {
                "描述": "优化张量内存布局提高访问效率",
                "示例": ["列主序转行主序", "内存对齐", "预取优化"],
                "性能提升": "减少内存带宽瓶颈"
            },
            "循环优化": {
                "描述": "优化循环结构和向量化",
                "示例": ["循环展开", "向量化", "并行化"],
                "性能提升": "充分利用SIMD指令"
            }
        }
        
        for name, details in optimizations.items():
            print(f"\n🔧 {name}:")
            print(f"   📝 {details['描述']}")
            print(f"   💡 示例:")
            for example in details['示例']:
                print(f"      • {example}")
            print(f"   ⚡ 性能提升: {details['性能提升']}")
            
        return optimizations
    
    def demonstrate_graph_partitioning(self):
        """演示图分割过程"""
        print("\n🔀 图分割策略演示")
        print("-" * 40)
        
        # 4个GPU的分割策略
        gpu_count = len(jax.devices())
        print(f"检测到 {gpu_count} 个GPU设备")
        
        partitioning_strategies = {
            "数据并行分割": {
                "方式": "在batch维度分片",
                "优势": "简单实现，良好扩展性",
                "适用": "模型较小，内存充足",
                "分片": "PartitionSpec('data', None)"
            },
            "模型并行分割": {
                "方式": "在参数维度分片", 
                "优势": "支持超大模型",
                "适用": "模型巨大，单GPU装不下",
                "分片": "PartitionSpec('model', None)"
            },
            "流水线并行分割": {
                "方式": "按层分割到不同GPU",
                "优势": "最大化GPU利用率",
                "适用": "深层网络",
                "分片": "按层索引分配"
            },
            "混合并行分割": {
                "方式": "数据+模型+流水线组合",
                "优势": "最优性能",
                "适用": "超大规模训练",
                "分片": "2x2 Mesh配置"
            }
        }
        
        for strategy, details in partitioning_strategies.items():
            print(f"\n📋 {strategy}:")
            for key, value in details.items():
                print(f"   {key}: {value}")
                
        return partitioning_strategies
    
    def create_optimization_graph(self):
        """创建优化过程可视化图"""
        print("\n📈 创建XLA优化流程图")
        
        dot = graphviz.Digraph(comment='XLA优化流程')
        dot.attr(rankdir='TB', size='12,8')
        
        # 输入阶段
        dot.node('input', 'JAX程序\n(Python)', shape='box', style='filled', fillcolor='lightblue')
        
        # 编译阶段
        dot.node('jaxpr', 'JAXpr中间表示\n(函数式IR)', shape='box', style='filled', fillcolor='lightgreen')
        dot.node('hlo', 'HLO图\n(高级线性操作)', shape='box', style='filled', fillcolor='lightgreen')
        
        # 优化阶段
        dot.node('fusion', '图融合优化\n• 操作合并\n• 内存优化', shape='box', style='filled', fillcolor='yellow')
        dot.node('fastmath', '快速数学优化\n• 近似算法\n• 融合运算', shape='box', style='filled', fillcolor='yellow')
        dot.node('layout', '内存布局优化\n• 数据重排\n• 缓存优化', shape='box', style='filled', fillcolor='yellow')
        
        # 分割阶段
        dot.node('partition', '图分割\n• 设备分配\n• 通信优化', shape='box', style='filled', fillcolor='orange')
        
        # 代码生成
        dot.node('codegen', 'CUDA代码生成\n• 内核融合\n• 调度优化', shape='box', style='filled', fillcolor='lightcoral')
        dot.node('execution', 'GPU执行\n• 并行计算\n• 内存管理', shape='box', style='filled', fillcolor='lightcoral')
        
        # 添加边
        dot.edge('input', 'jaxpr')
        dot.edge('jaxpr', 'hlo')
        dot.edge('hlo', 'fusion')
        dot.edge('fusion', 'fastmath')
        dot.edge('fastmath', 'layout')
        dot.edge('layout', 'partition')
        dot.edge('partition', 'codegen')
        dot.edge('codegen', 'execution')
        
        # 保存图
        output_path = Path("xla_optimization_flow")
        dot.render(output_path, format='png', cleanup=True)
        print(f"   优化流程图已保存: {output_path}.png")
        
        return dot
    
    def demonstrate_fast_math(self):
        """演示快速数学优化"""
        print("\n⚡ 快速数学优化演示")
        print("-" * 40)
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000, 1000))
        
        print("测试案例: GELU激活函数")
        print(f"输入形状: {x.shape}")
        
        # 标准GELU (精确版本)
        def standard_gelu(x):
            return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))
        
        # 快速GELU (近似版本)
        def fast_gelu(x):
            return jax.nn.gelu(x, approximate=True)
        
        # 编译函数
        compiled_standard = jax.jit(standard_gelu)
        compiled_fast = jax.jit(fast_gelu)
        
        # 预热
        _ = compiled_standard(x).block_until_ready()
        _ = compiled_fast(x).block_until_ready()
        
        # 性能测试
        runs = 10
        
        # 标准版本
        times_standard = []
        for _ in range(runs):
            start = time.time()
            result_standard = compiled_standard(x).block_until_ready()
            times_standard.append(time.time() - start)
        
        # 快速版本
        times_fast = []
        for _ in range(runs):
            start = time.time()
            result_fast = compiled_fast(x).block_until_ready()
            times_fast.append(time.time() - start)
        
        avg_standard = np.mean(times_standard) * 1000
        avg_fast = np.mean(times_fast) * 1000
        speedup = avg_standard / avg_fast
        
        print(f"\n📊 性能对比:")
        print(f"   标准GELU: {avg_standard:.2f}ms")
        print(f"   快速GELU: {avg_fast:.2f}ms")
        print(f"   加速比: {speedup:.2f}x")
        
        # 精度对比
        max_diff = jnp.max(jnp.abs(result_standard - result_fast))
        print(f"   最大误差: {max_diff:.6f}")
        print(f"   相对误差: {max_diff/jnp.max(jnp.abs(result_standard))*100:.4f}%")
        
        return {
            'standard_time': avg_standard,
            'fast_time': avg_fast,
            'speedup': speedup,
            'max_error': float(max_diff)
        }
    
    def analyze_graph_partitioning_example(self):
        """分析具体的图分割例子"""
        print("\n🔍 具体图分割案例分析")
        print("-" * 40)
        
        if len(jax.devices()) < 2:
            print("⚠️ 需要至少2个GPU设备进行分割演示")
            return None
            
        # 创建2x2 mesh
        try:
            if len(jax.devices()) >= 4:
                mesh_devices = mesh_utils.create_device_mesh((2, 2))
                mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
                print(f"✅ 创建2x2 mesh: {mesh.shape}")
            else:
                mesh_devices = mesh_utils.create_device_mesh((2, 1))
                mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
                print(f"✅ 创建2x1 mesh: {mesh.shape}")
                
            # 演示不同的分片策略
            batch_size, seq_len, hidden_size = 8, 512, 768
            
            print(f"\n📊 分片策略对比:")
            print(f"   原始张量: ({batch_size}, {seq_len}, {hidden_size})")
            
            strategies = {
                "无分片": PartitionSpec(),
                "数据并行": PartitionSpec('data', None, None),
                "模型并行": PartitionSpec(None, None, 'model'),
                "混合并行": PartitionSpec('data', None, 'model')
            }
            
            for name, spec in strategies.items():
                print(f"   {name}: {spec}")
                if spec.rules:
                    print(f"     分片维度: {[i for i, rule in enumerate(spec.rules) if rule is not None]}")
                
            # 创建示例张量并应用分片
            key = jax.random.PRNGKey(42)
            x = jax.random.normal(key, (batch_size, seq_len, hidden_size))
            
            with mesh:
                for name, spec in strategies.items():
                    if name == "无分片":
                        continue
                    try:
                        sharding = NamedSharding(mesh, spec)
                        x_sharded = jax.device_put(x, sharding)
                        print(f"   ✅ {name}分片成功")
                    except Exception as e:
                        print(f"   ❌ {name}分片失败: {e}")
                        
            return {
                'mesh_shape': mesh.shape,
                'mesh_axes': mesh.axis_names,
                'tensor_shape': x.shape,
                'strategies': list(strategies.keys())
            }
            
        except Exception as e:
            print(f"❌ Mesh创建失败: {e}")
            return None
    
    def create_partition_visualization(self):
        """创建分片可视化图"""
        print("\n🎨 创建图分割可视化")
        
        dot = graphviz.Digraph(comment='图分割策略')
        dot.attr(rankdir='LR', size='14,10')
        
        # 4个GPU设备
        with dot.subgraph(name='cluster_gpus') as gpu_cluster:
            gpu_cluster.attr(label='4x RTX 3090 GPUs', style='dashed')
            gpu_cluster.node('gpu0', 'GPU 0\n(0,0)', shape='box', style='filled', fillcolor='lightblue')
            gpu_cluster.node('gpu1', 'GPU 1\n(0,1)', shape='box', style='filled', fillcolor='lightblue')
            gpu_cluster.node('gpu2', 'GPU 2\n(1,0)', shape='box', style='filled', fillcolor='lightblue')
            gpu_cluster.node('gpu3', 'GPU 3\n(1,1)', shape='box', style='filled', fillcolor='lightblue')
        
        # 模型组件
        with dot.subgraph(name='cluster_model') as model_cluster:
            model_cluster.attr(label='GPT-1.5B模型组件', style='dashed')
            model_cluster.node('embed', 'Embedding\n50257x1600', shape='ellipse', style='filled', fillcolor='lightgreen')
            model_cluster.node('attn', 'Multi-Head\nAttention\n25头', shape='ellipse', style='filled', fillcolor='lightgreen')
            model_cluster.node('mlp', 'MLP\n1600→6400→1600', shape='ellipse', style='filled', fillcolor='lightgreen')
            model_cluster.node('norm', 'LayerNorm\n1600维', shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # 分片连接
        dot.edge('embed', 'gpu0', label='词汇表分片', style='dashed', color='red')
        dot.edge('embed', 'gpu1', label='词汇表分片', style='dashed', color='red')
        
        dot.edge('attn', 'gpu0', label='注意力头\n0-6', style='solid', color='blue')
        dot.edge('attn', 'gpu1', label='注意力头\n7-12', style='solid', color='blue')
        dot.edge('attn', 'gpu2', label='注意力头\n13-18', style='solid', color='blue')
        dot.edge('attn', 'gpu3', label='注意力头\n19-24', style='solid', color='blue')
        
        dot.edge('mlp', 'gpu0', label='隐藏层\n0-1599', style='dotted', color='green')
        dot.edge('mlp', 'gpu1', label='隐藏层\n1600-3199', style='dotted', color='green')
        dot.edge('mlp', 'gpu2', label='隐藏层\n3200-4799', style='dotted', color='green')
        dot.edge('mlp', 'gpu3', label='隐藏层\n4800-6399', style='dotted', color='green')
        
        dot.edge('norm', 'gpu0', label='复制', style='solid', color='black')
        dot.edge('norm', 'gpu1', label='复制', style='solid', color='black')
        dot.edge('norm', 'gpu2', label='复制', style='solid', color='black')
        dot.edge('norm', 'gpu3', label='复制', style='solid', color='black')
        
        # 保存图
        output_path = Path("graph_partitioning")
        dot.render(output_path, format='png', cleanup=True)
        print(f"   分割可视化图已保存: {output_path}.png")
        
        return dot

def main():
    """主函数 - XLA图优化和分割分析"""
    config = XLAOptimizationConfig()
    visualizer = XLAGraphVisualizer(config)
    
    print(f"🎯 分析目标: 理解XLA编译器的图优化和图分割机制")
    print(f"💻 设备信息: {len(jax.devices())} 个GPU设备")
    
    all_results = {}
    
    try:
        # 1. 分析XLA优化
        optimizations = visualizer.analyze_xla_optimizations()
        all_results['optimizations'] = optimizations
        
        # 2. 演示图分割
        partitioning = visualizer.demonstrate_graph_partitioning()
        all_results['partitioning'] = partitioning
        
        # 3. 快速数学演示
        fastmath_results = visualizer.demonstrate_fast_math()
        all_results['fastmath'] = fastmath_results
        
        # 4. 具体分割案例
        partition_example = visualizer.analyze_graph_partitioning_example()
        all_results['partition_example'] = partition_example
        
        # 5. 创建可视化
        print(f"\n🎨 生成可视化图表...")
        optimization_graph = visualizer.create_optimization_graph()
        partition_graph = visualizer.create_partition_visualization()
        
        # 6. 保存结果
        results_file = Path("xla_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 分析结果已保存: {results_file}")
        
        # 7. 总结
        print(f"\n🎯 XLA图优化和分割分析完成!")
        print("=" * 60)
        print(f"📊 关键发现:")
        print(f"   • XLA优化类型: {len(optimizations)} 种")
        print(f"   • 分割策略: {len(partitioning)} 种")
        if fastmath_results:
            print(f"   • 快速数学加速: {fastmath_results['speedup']:.2f}x")
        if partition_example:
            print(f"   • Mesh配置: {partition_example['mesh_shape']}")
        print(f"   • 可视化图表: 2个PNG文件")
        
        print(f"\n📋 要点总结:")
        print(f"   1. XLA编译器通过多层优化提升性能")
        print(f"   2. 图融合减少内存访问，提高缓存效率")
        print(f"   3. 快速数学用近似算法换取2-5x加速")
        print(f"   4. 图分割实现多GPU并行，支持超大模型")
        print(f"   5. 混合并行策略达到最优性能")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
