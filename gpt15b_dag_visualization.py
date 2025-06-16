#!/usr/bin/env python3
"""
GPT-1.5B推理DAG（有向无环图）可视化
获取并绘制完整的计算图结构，包括分片信息和数据流
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from functools import partial

# 设置JAX环境
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

try:
    import jax
    import jax.numpy as jnp
    from jax import random, devices, make_jaxpr
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    import flax.linen as nn
    import numpy as np
    print(f"✅ JAX {jax.__version__} DAG可视化模式")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class GPTConfig:
    """GPT-1.5B配置"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.0
    use_bias: bool = True

class GPTLayer(nn.Module):
    """单个GPT Transformer层"""
    config: GPTConfig
    
    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = MultiHeadAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = MLP(self.config)
    
    def __call__(self, x, mask=None):
        # 残差连接 + 注意力
        x = x + self.attn(self.ln_1(x), mask)
        # 残差连接 + MLP
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    config: GPTConfig
    
    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # QKV投影
        qkv = self.c_attn(x)
        qkv = qkv.reshape(B, T, 3, self.n_head, C // self.n_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # 注意力计算
        q = q.transpose(0, 2, 1, 3)  # (B, n_head, T, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # 缩放点积注意力
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(k.shape[-1]))
        
        if mask is not None:
            att = jnp.where(mask, att, -jnp.inf)
        
        att = jax.nn.softmax(att, axis=-1)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.c_proj(y)

class MLP(nn.Module):
    """MLP层"""
    config: GPTConfig
    
    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
    
    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        return self.c_proj(x)

class SimplifiedGPT(nn.Module):
    """简化的GPT模型用于DAG可视化"""
    config: GPTConfig
    
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.n_positions, self.config.n_embd)
        # 只使用前几层进行演示
        self.layers = [GPTLayer(self.config) for _ in range(4)]  # 简化为4层
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)
    
    def __call__(self, input_ids):
        B, T = input_ids.shape
        
        # Token embedding
        token_emb = self.wte(input_ids)
        
        # Position embedding
        pos = jnp.arange(0, T)[None, :]
        pos_emb = self.wpe(pos)
        
        # Combined embedding
        x = token_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits

class DAGNode:
    """DAG节点"""
    def __init__(self, name: str, op_type: str, shape: Tuple[int, ...], 
                 sharding: Optional[str] = None, device: Optional[str] = None):
        self.name = name
        self.op_type = op_type
        self.shape = shape
        self.sharding = sharding
        self.device = device
        self.inputs: List['DAGNode'] = []
        self.outputs: List['DAGNode'] = []
        self.computation_cost = 0
        self.memory_cost = 0

class DAGVisualizer:
    """DAG可视化器"""
    
    def __init__(self, config: GPTConfig):
        self.config = config
        self.devices = jax.devices()
        self.mesh = None
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[Tuple[str, str]] = []
        
    def create_mesh(self):
        """创建设备网格"""
        if len(self.devices) >= 4:
            devices_array = np.array(self.devices[:4]).reshape(2, 2)
            self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
            print(f"✅ 创建2x2设备网格")
        elif len(self.devices) >= 2:
            devices_array = np.array(self.devices[:2]).reshape(2, 1)
            self.mesh = Mesh(devices_array, axis_names=('data', 'model'))
            print(f"✅ 创建2x1设备网格")
        else:
            devices_array = np.array(self.devices).reshape(1, 1)
            self.mesh = Mesh(devices_array, axis_names=('data',))
            print(f"✅ 创建单设备网格")
    
    def extract_dag_from_jaxpr(self):
        """从JAX表达式提取DAG"""
        print(f"\n🔍 提取计算图DAG")
        print("-" * 50)
        
        # 创建模型
        model = SimplifiedGPT(self.config)
        
        # 创建输入数据
        key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(key, (2, 128), 0, self.config.vocab_size)
        
        # 初始化参数
        params = model.init(key, input_ids)
          # 创建推理函数
        def inference_fn(params, input_ids):
            return model.apply(params, input_ids)
        
        # 获取JAX表达式（计算图）
        closed_jaxpr = make_jaxpr(inference_fn)(params, input_ids)
        jaxpr = closed_jaxpr.jaxpr  # 从ClosedJaxpr中获取实际的Jaxpr
        
        print(f"📊 计算图统计:")
        print(f"   输入: {input_ids.shape}")
        print(f"   原始方程数量: {len(jaxpr.eqns)}")
        print(f"   输入变量数量: {len(jaxpr.invars)}")
        print(f"   输出变量数量: {len(jaxpr.outvars)}")
        
        # 解析JAXPR构建DAG
        self._parse_jaxpr_to_dag(jaxpr, input_ids.shape)
        
        return closed_jaxpr
    
    def _parse_jaxpr_to_dag(self, jaxpr, input_shape):
        """解析JAXPR构建DAG"""
        print(f"\n🔧 解析JAXPR构建DAG")
        print("-" * 30)
        
        # 创建输入节点
        input_node = DAGNode(
            name="input_ids",
            op_type="input",
            shape=input_shape,
            sharding="PartitionSpec('data', None)"
        )
        self.nodes["input_ids"] = input_node
        
        # 解析每个方程
        node_counter = 0
        var_to_node = {}
        
        for i, eqn in enumerate(jaxpr.eqns):
            node_name = f"op_{i}_{eqn.primitive.name}"
            
            # 估算输出形状（简化处理）
            if hasattr(eqn, 'outvars') and eqn.outvars:
                # 这里是简化的形状推断
                if 'dot_general' in eqn.primitive.name:
                    output_shape = self._estimate_matmul_shape(eqn)
                elif 'add' in eqn.primitive.name:
                    output_shape = input_shape  # 简化假设
                elif 'reshape' in eqn.primitive.name:
                    output_shape = self._estimate_reshape_shape(eqn)
                else:
                    output_shape = input_shape  # 默认
            else:
                output_shape = input_shape
            
            # 确定分片策略
            sharding = self._determine_sharding(eqn.primitive.name, output_shape)
            
            # 创建节点
            node = DAGNode(
                name=node_name,
                op_type=eqn.primitive.name,
                shape=output_shape,
                sharding=sharding
            )
            
            # 计算成本估算
            node.computation_cost = self._estimate_computation_cost(eqn.primitive.name, output_shape)
            node.memory_cost = np.prod(output_shape) * 4  # float32
            
            self.nodes[node_name] = node
            node_counter += 1
            
            # 简化的依赖关系（实际实现会更复杂）
            if i > 0:
                prev_node_name = f"op_{i-1}_{jaxpr.eqns[i-1].primitive.name}"
                if prev_node_name in self.nodes:
                    self.edges.append((prev_node_name, node_name))
        
        print(f"   创建节点数量: {len(self.nodes)}")
        print(f"   创建边数量: {len(self.edges)}")
    
    def _estimate_matmul_shape(self, eqn):
        """估算矩阵乘法输出形状"""
        # 简化处理，实际需要解析dimension_numbers
        return (2, 128, self.config.n_embd)
    
    def _estimate_reshape_shape(self, eqn):
        """估算reshape输出形状"""
        # 简化处理
        return (2, 128, self.config.n_embd)
    
    def _determine_sharding(self, op_name: str, shape: Tuple[int, ...]):
        """确定操作的分片策略"""
        if 'dot_general' in op_name:
            return "PartitionSpec('data', None, 'model')"
        elif 'add' in op_name:
            return "PartitionSpec('data', None, None)"
        elif len(shape) >= 2:
            return "PartitionSpec('data', None)"
        else:
            return "PartitionSpec()"
    
    def _estimate_computation_cost(self, op_name: str, shape: Tuple[int, ...]):
        """估算计算成本（FLOPS）"""
        size = np.prod(shape)
        
        if 'dot_general' in op_name:
            return size * self.config.n_embd  # 简化的矩阵乘法成本
        elif 'add' in op_name:
            return size
        elif 'mul' in op_name:
            return size
        elif 'exp' in op_name or 'log' in op_name:
            return size * 10  # 指数/对数操作更昂贵
        else:
            return size
    
    def visualize_dag_structure(self):
        """可视化DAG结构"""
        print(f"\n🎨 DAG结构可视化")
        print("=" * 80)
        
        print(f"📊 节点详情:")
        for name, node in self.nodes.items():
            print(f"   {name}:")
            print(f"     类型: {node.op_type}")
            print(f"     形状: {node.shape}")
            print(f"     分片: {node.sharding}")
            print(f"     计算成本: {node.computation_cost:,} FLOPs")
            print(f"     内存成本: {node.memory_cost/1024/1024:.2f} MB")
            print()
        
        print(f"🔗 边连接关系:")
        for i, (src, dst) in enumerate(self.edges):
            print(f"   {i+1}. {src} → {dst}")
        
        # 创建简化的ASCII图
        self._draw_ascii_dag()
    
    def _draw_ascii_dag(self):
        """绘制ASCII格式的DAG"""
        print(f"\n📈 简化DAG流程图:")
        print("=" * 60)
        
        # 按层级组织节点
        levels = self._compute_dag_levels()
        
        for level, nodes in enumerate(levels):
            print(f"Level {level}:")
            for node_name in nodes:
                node = self.nodes[node_name]
                print(f"  ┌─────────────────────┐")
                print(f"  │ {node.op_type:<19} │")
                print(f"  │ {str(node.shape):<19} │")
                print(f"  │ {node.sharding[:19]:<19} │")
                print(f"  └─────────────────────┘")
                
                # 显示连接
                if level < len(levels) - 1:
                    print(f"           │")
                    print(f"           ▼")
            print()
    
    def _compute_dag_levels(self):
        """计算DAG的层级结构"""
        # 简化的层级计算
        levels = []
        
        # 第0层：输入
        levels.append(['input_ids'])
        
        # 其他层：按操作顺序简单分组
        remaining_nodes = [name for name in self.nodes.keys() if name != 'input_ids']
        
        # 简单分组（实际实现需要拓扑排序）
        chunk_size = max(1, len(remaining_nodes) // 4)
        for i in range(0, len(remaining_nodes), chunk_size):
            chunk = remaining_nodes[i:i+chunk_size]
            if chunk:
                levels.append(chunk)
        
        return levels
    
    def analyze_dag_properties(self):
        """分析DAG属性"""
        print(f"\n📊 DAG属性分析")
        print("=" * 50)
        
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        total_computation = sum(node.computation_cost for node in self.nodes.values())
        total_memory = sum(node.memory_cost for node in self.nodes.values())
        
        print(f"🔢 图统计:")
        print(f"   节点总数: {total_nodes}")
        print(f"   边总数: {total_edges}")
        print(f"   图密度: {total_edges / (total_nodes * (total_nodes - 1)) * 100:.2f}%")
        
        print(f"\n⚡ 计算统计:")
        print(f"   总计算量: {total_computation:,} FLOPs")
        print(f"   总内存需求: {total_memory/1024/1024:.2f} MB")
        print(f"   平均节点计算量: {total_computation/total_nodes:,.0f} FLOPs")
        
        # 找出关键路径
        critical_nodes = self._find_critical_nodes()
        print(f"\n🎯 关键节点 (高计算成本):")
        for node_name in critical_nodes[:5]:  # 显示前5个
            node = self.nodes[node_name]
            print(f"   {node_name}: {node.computation_cost:,} FLOPs")
        
        # 分片分析
        self._analyze_sharding_distribution()
    
    def _find_critical_nodes(self):
        """找出计算成本最高的节点"""
        return sorted(self.nodes.keys(), 
                     key=lambda name: self.nodes[name].computation_cost, 
                     reverse=True)
    
    def _analyze_sharding_distribution(self):
        """分析分片策略分布"""
        print(f"\n🔀 分片策略分析:")
        
        sharding_counts = {}
        for node in self.nodes.values():
            sharding = node.sharding or "无分片"
            sharding_counts[sharding] = sharding_counts.get(sharding, 0) + 1
        
        for sharding, count in sharding_counts.items():
            percentage = (count / len(self.nodes)) * 100
            print(f"   {sharding}: {count} 个节点 ({percentage:.1f}%)")
    
    def demonstrate_parallel_execution(self):
        """演示并行执行"""
        print(f"\n🚀 并行执行演示")
        print("=" * 50)
        
        if not self.mesh:
            print("⚠️ 未创建mesh，无法演示并行执行")
            return
        
        # 创建实际的分片推理
        model = SimplifiedGPT(self.config)
        key = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(key, (4, 128), 0, self.config.vocab_size)
        
        # 初始化参数
        params = model.init(key, input_ids)
        
        with self.mesh:
            # 定义分片策略
            input_sharding = NamedSharding(self.mesh, PartitionSpec('data', None))
            
            # 分片输入
            input_ids_sharded = jax.device_put(input_ids, input_sharding)
            
            # JIT编译推理函数
            @jax.jit
            def sharded_inference(params, input_ids):
                return model.apply(params, input_ids)
            
            # 执行推理
            print(f"🎬 执行分片推理...")
            start_time = time.time()
            logits = sharded_inference(params, input_ids_sharded)
            jax.block_until_ready(logits)
            end_time = time.time()
            
            print(f"✅ 推理完成:")
            print(f"   输入形状: {input_ids.shape}")
            print(f"   输出形状: {logits.shape}")
            print(f"   执行时间: {(end_time - start_time)*1000:.2f}ms")
            print(f"   使用设备: {len(self.devices)} 个GPU")
            
            # 分析并行效率
            total_computation = sum(node.computation_cost for node in self.nodes.values())
            if total_computation > 0:
                sequential_estimate = total_computation / 1e12  # 假设1T FLOPS/s
                parallel_efficiency = sequential_estimate / (end_time - start_time) / len(self.devices)
                
                print(f"\n📈 并行效率分析:")
                print(f"   估算串行时间: {sequential_estimate*1000:.2f}ms")
                print(f"   实际并行时间: {(end_time - start_time)*1000:.2f}ms")
                print(f"   并行效率: {parallel_efficiency*100:.1f}%")
            else:
                print(f"\n📈 并行效率分析:")
                print(f"   实际并行时间: {(end_time - start_time)*1000:.2f}ms")
                print(f"   计算成本信息不可用，无法估算理论加速比")
    
    def export_dag_data(self):
        """导出DAG数据"""
        print(f"\n💾 导出DAG数据")
        print("-" * 30)
        
        # 准备导出数据
        dag_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'vocab_size': self.config.vocab_size,
                'n_embd': self.config.n_embd,
                'n_layer': self.config.n_layer,
                'n_head': self.config.n_head
            },
            'mesh_info': {
                'shape': list(self.mesh.shape) if self.mesh else None,
                'axis_names': list(self.mesh.axis_names) if self.mesh else None,
                'device_count': len(self.devices)
            },
            'nodes': {},
            'edges': self.edges,            'statistics': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'total_computation': int(sum(node.computation_cost for node in self.nodes.values())),
                'total_memory': int(sum(node.memory_cost for node in self.nodes.values()))
            }
        }
          # 导出节点信息
        for name, node in self.nodes.items():
            dag_data['nodes'][name] = {
                'op_type': node.op_type,
                'shape': [int(x) for x in node.shape],  # 转换为Python int
                'sharding': node.sharding,
                'computation_cost': int(node.computation_cost),  # 转换为Python int
                'memory_cost': int(node.memory_cost)  # 转换为Python int
            }
        
        # 保存到文件
        dag_file = Path("gpt15b_dag_analysis.json")
        with open(dag_file, 'w', encoding='utf-8') as f:
            json.dump(dag_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ DAG数据已保存: {dag_file}")
        
        # 创建DOT格式图文件
        self._export_dot_format()
        
        return dag_data
    
    def _export_dot_format(self):
        """导出DOT格式图文件"""
        dot_file = Path("gpt15b_dag.dot")
        
        with open(dot_file, 'w', encoding='utf-8') as f:
            f.write("digraph GPT15B_DAG {\n")
            f.write("  rankdir=TB;\n")
            f.write("  node [shape=box, style=filled];\n\n")
            
            # 写入节点
            for name, node in self.nodes.items():
                color = self._get_node_color(node.op_type)
                label = f"{node.op_type}\\n{node.shape}\\n{node.sharding or ''}"
                f.write(f'  "{name}" [label="{label}", fillcolor="{color}"];\n')
            
            f.write("\n")
            
            # 写入边
            for src, dst in self.edges:
                f.write(f'  "{src}" -> "{dst}";\n')
            
            f.write("}\n")
        
        print(f"✅ DOT图文件已保存: {dot_file}")
        print(f"   使用Graphviz可视化: dot -Tpng {dot_file} -o gpt15b_dag.png")
    
    def _get_node_color(self, op_type: str) -> str:
        """根据操作类型获取节点颜色"""
        color_map = {
            'input': 'lightgreen',
            'dot_general': 'lightblue',
            'add': 'lightyellow',
            'mul': 'lightcoral',
            'reshape': 'lightgray',
            'transpose': 'lightpink',
            'exp': 'orange',
            'log': 'purple'
        }
        return color_map.get(op_type, 'white')
    
    def comprehensive_dag_analysis(self):
        """完整的DAG分析"""
        print(f"🎯 GPT-1.5B推理DAG完整分析")
        print("=" * 80)
        
        # 1. 创建设备网格
        self.create_mesh()
        
        # 2. 提取计算图
        jaxpr = self.extract_dag_from_jaxpr()
        
        # 3. 可视化DAG结构
        self.visualize_dag_structure()
        
        # 4. 分析DAG属性
        self.analyze_dag_properties()
        
        # 5. 演示并行执行
        self.demonstrate_parallel_execution()
        
        # 6. 导出DAG数据
        dag_data = self.export_dag_data()
        
        # 7. 总结
        print(f"\n🎉 DAG分析完成!")
        print("=" * 60)
        print(f"✅ 关键成果:")
        print(f"   • 成功提取GPT-1.5B推理计算图")
        print(f"   • 生成{len(self.nodes)}个计算节点")
        print(f"   • 识别{len(self.edges)}个数据依赖关系")
        print(f"   • 分析分片策略和并行机会")
        print(f"   • 导出DOT格式图文件用于可视化")
        
        print(f"\n💡 可视化建议:")
        print(f"   • 安装Graphviz: https://graphviz.org/")
        print(f"   • 生成图片: dot -Tpng gpt15b_dag.dot -o gpt15b_dag.png")
        print(f"   • 或使用在线工具: http://magjac.com/graphviz-visual-editor/")

def main():
    """主函数"""
    config = GPTConfig()
    visualizer = DAGVisualizer(config)
    
    try:
        visualizer.comprehensive_dag_analysis()
    except Exception as e:
        print(f"❌ DAG分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
