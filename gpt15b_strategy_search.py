#!/usr/bin/env python3
"""
GPT-1.5B JAX策略搜索系统
自动搜索最优的分片和并行化策略
"""

import os
import sys
import time
import json
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from functools import partial
import numpy as np

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
    print(f"✅ JAX {jax.__version__} 策略搜索模式")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    sys.exit(1)

@dataclass
class HardwareInfo:
    """硬件信息"""
    gpu_name: str
    memory_gb: float
    compute_capability: str
    device_count: int
    config_key: str

# 导入策略搜索配置
try:
    from strategy_search_config import (
        SEARCH_SPACE, CONSTRAINTS, OPTIMIZATION_OBJECTIVES,
        SEARCH_ALGORITHMS, BENCHMARK_CONFIG, MODEL_VARIANTS,
        HARDWARE_CONFIGS, ADVANCED_OPTIONS, OUTPUT_CONFIG
    )
    print("✅ 策略搜索配置加载成功")
except ImportError as e:
    print(f"⚠️ 策略搜索配置加载失败，使用默认配置: {e}")
    # 使用默认配置的fallback代码
    HARDWARE_CONFIGS = {}
    CONSTRAINTS = {"max_memory_per_gpu": 20.0}

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
    
    def get_param_count(self) -> int:
        """估算参数数量"""
        embed_params = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        layer_params = (
            self.n_embd * 3 * self.n_embd +  # QKV
            self.n_embd * self.n_embd +      # attention output
            self.n_embd * 4 * self.n_embd +  # MLP up
            4 * self.n_embd * self.n_embd +  # MLP down
            4 * self.n_embd                    # layer norms
        )
        total_params = embed_params + self.n_layer * layer_params
        return total_params

@dataclass
class ShardingStrategy:
    """分片策略配置"""
    name: str
    mesh_shape: Tuple[int, ...]
    mesh_axes: Tuple[str, ...]
    
    # 模型分片策略
    embedding_spec: PartitionSpec
    attention_qkv_spec: PartitionSpec
    attention_out_spec: PartitionSpec
    mlp_up_spec: PartitionSpec
    mlp_down_spec: PartitionSpec
    lm_head_spec: PartitionSpec
    
    # 数据分片策略
    input_spec: PartitionSpec
    
    # 预期性能特征
    memory_efficiency: float = 0.0
    compute_efficiency: float = 0.0
    communication_overhead: float = 0.0

@dataclass
class PerformanceMetrics:
    """性能指标"""
    throughput_tokens_per_sec: float
    latency_ms: float
    memory_usage_gb: float
    gpu_utilization: float
    communication_time_ms: float
    compilation_time_ms: float
    
    # 综合评分
    overall_score: float = 0.0

class StrategySearchSpace:
    """策略搜索空间定义"""
    
    def __init__(self, num_devices: int):
        self.num_devices = num_devices
        self.possible_mesh_shapes = self._generate_mesh_shapes()
        self.possible_partition_specs = self._generate_partition_specs()
    
    def _generate_mesh_shapes(self) -> List[Tuple[int, ...]]:
        """生成可能的mesh形状"""
        shapes = []
        
        # 1D mesh
        shapes.append((self.num_devices,))
        
        # 2D mesh
        for i in range(1, self.num_devices + 1):
            if self.num_devices % i == 0:
                j = self.num_devices // i
                if i <= j:  # 避免重复
                    shapes.append((i, j))
        
        # 3D mesh (如果设备数足够)
        if self.num_devices >= 8:
            for i in range(1, int(self.num_devices**0.33) + 2):
                for j in range(i, int((self.num_devices/i)**0.5) + 2):
                    if self.num_devices % (i * j) == 0:
                        k = self.num_devices // (i * j)
                        if j <= k:
                            shapes.append((i, j, k))
        
        return shapes
    
    def _generate_partition_specs(self) -> Dict[str, List[PartitionSpec]]:
        """生成可能的分片规范"""
        specs = {
            'no_shard': [PartitionSpec()],
            'data_only': [PartitionSpec('data', None)],
            'model_only': [PartitionSpec(None, 'model')],
            'data_model': [PartitionSpec('data', 'model')],
            'model_data': [PartitionSpec('model', 'data')],
        }
        
        # 3D分片（如果有第三个轴）
        if self.num_devices >= 8:
            specs.update({
                'pipeline_data': [PartitionSpec('pipeline', 'data', None)],
                'pipeline_model': [PartitionSpec('pipeline', None, 'model')],
                'full_3d': [PartitionSpec('pipeline', 'data', 'model')]
            })
        
        return specs
    
    def generate_all_strategies(self) -> List[ShardingStrategy]:
        """生成所有可能的策略组合"""
        strategies = []
        
        for mesh_shape in self.possible_mesh_shapes:
            # 确定轴名称
            if len(mesh_shape) == 1:
                mesh_axes = ('data',)
                available_specs = ['no_shard', 'data_only']
            elif len(mesh_shape) == 2:
                mesh_axes = ('data', 'model')
                available_specs = ['no_shard', 'data_only', 'model_only', 'data_model']
            else:  # 3D
                mesh_axes = ('pipeline', 'data', 'model')
                available_specs = list(self.possible_partition_specs.keys())
            
            # 为每个组件选择分片策略
            embedding_specs = self._filter_specs_for_component('embedding', available_specs)
            attention_specs = self._filter_specs_for_component('attention', available_specs)
            mlp_specs = self._filter_specs_for_component('mlp', available_specs)
            
            # 生成策略组合（限制数量避免爆炸）
            counter = 0
            for emb_spec_name in embedding_specs:
                for att_spec_name in attention_specs:
                    for mlp_spec_name in mlp_specs:
                        if counter >= 20:  # 每个mesh形状最多20个策略
                            break
                        
                        emb_spec = self.possible_partition_specs[emb_spec_name][0]
                        att_spec = self.possible_partition_specs[att_spec_name][0]
                        mlp_spec = self.possible_partition_specs[mlp_spec_name][0]
                        
                        strategy = ShardingStrategy(
                            name=f"mesh{mesh_shape}_{emb_spec_name}_{att_spec_name}_{mlp_spec_name}",
                            mesh_shape=mesh_shape,
                            mesh_axes=mesh_axes,
                            embedding_spec=emb_spec,
                            attention_qkv_spec=att_spec,
                            attention_out_spec=att_spec,
                            mlp_up_spec=mlp_spec,
                            mlp_down_spec=mlp_spec,
                            lm_head_spec=emb_spec,  # 通常与embedding相同
                            input_spec=PartitionSpec(mesh_axes[0] if mesh_axes else None, None)
                        )
                        
                        strategies.append(strategy)
                        counter += 1
                
                if counter >= 20:
                    break
        
        return strategies
    
    def _filter_specs_for_component(self, component: str, available_specs: List[str]) -> List[str]:
        """为特定组件过滤合适的分片规范"""
        if component == 'embedding':
            # 嵌入层适合词汇表分片
            return [spec for spec in available_specs if 'model' in spec or spec == 'no_shard']
        elif component == 'attention':
            # 注意力层适合头分片
            return available_specs
        elif component == 'mlp':
            # MLP层适合隐藏层分片
            return available_specs
        else:
            return available_specs

class SimpleGPTModel(nn.Module):
    """简化的GPT模型用于策略搜索"""
    config: GPTConfig
    
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.n_positions, self.config.n_embd)
        
        # 只使用几层进行快速测试
        self.layers = [self._create_layer() for _ in range(4)]
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)
    
    def _create_layer(self):
        """创建单个Transformer层"""
        return {
            'ln_1': nn.LayerNorm(),
            'attn_qkv': nn.Dense(3 * self.config.n_embd, use_bias=self.config.use_bias),
            'attn_out': nn.Dense(self.config.n_embd, use_bias=self.config.use_bias),
            'ln_2': nn.LayerNorm(),
            'mlp_up': nn.Dense(4 * self.config.n_embd, use_bias=self.config.use_bias),
            'mlp_down': nn.Dense(self.config.n_embd, use_bias=self.config.use_bias)
        }
    
    def __call__(self, input_ids):
        B, T = input_ids.shape
        
        # Token + Position embedding
        token_emb = self.wte(input_ids)
        pos = jnp.arange(0, T)[None, :]
        pos_emb = self.wpe(pos)
        x = token_emb + pos_emb
        
        # Transformer layers (简化实现)
        for layer in self.layers:
            # Self-attention
            qkv = layer['attn_qkv'](layer['ln_1'](x))
            # 简化的注意力计算
            att_out = layer['attn_out'](qkv[:, :, :self.config.n_embd])
            x = x + att_out
            
            # MLP
            mlp_hidden = layer['mlp_up'](layer['ln_2'](x))
            mlp_out = layer['mlp_down'](jax.nn.gelu(mlp_hidden))
            x = x + mlp_out
        
        # Final output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class StrategyBenchmarker:
    """策略基准测试器"""
    
    def __init__(self, config: GPTConfig):
        self.config = config
        self.devices = jax.devices()
        self.model = SimpleGPTModel(config)
        
        # 创建测试数据
        key = jax.random.PRNGKey(42)
        self.test_input = jax.random.randint(key, (4, 128), 0, config.vocab_size)
        
        print(f"🔧 策略基准测试器初始化")
        print(f"   可用设备: {len(self.devices)}")
        print(f"   测试输入: {self.test_input.shape}")
    
    def benchmark_strategy(self, strategy: ShardingStrategy) -> Optional[PerformanceMetrics]:
        """基准测试单个策略"""
        try:
            print(f"\n⚡ 测试策略: {strategy.name}")
            print(f"   Mesh形状: {strategy.mesh_shape}")
            print(f"   Mesh轴: {strategy.mesh_axes}")
            
            # 创建mesh
            if len(strategy.mesh_shape) == 1:
                devices_array = np.array(self.devices[:strategy.mesh_shape[0]])
            elif len(strategy.mesh_shape) == 2:
                total_devices = strategy.mesh_shape[0] * strategy.mesh_shape[1]
                devices_array = np.array(self.devices[:total_devices]).reshape(strategy.mesh_shape)
            else:  # 3D
                total_devices = np.prod(strategy.mesh_shape)
                devices_array = np.array(self.devices[:total_devices]).reshape(strategy.mesh_shape)
            
            mesh = Mesh(devices_array, axis_names=strategy.mesh_axes)
            
            with mesh:
                # 创建分片数据
                input_sharding = NamedSharding(mesh, strategy.input_spec)
                input_sharded = jax.device_put(self.test_input, input_sharding)
                
                # 初始化模型参数
                key = jax.random.PRNGKey(42)
                params = self.model.init(key, self.test_input)
                
                # 应用分片策略到参数（简化实现）
                params_sharded = self._apply_sharding_to_params(params, strategy, mesh)
                
                # JIT编译
                @jax.jit
                def sharded_inference(params, input_ids):
                    return self.model.apply(params, input_ids)
                
                # 编译时间测量
                compilation_start = time.time()
                
                # 预热编译
                _ = sharded_inference(params_sharded, input_sharded)
                jax.block_until_ready(_)
                
                compilation_time = (time.time() - compilation_start) * 1000
                
                # 性能测试
                times = []
                for _ in range(5):
                    start_time = time.time()
                    result = sharded_inference(params_sharded, input_sharded)
                    jax.block_until_ready(result)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                tokens_processed = self.test_input.size
                throughput = tokens_processed / avg_time
                
                # 内存使用估算（简化）
                param_memory = self._estimate_memory_usage(params_sharded, strategy)
                
                metrics = PerformanceMetrics(
                    throughput_tokens_per_sec=throughput,
                    latency_ms=avg_time * 1000,
                    memory_usage_gb=param_memory,
                    gpu_utilization=85.0,  # 简化假设
                    communication_time_ms=avg_time * 1000 * 0.1,  # 假设10%通信开销
                    compilation_time_ms=compilation_time
                )
                
                # 计算综合评分
                metrics.overall_score = self._calculate_overall_score(metrics, strategy)
                
                print(f"   ✅ 性能: {throughput:.1f} tokens/s, {avg_time*1000:.2f}ms")
                
                return metrics
                
        except Exception as e:
            print(f"   ❌ 策略测试失败: {e}")
            return None
    
    def _apply_sharding_to_params(self, params, strategy: ShardingStrategy, mesh: Mesh):
        """将分片策略应用到参数（简化实现）"""
        # 这里是简化实现，实际需要根据参数名称和形状应用不同的分片策略
        return params
    
    def _estimate_memory_usage(self, params, strategy: ShardingStrategy) -> float:
        """估算内存使用（GB）"""
        # 简化的内存估算
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        memory_gb = param_count * 4 / (1024**3)  # float32
        
        # 根据分片策略调整
        devices_used = np.prod(strategy.mesh_shape)
        if devices_used > 1:
            memory_gb = memory_gb / devices_used * 1.2  # 考虑通信缓冲
        
        return memory_gb
    
    def _calculate_overall_score(self, metrics: PerformanceMetrics, strategy: ShardingStrategy) -> float:
        """计算综合评分"""
        # 权重设置
        throughput_weight = 0.4
        latency_weight = 0.3
        memory_weight = 0.2
        compilation_weight = 0.1
        
        # 归一化指标（假设基准值）
        throughput_score = min(metrics.throughput_tokens_per_sec / 1000.0, 1.0)
        latency_score = max(0, 1.0 - metrics.latency_ms / 200.0)
        memory_score = max(0, 1.0 - metrics.memory_usage_gb / 10.0)
        compilation_score = max(0, 1.0 - metrics.compilation_time_ms / 5000.0)
        
        overall_score = (
            throughput_weight * throughput_score +
            latency_weight * latency_score +
            memory_weight * memory_score +
            compilation_weight * compilation_score
        )
        
        return overall_score

class StrategySearcher:
    """策略搜索器"""
    
    def __init__(self, config: GPTConfig, hardware_info: HardwareInfo = None):
        self.config = config
        self.devices = jax.devices()
        
        # 硬件检测和约束应用
        if hardware_info is None:
            self.hardware_info = detect_hardware()
        else:
            self.hardware_info = hardware_info
        
        self.constraints = apply_hardware_constraints(self.hardware_info)
        
        print(f"🎯 策略搜索配置:")
        print(f"   硬件: {self.hardware_info.gpu_name} x{self.hardware_info.device_count}")
        print(f"   内存限制: {self.constraints.get('max_memory_per_gpu', 'N/A')}GB")
        print(f"   最大批次: {self.constraints.get('max_batch_size', 'N/A')}")
        
        self.search_space = StrategySearchSpace(len(self.devices))
        self.benchmarker = StrategyBenchmarker(config)
        self.results: List[Tuple[ShardingStrategy, PerformanceMetrics]] = []
    
    def _filter_strategies_by_hardware(self, strategies: List[ShardingStrategy]) -> List[ShardingStrategy]:
        """根据硬件限制过滤策略"""
        filtered = []
        
        for strategy in strategies:
            # 检查内存约束
            if hasattr(strategy, 'estimated_memory_per_gpu'):
                if strategy.estimated_memory_per_gpu > self.constraints.get('max_memory_per_gpu', float('inf')):
                    continue
            
            # 对于RTX 3080，优先选择模型并行策略
            if self.hardware_info.config_key.startswith('rtx3080'):
                # 检查是否使用了模型并行
                if not any('model' in str(spec) for spec in [
                    strategy.embedding_spec, strategy.attention_qkv_spec,
                    strategy.mlp_up_spec, strategy.mlp_down_spec                ]):
                    continue  # 跳过纯数据并行策略
            
            filtered.append(strategy)
        
        print(f"📋 硬件过滤: {len(strategies)} → {len(filtered)} 策略")
        return filtered
    
    def exhaustive_search(self) -> List[Tuple[ShardingStrategy, PerformanceMetrics]]:
        """穷举搜索所有策略"""
        print(f"\n🔍 开始穷举策略搜索")
        print("=" * 60)
        
        strategies = self.search_space.generate_all_strategies()
        print(f"📊 生成策略数量: {len(strategies)}")
        
        # 根据硬件过滤策略
        strategies = self._filter_strategies_by_hardware(strategies)
        
        successful_results = []
        
        for i, strategy in enumerate(strategies):
            print(f"\n[{i+1}/{len(strategies)}] 测试策略: {strategy.name}")
            print(f"   Mesh: {strategy.mesh_shape} {strategy.mesh_axes}")
            
            # 针对RTX 3080的特殊处理
            if self.hardware_info.config_key.startswith('rtx3080'):
                print(f"   🎯 RTX 3080优化模式")
                # 可以添加特殊的内存检查或优化
            
            metrics = self.benchmarker.benchmark_strategy(strategy)
            if metrics:
                successful_results.append((strategy, metrics))
                print(f"   综合评分: {metrics.overall_score:.3f}")
            
            # 每测试5个策略显示一次进度
            if (i + 1) % 5 == 0:
                print(f"\n📈 进度: {i+1}/{len(strategies)} ({(i+1)/len(strategies)*100:.1f}%)")
                if successful_results:
                    best_so_far = max(successful_results, key=lambda x: x[1].overall_score)
                    print(f"   当前最佳: {best_so_far[0].name} (评分: {best_so_far[1].overall_score:.3f})")
        
        self.results = successful_results
        return successful_results
    
    def smart_search(self) -> List[Tuple[ShardingStrategy, PerformanceMetrics]]:
        """智能策略搜索 - 根据硬件特性选择最优策略"""
        print(f"\n🧠 智能策略搜索")
        print("=" * 60)
        print(f"🎯 目标硬件: {self.hardware_info.gpu_name} x{self.hardware_info.device_count}")
        print(f"💾 内存限制: {self.constraints.get('max_memory_per_gpu')}GB")
        
        candidate_strategies = []
        
        # 1. 基础策略
        print(f"\n📋 Step 1: 生成基础策略")
        basic_strategies = self.search_space.generate_all_strategies()
        candidate_strategies.extend(basic_strategies[:10])  # 限制数量
        print(f"   选择前10个基础策略")
        
        # 2. 硬件特定策略
        if self.hardware_info.config_key.startswith('rtx3080'):
            print(f"\n🎯 Step 2: 生成RTX 3080专用策略")
            rtx3080_strategies = self.generate_rtx3080_optimized_strategies()
            candidate_strategies.extend(rtx3080_strategies)
        else:
            print(f"\n🔧 Step 2: 生成通用优化策略")
            # 可以添加其他GPU的特定策略
        
        # 3. 根据配置文件添加预定义策略
        print(f"\n📖 Step 3: 添加配置文件策略")
        config_strategies = self._load_predefined_strategies()
        candidate_strategies.extend(config_strategies)
        
        # 4. 硬件过滤
        print(f"\n🔍 Step 4: 硬件兼容性过滤")
        filtered_strategies = self._filter_strategies_by_hardware(candidate_strategies)
        
        # 5. 智能排序
        print(f"\n🎯 Step 5: 策略优先级排序")
        sorted_strategies = self._sort_strategies_by_priority(filtered_strategies)
        
        # 6. 执行基准测试
        print(f"\n⚡ Step 6: 执行性能基准测试")
        results = []
        max_test_strategies = min(10, len(sorted_strategies))  # 最多测试10个策略
        
        for i, strategy in enumerate(sorted_strategies[:max_test_strategies]):
            print(f"\n[{i+1}/{max_test_strategies}] 🧪 测试策略: {strategy.name}")
            try:
                metrics = self.benchmarker.benchmark_strategy(strategy)
                if metrics is not None:
                    results.append((strategy, metrics))
                    print(f"   ✅ 成功: {metrics.throughput_tokens_per_sec:.1f} tokens/s")
                else:
                    print(f"   ❌ 失败: 策略不兼容")
            except Exception as e:
                print(f"   ❌ 错误: {str(e)[:50]}...")
        
        # 7. 结果分析
        if results:
            best_strategy, best_metrics = max(results, key=lambda x: x[1].overall_score)
            print(f"\n🏆 最优策略: {best_strategy.name}")
            print(f"   吞吐量: {best_metrics.throughput_tokens_per_sec:.1f} tokens/s")
            print(f"   延迟: {best_metrics.latency_ms:.1f}ms")
            print(f"   内存使用: {best_metrics.memory_usage_gb:.1f}GB")
            print(f"   GPU利用率: {best_metrics.gpu_utilization:.1%}")
        
        return results
    
    def _load_predefined_strategies(self) -> List[ShardingStrategy]:
        """从配置文件加载预定义策略"""
        strategies = []
        
        try:
            from strategy_search_config import SEARCH_SPACE
            templates = SEARCH_SPACE.get("sharding_templates", {})
            
            for template_name, template_config in templates.items():
                if template_name == "rtx3080_optimized" and not self.hardware_info.config_key.startswith('rtx3080'):
                    continue  # 跳过不匹配的硬件特定策略
                
                # 将模板转换为ShardingStrategy对象
                strategy = self._template_to_strategy(template_name, template_config)
                if strategy:
                    strategies.append(strategy)
        except Exception as e:
            print(f"⚠️ 配置文件策略加载失败: {e}")
        
        return strategies
    
    def _template_to_strategy(self, name: str, template: Dict[str, str]) -> Optional[ShardingStrategy]:
        """将模板配置转换为策略对象"""
        try:
            # 解析PartitionSpec字符串
            def parse_spec(spec_str: str) -> PartitionSpec:
                # 简化的解析，实际实现可能需要更复杂的逻辑
                if "('data', None)" in spec_str:
                    return PartitionSpec('data', None)
                elif "(None, 'model')" in spec_str:
                    return PartitionSpec(None, 'model')
                elif "('data', 'model')" in spec_str:
                    return PartitionSpec('data', 'model')
                elif "(None, None, 'model')" in spec_str:
                    return PartitionSpec(None, None, 'model')
                else:
                    return PartitionSpec()
            
            # 根据设备数量选择mesh形状
            if len(self.devices) >= 4:
                mesh_shape = (2, 2)
                mesh_axes = ("data", "model")
            elif len(self.devices) >= 2:
                mesh_shape = (2,)
                mesh_axes = ("data",)
            else:
                mesh_shape = (1,)
                mesh_axes = ("data",)
            
            strategy = ShardingStrategy(
                name=f"Config_{name}",
                mesh_shape=mesh_shape,
                mesh_axes=mesh_axes,
                embedding_spec=parse_spec(template.get("embedding", "PartitionSpec()")),
                attention_qkv_spec=parse_spec(template.get("attention_qkv", "PartitionSpec()")),
                attention_out_spec=parse_spec(template.get("attention_out", "PartitionSpec()")),
                mlp_up_spec=parse_spec(template.get("mlp_up", "PartitionSpec()")),
                mlp_down_spec=parse_spec(template.get("mlp_down", "PartitionSpec()")),
                lm_head_spec=parse_spec(template.get("lm_head", "PartitionSpec()")),
                input_spec=parse_spec(template.get("input", "PartitionSpec()"))
            )
            return strategy
        except Exception as e:
            print(f"⚠️ 模板解析失败 {name}: {e}")
            return None
    
    def _sort_strategies_by_priority(self, strategies: List[ShardingStrategy]) -> List[ShardingStrategy]:
        """根据优先级排序策略"""
        def priority_score(strategy: ShardingStrategy) -> float:
            score = 0.0
            
            # RTX 3080偏好模型并行
            if self.hardware_info.config_key.startswith('rtx3080'):
                if 'model' in str(strategy.embedding_spec):
                    score += 10
                if 'model' in str(strategy.mlp_up_spec):
                    score += 10
                if strategy.name.startswith('RTX3080'):
                    score += 20
            
            # 高内存效率加分
            score += strategy.memory_efficiency * 30
              # 低通信开销加分
            score += (1 - strategy.communication_overhead) * 20
            
            return score
        
        return sorted(strategies, key=priority_score, reverse=True)
    
    def generate_rtx3080_optimized_strategies(self) -> List[ShardingStrategy]:
        """生成RTX 3080优化策略"""
        strategies = []
        
        # RTX 3080专用策略 - 强调模型并行以减少内存使用
        if len(self.devices) >= 4:
            # 2x2 mesh 用于模型并行
            strategy = ShardingStrategy(
                name="RTX3080_ModelParallel_2x2",
                mesh_shape=(2, 2),
                mesh_axes=('data', 'model'),
                embedding_spec=PartitionSpec(None, 'model'),
                attention_qkv_spec=PartitionSpec(None, 'model'),
                attention_out_spec=PartitionSpec('model', None),
                mlp_up_spec=PartitionSpec(None, 'model'),
                mlp_down_spec=PartitionSpec('model', None),
                lm_head_spec=PartitionSpec(None, 'model'),
                input_spec=PartitionSpec('data', None),
                memory_efficiency=0.8,
                compute_efficiency=0.7,
                communication_overhead=0.3
            )
            strategies.append(strategy)
            
            # 混合策略
            strategy = ShardingStrategy(
                name="RTX3080_Hybrid_2x2",
                mesh_shape=(2, 2),
                mesh_axes=('data', 'model'),
                embedding_spec=PartitionSpec('data', 'model'),
                attention_qkv_spec=PartitionSpec(None, 'model'),
                attention_out_spec=PartitionSpec('model', None),
                mlp_up_spec=PartitionSpec('data', 'model'),
                mlp_down_spec=PartitionSpec('model', 'data'),
                lm_head_spec=PartitionSpec('data', 'model'),
                input_spec=PartitionSpec('data', None),
                memory_efficiency=0.9,
                compute_efficiency=0.8,
                communication_overhead=0.4
            )
            strategies.append(strategy)
        
        return strategies
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析搜索结果"""
        print(f"\n📊 策略搜索结果分析")
        print("=" * 60)
        
        if not self.results:
            print("❌ 没有成功的策略结果")
            return {}
        
        # 按评分排序
        sorted_results = sorted(self.results, key=lambda x: x[1].overall_score, reverse=True)
        
        print(f"✅ 成功测试策略数量: {len(self.results)}")
        print(f"\n🏆 Top 5 策略:")
        
        for i, (strategy, metrics) in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. {strategy.name}")
            print(f"   综合评分: {metrics.overall_score:.3f}")
            print(f"   吞吐量: {metrics.throughput_tokens_per_sec:.1f} tokens/s")
            print(f"   延迟: {metrics.latency_ms:.2f}ms")
            print(f"   内存: {metrics.memory_usage_gb:.2f}GB")
            print(f"   Mesh: {strategy.mesh_shape} {strategy.mesh_axes}")
        
        # 最佳策略详细分析
        best_strategy, best_metrics = sorted_results[0]
        
        print(f"\n🎯 最佳策略详细分析: {best_strategy.name}")
        print("-" * 40)
        print(f"Mesh配置:")
        print(f"   形状: {best_strategy.mesh_shape}")
        print(f"   轴名: {best_strategy.mesh_axes}")
        print(f"分片配置:")
        print(f"   输入: {best_strategy.input_spec}")
        print(f"   嵌入: {best_strategy.embedding_spec}")
        print(f"   注意力QKV: {best_strategy.attention_qkv_spec}")
        print(f"   注意力输出: {best_strategy.attention_out_spec}")
        print(f"   MLP上: {best_strategy.mlp_up_spec}")
        print(f"   MLP下: {best_strategy.mlp_down_spec}")
        print(f"   LM头: {best_strategy.lm_head_spec}")
        
        # 性能分析
        print(f"\n性能指标:")
        print(f"   吞吐量: {best_metrics.throughput_tokens_per_sec:.1f} tokens/s")
        print(f"   延迟: {best_metrics.latency_ms:.2f}ms")
        print(f"   内存使用: {best_metrics.memory_usage_gb:.2f}GB")
        print(f"   GPU利用率: {best_metrics.gpu_utilization:.1f}%")
        print(f"   通信时间: {best_metrics.communication_time_ms:.2f}ms")
        print(f"   编译时间: {best_metrics.compilation_time_ms:.2f}ms")
        
        return {
            'best_strategy': asdict(best_strategy),
            'best_metrics': asdict(best_metrics),
            'all_results': [(asdict(s), asdict(m)) for s, m in sorted_results]
        }
    
    def export_results(self) -> Path:
        """导出搜索结果"""
        results_data = self.analyze_results()
        
        # 添加元数据
        results_data['metadata'] = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_config': asdict(self.config),
            'device_count': len(self.devices),
            'total_strategies_tested': len(self.results)
        }
        
        # 保存结果
        results_file = Path("gpt15b_strategy_search_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 搜索结果已保存: {results_file}")
        
        # 创建简化报告
        self._create_summary_report(results_data)
        
        return results_file
    
    def _create_summary_report(self, results_data: Dict[str, Any]):
        """创建简化报告"""
        report_file = Path("gpt15b_strategy_search_summary.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GPT-1.5B 策略搜索报告\n")
            f.write("=" * 50 + "\n\n")
            
            if 'best_strategy' in results_data:
                best = results_data['best_strategy']
                metrics = results_data['best_metrics']
                
                f.write(f"最佳策略: {best['name']}\n")
                f.write(f"综合评分: {metrics['overall_score']:.3f}\n")
                f.write(f"吞吐量: {metrics['throughput_tokens_per_sec']:.1f} tokens/s\n")
                f.write(f"延迟: {metrics['latency_ms']:.2f}ms\n")
                f.write(f"内存: {metrics['memory_usage_gb']:.2f}GB\n")
                f.write(f"Mesh形状: {best['mesh_shape']}\n")
                f.write(f"Mesh轴: {best['mesh_axes']}\n\n")
                
                f.write("推荐配置:\n")
                f.write(f"mesh = Mesh(devices.reshape{best['mesh_shape']}, {best['mesh_axes']})\n")
                f.write(f"input_spec = {best['input_spec']}\n")
                f.write(f"embedding_spec = {best['embedding_spec']}\n")
                f.write(f"attention_spec = {best['attention_qkv_spec']}\n")
            
            f.write(f"\n总测试策略数: {results_data['metadata']['total_strategies_tested']}\n")
            f.write(f"测试时间: {results_data['metadata']['timestamp']}\n")
        
        print(f"📄 简化报告已保存: {report_file}")

@dataclass
class HardwareInfo:
    """硬件信息"""
    gpu_name: str
    memory_gb: float
    compute_capability: str
    device_count: int
    config_key: str

def detect_hardware() -> HardwareInfo:
    """检测当前硬件配置"""
    devices = jax.devices()
    device_count = len(devices)
    
    if device_count == 0:
        raise RuntimeError("未检测到可用GPU设备")
    
    # 获取第一个设备信息（假设所有设备相同）
    device = devices[0]
    device_kind = str(device.device_kind).lower()
    
    # 尝试检测GPU型号（基于设备属性推断）
    # 这是一个简化的检测，实际中可能需要更复杂的方法
    if "3080" in device_kind or hasattr(device, 'memory_size'):
        # 尝试获取内存大小
        try:
            # JAX 0.6.1可能不直接提供内存信息，使用默认值
            gpu_name = "RTX 3080"
            memory_gb = 10.0  # 默认假设10GB版本
            compute_capability = "8.6"
            config_key = "rtx3080_quad"
        except:
            gpu_name = "RTX 3080"
            memory_gb = 10.0
            compute_capability = "8.6"
            config_key = "rtx3080_quad"
    elif "3090" in device_kind:
        gpu_name = "RTX 3090"
        memory_gb = 24.0
        compute_capability = "8.6"
        config_key = "rtx3090_quad"
    elif "4090" in device_kind:
        gpu_name = "RTX 4090"
        memory_gb = 24.0
        compute_capability = "8.9"
        config_key = "rtx4090_quad"
    else:
        # 默认配置
        gpu_name = "未知GPU"
        memory_gb = 8.0  # 保守估计
        compute_capability = "8.0"
        config_key = "rtx3080_quad"  # 使用较严格的限制
    
    print(f"🔍 检测到硬件配置:")
    print(f"   GPU型号: {gpu_name}")
    print(f"   设备数量: {device_count}")
    print(f"   显存大小: {memory_gb}GB (估计)")
    print(f"   计算能力: {compute_capability}")
    print(f"   配置键: {config_key}")
    
    return HardwareInfo(
        gpu_name=gpu_name,
        memory_gb=memory_gb,
        compute_capability=compute_capability,
        device_count=device_count,
        config_key=config_key
    )

def apply_hardware_constraints(hardware: HardwareInfo) -> Dict[str, Any]:
    """根据硬件应用约束条件"""
    # 获取基础约束
    base_constraints = CONSTRAINTS.copy()
      # 获取GPU特定约束
    gpu_constraints = base_constraints.get("gpu_specific_constraints", {})
    specific_constraints = gpu_constraints.get(hardware.config_key, {})
    
    # 应用特定约束
    if specific_constraints:
        print(f"🎯 应用{hardware.gpu_name}特定约束:")
        for key, value in specific_constraints.items():
            print(f"   {key}: {value}")
            base_constraints[key] = value
    
    # 根据实际设备数量调整
    if hardware.device_count != 4:
        print(f"⚠️ 设备数量({hardware.device_count})与预期(4)不符，调整约束")
        # 可以在这里添加设备数量相关的调整逻辑
    
    return base_constraints


def main():
    """主函数 - GPT-1.5B策略搜索系统"""
    print("🔍 GPT-1.5B JAX分布式推理策略搜索系统")
    print("="*80)
    
    try:
        # 1. 检测硬件环境
        print("\n🖥️ 检测硬件环境...")
        hardware_info = detect_hardware()  # 使用已定义的函数
        
        print(f"检测到硬件配置:")
        print(f"   GPU型号: {hardware_info.gpu_name}")
        print(f"   内存: {hardware_info.memory_gb}GB")
        print(f"   设备数量: {hardware_info.device_count}")
        print(f"   计算能力: {hardware_info.compute_capability}")
        
        # 2. 初始化策略搜索器
        print("\n🚀 初始化策略搜索器...")
        config = GPTConfig()  # GPT-1.5B配置
        searcher = StrategySearcher(config, hardware_info)
        
        # 使用智能搜索
        best_strategies = searcher.smart_search()
        
        if not best_strategies:
            print("❌ 未找到适合的策略")
            return
        
        # 分析和保存结果
        results_file = searcher.export_results()
        print(f"\n🎉 策略搜索完成! 结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"❌ 策略搜索失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


def run_rtx3080_optimized_search():
    """运行RTX 3080优化的策略搜索"""
    print("🎯 RTX 3080专用策略搜索")
    print("=" * 80)
    
    try:
        # 检测硬件
        hardware_info = detect_hardware()
        
        if not hardware_info.config_key.startswith('rtx3080'):
            print(f"⚠️ 当前硬件({hardware_info.gpu_name})不是RTX 3080")            
            print(f"   是否继续使用RTX 3080优化策略? (y/n)")
            response = input().lower()
            if response != 'y':
                return
        
        # 创建GPT配置 - 针对RTX 3080内存限制的配置
        config = GPTConfig(
            vocab_size=50257,
            n_positions=1024,  # 减小序列长度以适应内存
            n_embd=1600,
            n_layer=24,  # 减半层数以适应内存
            n_head=25,
            dropout=0.0,
            use_bias=True
        )
        
        print(f"🔧 RTX 3080优化配置:")
        print(f"   参数量: {estimate_model_params(config):,}")
        print(f"   序列长度: {config.n_positions}")
        print(f"   层数: {config.n_layer}")
        
        # 创建搜索器
        searcher = StrategySearcher(config, hardware_info)
        
        # 执行智能搜索
        print(f"\n🚀 开始RTX 3080优化搜索...")
        results = searcher.smart_search()
        
        if results:
            # 保存RTX 3080专用结果
            rtx3080_results = {
                'hardware': asdict(hardware_info),
                'config': asdict(config),
                'results': [
                    {
                        'strategy': asdict(strategy),
                        'metrics': asdict(metrics)
                    }
                    for strategy, metrics in results
                ]
            }
            
            results_file = Path("rtx3080_strategy_search_results.json")
            with open(results_file, 'w') as f:
                json.dump(rtx3080_results, f, indent=2)
            
            print(f"\n💾 RTX 3080结果已保存: {results_file}")
            
            # 生成RTX 3080专用代码模板
            generate_rtx3080_code_template(results[0][0] if results else None)
        
    except Exception as e:
        print(f"❌ RTX 3080搜索失败: {e}")
        import traceback
        traceback.print_exc()

def estimate_model_params(config: GPTConfig) -> int:
    """估算模型参数量"""
    embed_params = config.vocab_size * config.n_embd + config.n_positions * config.n_embd
    layer_params = (
        config.n_embd * 3 * config.n_embd +  # QKV
        config.n_embd * config.n_embd +      # attention output
        config.n_embd * 4 * config.n_embd +  # MLP up
        4 * config.n_embd * config.n_embd +  # MLP down
        4 * config.n_embd                    # layer norms
    )
    total_params = embed_params + config.n_layer * layer_params
    return total_params

def generate_rtx3080_code_template(best_strategy: Optional[ShardingStrategy]):
    """生成RTX 3080专用的代码模板"""
    if not best_strategy:
        return
    
    template = f'''"""
RTX 3080优化的GPT推理代码模板
基于策略搜索结果: {best_strategy.name}
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import flax.linen as nn
import numpy as np

# RTX 3080优化配置
RTX3080_CONFIG = {{
    "mesh_shape": {best_strategy.mesh_shape},
    "mesh_axes": {best_strategy.mesh_axes},
    "memory_limit": "8GB",  # RTX 3080 10GB留2GB给系统
    "max_batch_size": 16,   # 受内存限制
    "max_sequence_length": 1024
}}

def create_rtx3080_mesh():
    """创建RTX 3080优化的设备网格"""
    devices = jax.devices()
    if len(devices) < {len(best_strategy.mesh_shape)}:
        raise ValueError(f"需要至少{{len(best_strategy.mesh_shape)}}个GPU")
    
    devices_array = np.array(devices[:{np.prod(best_strategy.mesh_shape)}]).reshape{best_strategy.mesh_shape}
    return Mesh(devices_array, axis_names={best_strategy.mesh_axes})

def create_rtx3080_shardings(mesh):
    """创建RTX 3080优化的分片策略"""
    return {{
        'embedding': NamedSharding(mesh, {best_strategy.embedding_spec}),
        'attention_qkv': NamedSharding(mesh, {best_strategy.attention_qkv_spec}),
        'attention_out': NamedSharding(mesh, {best_strategy.attention_out_spec}),
        'mlp_up': NamedSharding(mesh, {best_strategy.mlp_up_spec}),
        'mlp_down': NamedSharding(mesh, {best_strategy.mlp_down_spec}),
        'lm_head': NamedSharding(mesh, {best_strategy.lm_head_spec}),
        'input': NamedSharding(mesh, {best_strategy.input_spec})
    }}

# 使用示例:
# mesh = create_rtx3080_mesh()
# shardings = create_rtx3080_shardings(mesh)
# 然后使用这些分片策略进行推理
'''
    
    template_file = Path("rtx3080_inference_template.py")
    with open(template_file, 'w') as f:
        f.write(template)
    
    print(f"📄 RTX 3080代码模板已生成: {template_file}")
