#!/usr/bin/env python3
"""
GPT-1.5B JAX推理性能测试 - 独立运行脚本
解决所有导入和环境问题
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# 设置JAX环境（必须在导入JAX之前）
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# 清理可能存在的XLA_FLAGS
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

# 导入JAX相关包
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import numpy as np
    print(f"✅ JAX {jax.__version__} 加载成功")
except ImportError as e:
    print(f"❌ JAX导入失败: {e}")
    print("请运行：pip install jax==0.6.1 jaxlib==0.6.1+cuda118 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    sys.exit(1)

from typing import Dict, List, Optional, Any
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


class SimpleTokenizer:
    """简化的文本分词器"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """编码文本为token序列"""
        # 简化实现：基于字符的编码
        tokens = []
        for char in text.lower():
            token_id = min(ord(char), self.vocab_size - 1)
            tokens.append(token_id)
        
        # 处理长度限制
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # 填充到指定长度
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """解码token序列为文本"""
        chars = []
        for token_id in tokens:
            if token_id != self.pad_token_id and token_id > 0:
                chars.append(chr(min(token_id, 127)))  # 限制在ASCII范围内
        return ''.join(chars)


class DatasetLoader:
    """数据集加载器 - 处理JSONL格式数据"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """加载所有数据集配置"""
        print("📁 加载数据集...")
        
        if not self.dataset_dir.exists():
            print(f"❌ 数据集目录不存在: {self.dataset_dir}")
            return
        
        # 查找所有配置文件
        config_files = list(self.dataset_dir.glob("benchmark_dataset_config_*.jsonl"))
        print(f"🔍 找到 {len(config_files)} 个配置文件")
        
        for config_file in config_files:
            config_id = config_file.stem.split('_')[-1]
            samples = []
            
            # 检查文件是否为空
            if config_file.stat().st_size == 0:
                print(f"⚠️ 配置 {config_id} 文件为空，跳过")
                continue
            
            # 读取JSONL文件
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError as e:
                                print(f"⚠️ 配置 {config_id} 第 {line_num} 行JSON解析错误: {e}")
                                continue
                
                if samples:
                    self.datasets[config_id] = samples
                    print(f"📊 配置 {config_id}: {len(samples)} 个样本")
                    
                    # 显示示例
                    if samples:
                        sample = samples[0]
                        print(f"   示例: prompt_length={sample.get('prompt_length', 'N/A')}, "
                              f"generation_length={sample.get('generation_length', 'N/A')}")
                else:
                    print(f"⚠️ 配置 {config_id} 没有有效样本")
                    
            except Exception as e:
                print(f"❌ 加载配置 {config_id} 失败: {e}")
    
    def get_valid_datasets(self) -> Dict[str, List[Dict]]:
        """获取所有有效数据集"""
        return {k: v for k, v in self.datasets.items() if v}
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """获取数据集统计信息"""
        stats = {}
        for config_id, samples in self.datasets.items():
            if not samples:
                continue
                
            prompt_lengths = [s.get('prompt_length', 0) for s in samples]
            gen_lengths = [s.get('generation_length', 0) for s in samples]
            
            stats[config_id] = {
                'sample_count': len(samples),
                'avg_prompt_length': np.mean(prompt_lengths) if prompt_lengths else 0,
                'avg_generation_length': np.mean(gen_lengths) if gen_lengths else 0,
                'prompt_length_range': (min(prompt_lengths), max(prompt_lengths)) if prompt_lengths else (0, 0),
                'generation_length_range': (min(gen_lengths), max(gen_lengths)) if gen_lengths else (0, 0),
                'source_types': list(set(s.get('source_type', 'unknown') for s in samples))
            }
        
        return stats
    
    def print_dataset_summary(self):
        """打印数据集摘要"""
        stats = self.get_dataset_stats()
        print("\n📊 数据集摘要:")
        print("-" * 50)
        
        for config_id, stat in stats.items():
            print(f"配置 {config_id}:")
            print(f"  样本数: {stat['sample_count']}")
            print(f"  平均prompt长度: {stat['avg_prompt_length']:.1f}")
            print(f"  平均生成长度: {stat['avg_generation_length']:.1f}")
            print(f"  Prompt长度范围: {stat['prompt_length_range']}")
            print(f"  生成长度范围: {stat['generation_length_range']}")
            print(f"  数据源类型: {', '.join(stat['source_types'])}")
            print()


class InferenceBenchmark:
    """推理性能基准测试"""
    
    def __init__(self, model: GraphPartitionedGPT):
        self.model = model
        self.tokenizer = SimpleTokenizer()
        
    def benchmark_single_sample(self, sample: Dict) -> Dict:
        """测试单个样本的推理性能"""
        prompt = sample['prompt']
        generation_length = sample.get('generation_length', 32)
        sample_id = sample.get('id', 'unknown')
        
        # 编码输入
        prompt_tokens = self.tokenizer.encode(prompt, max_length=512)
        input_ids = jnp.array([prompt_tokens])  # 添加batch维度
        
        # 预热（确保JIT编译完成）
        _ = self.model.generate_text(input_ids, max_new_tokens=8)
        
        # 正式推理计时
        start_time = time.time()
        output_ids = self.model.generate_text(input_ids, max_new_tokens=generation_length)
        
        # 确保计算完成
        jax.block_until_ready(output_ids)
        end_time = time.time()
        
        # 计算性能指标
        inference_time = end_time - start_time
        input_length = len([t for t in prompt_tokens if t != 0])  # 去除padding
        total_tokens = output_ids.shape[1]
        generated_tokens = total_tokens - input_length
        
        throughput = generated_tokens / inference_time if inference_time > 0 else 0
        
        return {
            'sample_id': sample_id,
            'input_length': input_length,
            'generated_tokens': generated_tokens,
            'total_tokens': total_tokens,
            'inference_time': inference_time,
            'throughput_tokens_per_sec': throughput,
            'latency_per_token': inference_time / generated_tokens if generated_tokens > 0 else 0
        }
    
    def benchmark_config(self, dataset: List[Dict], config_id: str, max_samples: int = 10) -> Dict:
        """测试特定配置的性能"""
        if not dataset:
            return {}
        
        print(f"\n🧪 测试配置 {config_id}")
        print(f"   数据集大小: {len(dataset)} 个样本")
        print(f"   测试样本数: {min(max_samples, len(dataset))}")
        
        results = []
        test_samples = min(max_samples, len(dataset))
        
        for i in range(test_samples):
            sample = dataset[i]
            
            try:
                result = self.benchmark_single_sample(sample)
                results.append(result)
                
                # 每3个样本显示一次进度
                if (i + 1) % 3 == 0:
                    recent_results = results[-3:]
                    avg_time = np.mean([r['inference_time'] for r in recent_results])
                    avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in recent_results])
                    print(f"   进度 {i+1}/{test_samples}: 平均延迟 {avg_time:.3f}s, "
                          f"平均吞吐量 {avg_throughput:.1f} tokens/s")
                
            except Exception as e:
                print(f"⚠️ 样本 {i} 测试失败: {e}")
                continue
        
        # 计算统计信息
        if not results:
            return {}
        
        inference_times = [r['inference_time'] for r in results]
        throughputs = [r['throughput_tokens_per_sec'] for r in results]
        latencies = [r['latency_per_token'] for r in results]
        
        summary = {
            'config_id': config_id,
            'samples_tested': len(results),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs),
            'avg_latency_per_token': np.mean(latencies),
            'total_tokens_generated': sum(r['generated_tokens'] for r in results),
            'total_time': sum(inference_times),
            'detailed_results': results
        }
        
        print(f"✅ 配置 {config_id} 测试完成:")
        print(f"   平均推理时间: {summary['avg_inference_time']:.3f}±{summary['std_inference_time']:.3f}s")
        print(f"   平均吞吐量: {summary['avg_throughput']:.1f} tokens/s")
        print(f"   平均每token延迟: {summary['avg_latency_per_token']:.4f}s")
        
        return summary
    
    def run_full_benchmark(self, datasets: Dict[str, List[Dict]], max_samples_per_config: int = 10) -> Dict:
        """运行完整的基准测试"""
        print("\n🚀 开始GPT-1.5B JAX推理性能测试")
        print("=" * 60)
        print(f"📊 数据集配置数: {len(datasets)}")
        print(f"🔧 GPU数量: {len(jax.devices())}")
        print(f"📦 JAX版本: {jax.__version__}")
        print(f"🏗️ 模型规模: {self.model.config.n_layer}层, {self.model.config.n_head}头, {self.model.config.n_embd}维")
        
        all_results = {}
        total_start_time = time.time()
        
        # 逐个测试配置
        for config_id, dataset in datasets.items():
            config_result = self.benchmark_config(dataset, config_id, max_samples_per_config)
            if config_result:
                all_results[config_id] = config_result
        
        total_time = time.time() - total_start_time
        
        # 生成综合报告
        benchmark_summary = {
            'benchmark_info': {
                'total_execution_time': total_time,
                'configs_tested': len(all_results),
                'gpu_count': len(jax.devices()),
                'jax_version': jax.__version__,
                'model_config': {
                    'n_layer': self.model.config.n_layer,
                    'n_head': self.model.config.n_head,
                    'n_embd': self.model.config.n_embd,
                    'vocab_size': self.model.config.vocab_size
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'platform': 'Windows',
                'cuda_version': '11.8'
            },
            'results': all_results
        }
        
        return benchmark_summary


def save_benchmark_results(results: Dict, output_dir: str = 'results') -> tuple:
    """保存基准测试结果"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存详细JSON结果
    json_file = output_path / f"gpt15b_benchmark_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成可读性报告
    report_file = output_path / f"performance_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("GPT-1.5B JAX 推理性能测试报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        info = results['benchmark_info']
        f.write(f"测试时间: {info['timestamp']}\n")
        f.write(f"总执行时间: {info['total_execution_time']:.2f}秒\n")
        f.write(f"GPU数量: {info['gpu_count']}\n")
        f.write(f"JAX版本: {info['jax_version']}\n")
        f.write(f"CUDA版本: {info['cuda_version']}\n")
        f.write(f"平台: {info['platform']}\n\n")
        
        # 模型配置
        model_cfg = info['model_config']
        f.write("模型配置:\n")
        f.write(f"  层数: {model_cfg['n_layer']}\n")
        f.write(f"  注意力头数: {model_cfg['n_head']}\n")
        f.write(f"  嵌入维度: {model_cfg['n_embd']}\n")
        f.write(f"  词汇表大小: {model_cfg['vocab_size']}\n\n")
        
        # 性能结果
        f.write("性能测试结果:\n")
        f.write("-" * 40 + "\n")
        
        all_throughputs = []
        all_latencies = []
        
        for config_id, result in results['results'].items():
            f.write(f"\n配置 {config_id}:\n")
            f.write(f"  测试样本数: {result['samples_tested']}\n")
            f.write(f"  平均推理时间: {result['avg_inference_time']:.3f}±{result['std_inference_time']:.3f}s\n")
            f.write(f"  平均吞吐量: {result['avg_throughput']:.1f} tokens/s\n")
            f.write(f"  吞吐量范围: {result['min_throughput']:.1f} - {result['max_throughput']:.1f} tokens/s\n")
            f.write(f"  平均每token延迟: {result['avg_latency_per_token']:.4f}s\n")
            f.write(f"  总生成token数: {result['total_tokens_generated']}\n")
            
            all_throughputs.append(result['avg_throughput'])
            all_latencies.append(result['avg_latency_per_token'])
        
        # 总体统计
        if all_throughputs:
            f.write(f"\n总体性能统计:\n")
            f.write(f"  平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s\n")
            f.write(f"  最高吞吐量: {max(all_throughputs):.1f} tokens/s\n")
            f.write(f"  平均延迟: {np.mean(all_latencies):.4f}s/token\n")
            f.write(f"  最低延迟: {min(all_latencies):.4f}s/token\n")
    
    print(f"\n💾 结果已保存:")
    print(f"   详细结果: {json_file}")
    print(f"   性能报告: {report_file}")
    
    return json_file, report_file


def check_gpu_setup():
    """检查GPU设置"""
    devices = jax.devices()
    print(f"🔍 GPU环境检查:")
    print(f"   检测到 {len(devices)} 个设备")
    
    for i, device in enumerate(devices):
        print(f"   设备 {i}: {device}")
    
    if len(devices) == 0:
        print("❌ 未检测到GPU设备，请检查CUDA和JAX安装")
        return False
    
    print(f"✅ GPU设置正常，共 {len(devices)} 个设备可用")
    return True


def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        memory_info = []
        for gpu in gpus:
            info = {
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil,
                'load': gpu.load,
                'temperature': gpu.temperature
            }
            memory_info.append(info)
        
        return memory_info
    except ImportError:
        print("⚠️ GPUtil未安装，无法获取详细GPU信息")
        return []


def print_gpu_status():
    """打印GPU状态信息"""
    memory_info = get_gpu_memory_info()
    
    if memory_info:
        print("\n💾 GPU内存状态:")
        print("-" * 50)
        
        for gpu in memory_info:
            print(f"GPU {gpu['id']} ({gpu['name']}):")
            print(f"  内存使用: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_util']*100:.1f}%)")
            print(f"  GPU利用率: {gpu['load']*100:.1f}%")
            print(f"  温度: {gpu['temperature']}°C")
            print()
    else:
        print("ℹ️ 无法获取详细GPU信息")


def print_performance_summary(results: Dict):
    """打印性能摘要"""
    print(f"\n🎉 基准测试完成!")
    print(f"📊 测试配置数: {len(results['results'])}")
    print(f"⏱️ 总执行时间: {results['benchmark_info']['total_execution_time']:.2f}秒")
    
    # 显示性能亮点
    if results['results']:
        all_throughputs = [r['avg_throughput'] for r in results['results'].values()]
        all_latencies = [r['avg_latency_per_token'] for r in results['results'].values()]
        
        print(f"\n🏆 性能亮点:")
        print(f"   最高吞吐量: {max(all_throughputs):.1f} tokens/s")
        print(f"   最低延迟: {min(all_latencies):.4f}s/token")
        print(f"   平均吞吐量: {np.mean(all_throughputs):.1f} tokens/s")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='GPT-1.5B JAX 推理性能测试')
    parser.add_argument('--dataset-dir', default='datasets', help='数据集目录路径')
    parser.add_argument('--output-dir', default='results', help='结果输出目录')
    parser.add_argument('--config', type=str, help='测试特定配置ID (例如: 0,1,3)')
    parser.add_argument('--max-samples', type=int, default=5, help='每个配置的最大测试样本数')
    parser.add_argument('--show-gpu-info', action='store_true', help='显示详细GPU信息')
    
    args = parser.parse_args()
    
    print("🎯 GPT-1.5B JAX 推理性能测试")
    print("=" * 60)
    print(f"💻 平台: Windows")
    print(f"🐍 Python: 3.10")
    print(f"⚡ CUDA: 11.8")
    print(f"📦 JAX: {jax.__version__}")
    
    # 检查GPU环境
    if not check_gpu_setup():
        print("❌ GPU环境检查失败，无法继续")
        return
    
    if args.show_gpu_info:
        print_gpu_status()
    
    # 初始化模型
    print(f"\n🏗️ 初始化GPT-1.5B模型...")
    config = GPTConfig()
    model = GraphPartitionedGPT(config)
    
    # 加载数据集
    print(f"\n📂 加载数据集从: {args.dataset_dir}")
    dataset_loader = DatasetLoader(args.dataset_dir)
    datasets = dataset_loader.get_valid_datasets()
    
    if not datasets:
        print("❌ 未找到有效数据集，请检查数据集文件")
        return
    
    # 显示数据集统计
    dataset_loader.print_dataset_summary()
    
    # 过滤特定配置
    if args.config:
        config_ids = [c.strip() for c in args.config.split(',')]
        filtered_datasets = {k: v for k, v in datasets.items() if k in config_ids}
        if filtered_datasets:
            datasets = filtered_datasets
            print(f"\n🎯 只测试指定配置: {list(datasets.keys())}")
        else:
            print(f"❌ 指定的配置 {config_ids} 不存在")
            return
    
    # 运行基准测试
    benchmark = InferenceBenchmark(model)
    results = benchmark.run_full_benchmark(datasets, args.max_samples)
    
    # 保存结果
    json_file, report_file = save_benchmark_results(results, args.output_dir)
    
    # 显示最终摘要
    print_performance_summary(results)
    
    print(f"\n📁 结果文件:")
    print(f"   {report_file.name}")
    print(f"   {json_file.name}")


if __name__ == "__main__":
    main()
