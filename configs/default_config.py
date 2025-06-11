#!/usr/bin/env python3
"""
默认配置文件 - GPT-1.5B JAX推理项目
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 1600
    n_layer: int = 48
    n_head: int = 25
    dropout: float = 0.1
    use_bias: bool = True


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    max_samples: int = 10
    batch_size: int = 1
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    num_warmup: int = 3
    num_runs: int = 5
    output_dir: str = "results"
    save_results: bool = True
    show_gpu_info: bool = True
    device_count: int = 4  # 4x RTX 3080


@dataclass
class SystemConfig:
    """系统配置"""
    cuda_visible_devices: Optional[str] = None
    jax_memory_fraction: float = 0.8
    enable_triton_fusion: bool = True
    log_level: str = "INFO"
    random_seed: int = 42


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = ModelConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    system: SystemConfig = SystemConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """从字典创建配置"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            benchmark=BenchmarkConfig(**config_dict.get('benchmark', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model': self.model.__dict__,
            'benchmark': self.benchmark.__dict__,
            'system': self.system.__dict__
        }


# 默认配置实例
DEFAULT_CONFIG = Config()

# 快速测试配置
QUICK_TEST_CONFIG = Config(
    benchmark=BenchmarkConfig(
        max_samples=3,
        num_warmup=1,
        num_runs=2,
        max_tokens=50
    )
)

# 完整基准测试配置
FULL_BENCHMARK_CONFIG = Config(
    benchmark=BenchmarkConfig(
        max_samples=100,
        num_warmup=5,
        num_runs=10,
        max_tokens=200
    )
)
