"""
GPT-1.5B JAX推理项目源代码包

这个包包含了以下模块：
- models: GPT模型实现和图分割
- data: 数据集加载和预处理
- inference: 推理引擎和基准测试
- utils: 工具函数和辅助功能
"""

__version__ = "1.0.0"
__author__ = "GPT-1.5B JAX Inference Team"
__description__ = "High-performance GPT-1.5B inference with JAX and multi-GPU support"

# 导出主要组件
from .models.gpt_model import GPTConfig, GraphPartitionedGPT
from .data.dataset_loader import DatasetLoader, SimpleTokenizer
from .inference.benchmark import InferenceBenchmark
from .utils.gpu_utils import setup_jax_environment, check_gpu_setup
from .utils.results import ResultManager

__all__ = [
    'GPTConfig',
    'GraphPartitionedGPT', 
    'DatasetLoader',
    'SimpleTokenizer',
    'InferenceBenchmark',
    'setup_jax_environment',
    'check_gpu_setup',
    'ResultManager'
]
