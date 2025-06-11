# 工具模块
"""
工具函数和辅助功能模块

包含：
- gpu_utils: GPU相关工具函数
- results: 结果处理和保存工具
"""

from .gpu_utils import setup_jax_environment, check_gpu_setup, print_gpu_status
from .results import ResultManager, save_benchmark_results, print_performance_summary

__all__ = [
    'setup_jax_environment',
    'check_gpu_setup', 
    'print_gpu_status',
    'ResultManager',
    'save_benchmark_results',
    'print_performance_summary'
]
