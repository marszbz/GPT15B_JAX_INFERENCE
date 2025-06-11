# 模型模块
"""
GPT模型实现和图分割模块

包含：
- GPTConfig: 模型配置类
- GraphPartitionedGPT: 支持多GPU的图分割GPT模型
- 设备网格管理函数
"""

from .gpt_model import GPTConfig, GraphPartitionedGPT, create_device_mesh

__all__ = ['GPTConfig', 'GraphPartitionedGPT', 'create_device_mesh']
