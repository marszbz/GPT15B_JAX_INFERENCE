# 测试目录

本目录包含 GPT-1.5B JAX 推理项目的测试文件。

## 测试文件

- `test_all.py` - 完整的测试套件，包含所有模块的测试

## 运行测试

### 快速测试

```bash
# 进入项目根目录
cd /path/to/gpt15b-jax-inference

# 运行所有测试
python tests/test_all.py

# 或使用Makefile
make test
```

### 单独测试模块

```bash
# 测试特定模块
python -m unittest tests.test_all.TestEnvironment
python -m unittest tests.test_all.TestDatasetLoader
python -m unittest tests.test_all.TestGPTModel
python -m unittest tests.test_all.TestUtils
```

## 测试覆盖的内容

1. **环境测试**：

   - JAX 安装和版本检查
   - GPU 可用性检查
   - 项目结构完整性

2. **数据集加载测试**：

   - JSONL 文件加载
   - 分词器功能
   - 数据预处理

3. **模型测试**：

   - GPT 模型创建
   - 设备网格配置
   - 模型参数初始化

4. **工具函数测试**：
   - GPU 监控工具
   - 结果保存和处理
   - 性能统计

## 测试要求

- Ubuntu 系统环境
- Python 3.8+
- JAX 和相关依赖已安装
- CUDA 环境（GPU 测试）

## 故障排除

如果测试失败，请检查：

1. 环境是否正确安装 (`make install`)
2. 数据集文件是否存在
3. GPU 驱动和 CUDA 是否正确配置
