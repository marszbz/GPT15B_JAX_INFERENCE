# GPT-1.5B JAX 推理性能测试项目

基于 JAX 实现的 GPT-1.5B 模型推理性能测试，支持图分割和多 GPU 并行化，专为 Ubuntu + 4x RTX 3080 GPU 环境优化。

## 🚀 项目特性

- ✅ 完整的 JAX/Flax GPT-1.5B 模型实现
- ✅ 图分割技术支持多 GPU 并行推理
- ✅ 自动加载和处理您的 JSONL 数据集
- ✅ 详细的性能基准测试和报告
- ✅ Ubuntu 系统优化
- ✅ 支持 CUDA 11.8 + JAX 0.6.1

## 📋 环境要求

- Ubuntu 20.04/22.04 LTS
- Python 3.8-3.10
- CUDA 11.8
- 4x NVIDIA RTX 3080 GPU
- JAX 0.6.1
- Conda/Miniconda

## 🛠️ 快速开始

### 1. 克隆或下载项目

```bash
# 如果已经有项目文件，直接进入目录
cd /path/to/gpt15b-jax-inference
```

### 2. 一键环境设置

```bash
# 使用Makefile快速安装
make install

# 或手动运行环境设置脚本
chmod +x setup_environment.sh
./setup_environment.sh
```

这个脚本会自动：

- 创建 conda 环境
- 安装 JAX (CUDA 11.8 版本)
- 安装所有依赖包
- 验证 GPU 设置

### 3. 运行推理测试

```bash
# 使用Makefile运行测试
make test

# 或手动运行基准测试
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```

### 4. 手动运行（可选）

```powershell
# 激活环境
conda activate gpt_inference

# 运行测试
python main.py

# 查看所有选项
python main.py --help
```

## 📊 数据集格式

项目使用您提供的 JSONL 格式数据集：

```json
{
  "id": "sample_0_0",
  "config_id": 0,
  "prompt": "Alberta Legislature Building...",
  "prompt_length": 32,
  "generation_length": 32,
  "source_type": "synthetic",
  "metadata": {
    "created_at": "2025-05-30",
    "tokenizer": "gpt2",
    "prompt_tokens": 32
  }
}
```

支持的配置：

- **配置 0**: 32 tokens prompt → 32 tokens 生成
- **配置 1**: 32 tokens prompt → 64 tokens 生成
- **配置 2**: 128 tokens prompt → 32 tokens 生成
- **配置 3**: 128 tokens prompt → 64 tokens 生成

## 🎯 使用示例

### 基本测试

```bash
python main.py
```

### 测试特定配置

```bash
# 只测试配置0和1
python main.py --config 0,1

# 自定义样本数
python main.py --max-samples 5

# 显示GPU详细信息
python main.py --show-gpu-info
```

### 自定义输出目录

```bash
python main.py --output-dir my_results
```

### 使用 Makefile 命令

```bash
# 显示所有可用命令
make help

# 完整基准测试
make benchmark

# 检查GPU环境
make check-gpu

# 生成性能报告
make report
```

## 📁 项目结构

```text
gpt15b-jax-inference/
├── main.py                    # 主程序入口
├── requirements.txt           # Python依赖
├── environment.yml            # Conda环境配置
├── Makefile                   # Ubuntu命令集合
├── setup_environment.sh       # 环境安装脚本
├── README.md                  # 项目说明
├── UBUNTU_INSTALL.md          # Ubuntu安装指南
├── upload_to_github_ubuntu.sh # GitHub上传脚本
├── src/                       # 源代码
│   ├── models/               # 模型定义
│   │   ├── gpt_model.py      # GPT-1.5B模型和图分割
│   │   └── __init__.py
│   ├── data/                 # 数据处理
│   │   ├── dataset_loader.py # 数据集加载器
│   │   └── __init__.py
│   ├── inference/            # 推理和测试
│   │   ├── benchmark.py      # 性能基准测试
│   │   └── __init__.py
│   └── utils/                # 工具函数
│       ├── gpu_utils.py      # GPU工具
│       ├── results.py        # 结果处理
│       └── __init__.py
├── datasets/                 # 数据集文件
│   ├── benchmark_dataset_config_0.jsonl
│   ├── benchmark_dataset_config_1.jsonl
│   ├── benchmark_dataset_config_2.jsonl
│   └── benchmark_dataset_config_3.jsonl
├── scripts/                  # 运行脚本
│   ├── setup_environment.sh  # 环境设置脚本
│   └── run_benchmark.sh      # 基准测试脚本
├── configs/                  # 配置文件目录
├── results/                  # 测试结果输出
└── tests/                    # 单元测试（待添加）
```

## 📈 性能指标

测试会生成以下性能指标：

- **推理延迟**: 每次推理的总时间
- **吞吐量**: tokens/秒
- **每 token 延迟**: 平均每个 token 的生成时间
- **GPU 利用率**: 多 GPU 并行效率
- **内存使用**: GPU 内存占用情况

## 📊 结果输出

### JSON 详细结果

```json
{
  "benchmark_info": {
    "total_execution_time": 45.67,
    "configs_tested": 4,
    "gpu_count": 4,
    "jax_version": "0.6.1"
  },
  "results": {
    "0": {
      "avg_throughput": 156.3,
      "avg_inference_time": 0.205,
      "samples_tested": 10
    }
  }
}
```

### 文本报告

```
GPT-1.5B JAX 推理性能测试报告
==================================

测试时间: 2025-06-11 13:45:23
总执行时间: 45.67秒
GPU数量: 4
JAX版本: 0.6.1

配置 0:
  平均推理时间: 0.205±0.015s
  平均吞吐量: 156.3 tokens/s

总体性能统计:
  最高吞吐量: 198.5 tokens/s
  最低延迟: 0.003s/token
```

## 🔧 图分割技术

项目实现了先进的图分割策略：

- **层级并行**: 48 个 Transformer 层分布到 4 个 GPU
- **注意力分割**: 25 个注意力头并行计算
- **参数分片**: 大矩阵按维度智能分片
- **流水线并行**: 减少 GPU 闲置时间

## ⚡ 性能优化

- JAX JIT 编译加速
- 内存高效的参数分片
- 优化的 CUDA 内核
- 自适应批次大小

## 🛠️ 故障排除

### 常见问题

1. **JAX 安装失败**

   ```bash
   # 手动安装JAX CUDA版本
   pip install jax==0.6.1 jaxlib==0.6.1+cuda118 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **GPU 未检测到**

   ```bash
   # 检查CUDA安装
   nvidia-smi
   nvcc --version

   # 验证JAX GPU
   python -c "import jax; print(jax.devices())"
   ```

3. **内存不足**
   - 减少 `--max-samples` 参数
   - 检查 GPU 内存使用: `nvidia-smi`

### 诊断命令

```bash
# 检查环境
python -c "import jax, flax, numpy; print('✅ 所有包正常')"

# 检查GPU
python -c "import jax; print(f'GPU数量: {len(jax.devices())}')"

# 检查数据集
python -c "import os; print('数据集文件:', [f for f in os.listdir('datasets') if f.endswith('.jsonl')])"

# 使用Makefile命令
make check-gpu    # 检查GPU环境
make status       # 查看项目状态
make help         # 显示所有可用命令
```

## 📞 支持

如果遇到问题：

1. 检查环境要求是否满足
2. 运行诊断命令
3. 查看错误日志
4. 确认数据集文件完整

## 📄 许可证

本项目仅供学习和研究使用。

---

🎯 **开始使用**: 运行 `scripts\setup_environment.bat` 然后 `scripts\run_benchmark.bat`

🚀 **快速测试**: `python main.py --config 0 --max-samples 3`
