# Ubuntu 系统安装指南

本项目支持在 Ubuntu 系统上运行，以下是详细的安装和配置指南。

## 🖥️ 系统要求

### 硬件要求

- **GPU**: 推荐 4x NVIDIA RTX 3080 或更高性能 GPU
- **内存**: 至少 32GB RAM
- **存储**: 至少 20GB 可用空间
- **CPU**: 多核处理器 (推荐 16 核心以上)

### 软件要求

- **操作系统**: Ubuntu 20.04 LTS / 22.04 LTS
- **Python**: 3.8-3.10
- **CUDA**: 11.8
- **cuDNN**: 8.6+
- **NVIDIA 驱动**: 520+

## 🚀 快速安装

### 步骤 1: 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要的系统依赖
sudo apt install -y build-essential curl wget git vim
sudo apt install -y python3-dev python3-pip
```

### 步骤 2: NVIDIA 驱动和 CUDA 安装

```bash
# 检查GPU
nvidia-smi

# 如果没有NVIDIA驱动，安装推荐驱动
sudo ubuntu-drivers autoinstall

# 安装CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加CUDA到PATH (添加到 ~/.bashrc)
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 3: Conda 安装

```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载shell配置
source ~/.bashrc

# 验证安装
conda --version
```

### 步骤 4: 项目安装

```bash
# 克隆项目
git clone https://github.com/marszbz/gpt15b-jax-inference.git
cd gpt15b-jax-inference

# 运行安装脚本
chmod +x setup_environment.sh
./setup_environment.sh
```

## 🔧 手动安装（如果自动脚本失败）

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate gpt_inference

# 安装JAX (Ubuntu CUDA版本)
pip install jax==0.6.1 jaxlib==0.6.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python -c "import jax; print(f'JAX版本: {jax.__version__}'); print(f'GPU数量: {len(jax.devices())}')"
```

## 🚦 运行测试

### 基本测试

```bash
# 激活环境
conda activate gpt_inference

# 运行快速测试
python main.py --max-samples 5

# 或使用脚本
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```

### 完整基准测试

```bash
# 运行所有配置的基准测试
python main.py --max-samples 100 --show-gpu-info --save-results
```

## 🐛 常见问题排查

### CUDA 问题

```bash
# 检查CUDA安装
nvcc --version
nvidia-smi

# 检查JAX是否识别GPU
python -c "import jax; print(jax.devices())"
```

### 内存问题

```bash
# 监控GPU内存使用
watch -n 1 nvidia-smi

# 如果内存不足，减少batch size
python main.py --max-samples 10
```

### 依赖问题

```bash
# 重新创建环境
conda env remove -n gpt_inference
conda env create -f environment.yml
```

## 📊 性能优化建议

### 系统优化

```bash
# 设置GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 1215,1410  # 根据你的GPU调整

# 设置CPU性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 环境变量优化

```bash
# 添加到 ~/.bashrc
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用所有4个GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## 🔄 更新项目

```bash
# 获取最新代码
git pull origin main

# 更新依赖
conda activate gpt_inference
pip install -r requirements.txt --upgrade
```

## 📈 监控和日志

### 实时监控

```bash
# GPU监控
watch -n 1 nvidia-smi

# 系统资源监控
htop

# 项目日志
tail -f results/benchmark_*.log
```

## 🆘 获取帮助

如果遇到问题，请：

1. 检查 `results/` 目录中的错误日志
2. 运行 `python main.py --help` 查看所有选项
3. 在 GitHub Issues 中报告问题
4. 提供完整的错误信息和系统配置

## 🎯 下一步

项目安装完成后，您可以：

1. 运行基准测试评估性能
2. 修改配置文件调整参数
3. 添加自己的数据集
4. 贡献代码改进项目
