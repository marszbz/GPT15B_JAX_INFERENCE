#!/bin/bash
# GPT-1.5B JAX 项目环境安装脚本 (Ubuntu版本)
# ================================

echo "🚀 GPT-1.5B JAX 项目环境安装脚本"
echo "================================"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装Miniconda或Anaconda"
    echo "Ubuntu安装命令:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    exit 1
fi

echo "📦 创建conda环境..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败"
    exit 1
fi

echo "🔧 激活环境并安装JAX..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt_inference

# 安装JAX CUDA版本 (Ubuntu)
echo "🎯 安装JAX (CUDA 11.8版本)..."
pip install --upgrade pip
pip install jax==0.6.1 jaxlib==0.6.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

if [ $? -ne 0 ]; then
    echo "❌ JAX安装失败"
    exit 1
fi

echo "📚 安装其他依赖..."
pip install -r requirements.txt

echo "🔍 验证安装..."
python -c "import jax; print(f'JAX版本: {jax.__version__}'); devices = jax.devices(); print(f'GPU数量: {len(devices)}'); [print(f'  GPU {i}: {device}') for i, device in enumerate(devices)]"

if [ $? -ne 0 ]; then
    echo "❌ 验证失败"
    exit 1
fi

echo "✅ 安装完成！"
echo "📋 使用方法:"
echo "   1. conda activate gpt_inference"
echo "   2. python main.py"
echo ""
echo "💡 提示: 如果遇到CUDA问题，请确保已安装NVIDIA驱动和CUDA 11.8"
