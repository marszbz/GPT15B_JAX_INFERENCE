#!/bin/bash
# GPT-1.5B JAX 推理项目环境设置脚本 (Ubuntu版本)
# ======================================

echo "🛠️ GPT-1.5B JAX 推理项目环境设置脚本"
echo "======================================"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    echo "Ubuntu安装命令:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "📦 创建conda环境..."
conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo "⚠️ 环境创建失败或已存在，尝试更新..."
    conda env update -f environment.yml
fi

echo "🔧 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt_inference

echo "🚀 安装JAX (CUDA 11.8版本)..."
pip install jax==0.6.1 jaxlib==0.6.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "📋 安装其他依赖..."
pip install -r requirements.txt

echo "🔍 验证安装..."
python -c "import jax; print(f'✅ JAX版本: {jax.__version__}'); devices = jax.devices(); print(f'✅ GPU数量: {len(devices)}'); [print(f'   GPU {i}: {device}') for i, device in enumerate(devices)]"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 环境设置完成！"
    echo "📋 下一步:"
    echo "  1. 运行 ./scripts/run_benchmark.sh 开始测试"
    echo "  2. 或运行 python main.py --help 查看选项"
else
    echo ""
    echo "❌ 环境设置失败，请检查错误信息"
fi

echo ""
echo "按Enter键继续..."
read
