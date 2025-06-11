#!/bin/bash
# GPT-1.5B JAX 推理性能测试启动脚本 (Ubuntu版本)
# =========================================

echo "🚀 GPT-1.5B JAX 推理性能测试启动脚本"
echo "========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装或未添加到PATH"
    echo "请先安装Miniconda或Anaconda"
    exit 1
fi

echo "📦 激活conda环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt_inference
if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败，请先创建环境:"
    echo "conda env create -f environment.yml"
    exit 1
fi

echo "🔍 检查JAX和GPU环境..."
python -c "import jax; print(f'JAX版本: {jax.__version__}'); devices = jax.devices(); print(f'GPU数量: {len(devices)}'); [print(f'  GPU {i}: {device}') for i, device in enumerate(devices)]"
if [ $? -ne 0 ]; then
    echo "❌ JAX或GPU检查失败"
    exit 1
fi

echo "📊 检查数据集文件..."
if [ ! -f "datasets/benchmark_dataset_config_0.jsonl" ]; then
    echo "❌ 数据集文件不存在"
    exit 1
fi

echo "📁 创建结果目录..."
mkdir -p results

echo "⚡ 开始运行基准测试..."
echo ""
python main.py --max-samples 10 --show-gpu-info

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试完成！"
    echo "📁 查看results目录获取详细结果"
    echo ""
    echo "🔍 快速查看结果:"
    if [ -f "results/benchmark_summary.txt" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cat results/benchmark_summary.txt
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
else
    echo ""
    echo "❌ 测试失败，请检查错误信息"
    echo "💡 常见问题排查:"
    echo "   1. 检查CUDA驱动是否正确安装"
    echo "   2. 检查JAX是否能正确识别GPU"
    echo "   3. 检查数据集文件是否存在"
fi

echo ""
echo "按Enter键继续..."
read
