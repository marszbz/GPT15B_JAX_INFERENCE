#!/bin/bash
# 安全运行GPT基准测试脚本 - Ubuntu版本
# =======================================

echo "🚀 GPT-1.5B JAX 安全启动脚本"
echo "============================"

# 清理可能冲突的环境变量
echo "🔧 清理环境变量..."
unset XLA_FLAGS
echo "✅ 清理了XLA_FLAGS"

# 设置推荐的环境变量
echo "⚙️ 设置JAX环境变量..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "✅ XLA_PYTHON_CLIENT_PREALLOCATE=false"
echo "✅ XLA_PYTHON_CLIENT_MEM_FRACTION=0.8"
echo "✅ CUDA_VISIBLE_DEVICES=0,1,2,3"

# 检测JAX
echo ""
echo "🔍 检测JAX状态..."
python3 -c "
import jax
print(f'JAX版本: {jax.__version__}')
devices = jax.devices()
print(f'检测到设备: {len(devices)}个')
for i, device in enumerate(devices):
    print(f'  设备{i}: {device}')
"

if [ $? -ne 0 ]; then
    echo "❌ JAX检测失败"
    exit 1
fi

echo ""
echo "🚀 启动基准测试..."
echo "==================="

# 运行基准测试
python3 run_benchmark.py --max-samples 3 --show-gpu-info

echo ""
echo "🎉 测试完成！"
