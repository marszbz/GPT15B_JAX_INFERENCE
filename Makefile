# Makefile for GPT-1.5B JAX Inference Project (Ubuntu)
# ===================================================

.PHONY: help install setup test benchmark clean upload-github

# 默认目标
help:
	@echo "🚀 GPT-1.5B JAX Inference Project - Ubuntu Commands"
	@echo "=================================================="
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install     - 完整安装项目环境"
	@echo "  make setup       - 设置开发环境"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test        - 运行快速测试"
	@echo "  make benchmark   - 运行完整基准测试"
	@echo ""
	@echo "🔧 Maintenance:"
	@echo "  make clean       - 清理临时文件"
	@echo "  make reset       - 重置环境"
	@echo ""
	@echo "📤 Deployment:"
	@echo "  make upload-github - 上传到GitHub"
	@echo ""
	@echo "💡 Requirements:"
	@echo "  - Ubuntu 20.04/22.04 LTS"
	@echo "  - NVIDIA GPU with CUDA 11.8"
	@echo "  - Conda/Miniconda installed"

# 安装项目环境
install:
	@echo "🚀 开始安装GPT-1.5B JAX推理环境..."
	@chmod +x setup_environment.sh
	@./setup_environment.sh
	@echo "✅ 安装完成！"

# 设置开发环境
setup:
	@echo "🔧 设置开发环境..."
	@chmod +x scripts/setup_environment.sh
	@./scripts/setup_environment.sh
	@echo "✅ 开发环境设置完成！"

# 快速测试
test:
	@echo "🧪 运行快速测试..."
	@chmod +x scripts/run_benchmark.sh
	@./scripts/run_benchmark.sh
	@echo "✅ 测试完成！"

# 完整基准测试
benchmark:
	@echo "📊 运行完整基准测试..."
	@conda run -n gpt_inference python main.py --max-samples 50 --show-gpu-info --save-results
	@echo "✅ 基准测试完成！查看results/目录获取结果"

# 验证GPU环境
check-gpu:
	@echo "🔍 检查GPU环境..."
	@nvidia-smi
	@echo ""
	@conda run -n gpt_inference python -c "import jax; print(f'JAX版本: {jax.__version__}'); devices = jax.devices(); print(f'GPU数量: {len(devices)}'); [print(f'  GPU {i}: {device}') for i, device in enumerate(devices)]"

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.log" -delete
	@rm -rf .jax_cache/
	@rm -rf temp/
	@rm -rf tmp/
	@echo "✅ 清理完成！"

# 重置环境
reset:
	@echo "🔄 重置conda环境..."
	@conda env remove -n gpt_inference -y || true
	@make install
	@echo "✅ 环境重置完成！"

# 上传到GitHub
upload-github:
	@echo "📤 准备上传到GitHub..."
	@chmod +x upload_to_github_ubuntu.sh
	@./upload_to_github_ubuntu.sh

# 显示项目状态
status:
	@echo "📊 项目状态信息"
	@echo "================"
	@echo "Python文件数量: $$(find . -name '*.py' -type f | wc -l)"
	@echo "总代码行数: $$(find . -name '*.py' -type f -exec wc -l {} + | tail -n 1 | awk '{print $$1}')"
	@echo "项目大小: $$(du -sh . | awk '{print $$1}')"
	@echo ""
	@echo "🔍 环境检查:"
	@conda info --envs | grep gpt_inference || echo "  ❌ gpt_inference环境未找到"
	@echo ""
	@echo "📁 数据集文件:"
	@ls -la datasets/ | grep -E "benchmark_dataset_config_[0-9]+\.jsonl" | wc -l | xargs echo "  配置文件数量:"
	@echo ""
	@echo "📋 最近的结果:"
	@ls -la results/ 2>/dev/null | tail -5 || echo "  暂无结果文件"

# 安装系统依赖
install-deps:
	@echo "📦 安装系统依赖..."
	@sudo apt update
	@sudo apt install -y build-essential curl wget git vim
	@sudo apt install -y python3-dev python3-pip
	@echo "✅ 系统依赖安装完成！"

# 检查CUDA安装
check-cuda:
	@echo "🔍 检查CUDA环境..."
	@nvcc --version || echo "❌ CUDA未安装或未添加到PATH"
	@nvidia-smi || echo "❌ NVIDIA驱动未安装"

# 生成性能报告
report:
	@echo "📈 生成性能报告..."
	@mkdir -p reports
	@conda run -n gpt_inference python -c "
import json
import os
from datetime import datetime

# 收集基本信息
info = {
    'timestamp': datetime.now().isoformat(),
    'system': 'Ubuntu',
    'project': 'GPT-1.5B JAX Inference',
    'files': len([f for f in os.listdir('.') if f.endswith('.py')]),
}

# 保存报告
with open('reports/system_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print('📊 报告已生成: reports/system_info.json')
"
	@echo "✅ 性能报告生成完成！"
