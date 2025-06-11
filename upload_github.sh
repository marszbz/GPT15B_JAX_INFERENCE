#!/bin/bash
# GitHub 上传脚本 - 请先设置您的仓库URL

echo "🚀 GPT-1.5B JAX推理项目 - GitHub上传"
echo "=================================="

# 检查是否已设置远程仓库
if git remote get-url origin 2>/dev/null; then
    echo "✅ 远程仓库已设置"
    git remote -v
else
    echo "⚠️  请先设置您的GitHub仓库URL:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/gpt15b-jax-inference.git"
    echo ""
    echo "🔗 替换YOUR_USERNAME为您的GitHub用户名"
    exit 1
fi

echo ""
echo "📤 开始上传到GitHub..."

# 推送到GitHub
if git push -u origin master; then
    echo ""
    echo "🎉 成功上传到GitHub!"
    echo "📝 访问您的仓库: $(git remote get-url origin)"
    echo ""
    echo "✨ 项目亮点:"
    echo "   - GPT-1.5B JAX实现"
    echo "   - 4x RTX 3080多GPU支持"
    echo "   - Ubuntu CUDA优化"
    echo "   - 完整基准测试套件"
    echo "   - 自动化部署脚本"
else
    echo "❌ 上传失败，请检查:"
    echo "   1. GitHub仓库URL是否正确"
    echo "   2. 是否有推送权限"
    echo "   3. 网络连接是否正常"
fi
