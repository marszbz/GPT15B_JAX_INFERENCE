#!/bin/bash
# GitHub 上传脚本 (Ubuntu版本)
# ========================================

echo "========================================"
echo "    GitHub 上传脚本"
echo "========================================"
echo ""
echo "请确保您已经在GitHub上创建了仓库！"
echo ""
echo "请将下面的 YOUR_GITHUB_URL 替换为您的实际GitHub仓库URL"
echo "例如: https://github.com/marszbz/gpt15b-jax-inference.git"
echo ""
read -p "按Enter键继续..."

echo "步骤1: 添加远程仓库..."
read -p "请输入您的GitHub仓库URL: " github_url

git remote add origin "$github_url"
if [ $? -ne 0 ]; then
    echo "远程仓库可能已存在，尝试更新..."
    git remote set-url origin "$github_url"
fi

echo ""
echo "步骤2: 推送到GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 成功上传到GitHub！"
    echo "🌐 仓库地址: $github_url"
    echo ""
    echo "📋 下一步建议:"
    echo "  1. 在GitHub上添加项目描述"
    echo "  2. 设置topics标签: jax, gpu, gpt, inference, performance"
    echo "  3. 创建Release版本"
    echo "  4. 添加贡献者指南"
else
    echo ""
    echo "❌ 上传失败！"
    echo "💡 可能的解决方案:"
    echo "  1. 检查GitHub仓库URL是否正确"
    echo "  2. 确认您有写入权限"
    echo "  3. 检查网络连接"
    echo "  4. 尝试使用SSH URL而不是HTTPS"
    echo ""
    echo "🔗 SSH配置方法:"
    echo "  ssh-keygen -t rsa -b 4096 -C \"your_email@example.com\""
    echo "  cat ~/.ssh/id_rsa.pub  # 复制到GitHub SSH Keys"
fi

echo ""
echo "📊 项目统计信息:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
find . -name "*.py" -type f | wc -l | xargs echo "Python文件数量:"
find . -name "*.py" -type f -exec wc -l {} + | tail -n 1 | awk '{print "总代码行数: " $1}'
du -sh . | awk '{print "项目大小: " $1}'
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "🎉 GitHub上传脚本执行完成！"
