#!/bin/bash

# 專案清理腳本
# 用於清理不必要的生成文件和快取

echo "🧹 開始清理專案文件..."

# 清理 Jekyll 相關
echo "清理 Jekyll 建置文件和快取..."
rm -rf _site
rm -rf .jekyll-cache
rm -rf .jekyll-metadata

# 清理 Node.js 相關（可選）
read -p "是否要清理 node_modules? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理 Node.js 依賴..."
    rm -rf node_modules
    rm -f package-lock.json
fi

# 清理 Ruby Bundle 相關（可選）
read -p "是否要清理 Ruby bundle 快取? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理 Ruby Bundle 快取..."
    rm -rf .bundle
    rm -f Gemfile.lock
fi

# 清理 Docker 相關（可選）
read -p "是否要清理 Docker 資源? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理 Docker 資源..."
    docker system prune -f
    docker volume prune -f
fi

# 清理臨時文件
echo "清理臨時文件..."
find . -name "*.tmp" -delete
find . -name "*.temp" -delete
find . -name "*.swp" -delete
find . -name "*.swo" -delete
find . -name "*~" -delete
find . -name ".DS_Store" -delete

echo "✅ 清理完成！"
echo ""
echo "建議接下來執行："
echo "  npm install          # 重新安裝 Node.js 依賴"
echo "  bundle install       # 重新安裝 Ruby 依賴"
echo "  ./docker-dev.sh dev  # 啟動 Docker 開發環境"
