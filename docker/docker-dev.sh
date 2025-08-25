#!/bin/bash

# Jekyll Docker 開發環境管理腳本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

help() {
    echo "Jekyll Docker 開發環境管理工具"
    echo ""
    echo "使用方式: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  dev        啟動開發服務器 (預設)"
    echo "  build      建置網站"
    echo "  test       運行測試"
    echo "  clean      清理 Docker 資源"
    echo "  shell      進入容器 shell"
    echo "  logs       查看日誌"
    echo "  stop       停止所有服務"
    echo "  restart    重啟開發服務器"
    echo "  help       顯示此說明"
    echo ""
    echo "範例:"
    echo "  $0 dev     # 啟動開發服務器"
    echo "  $0 build   # 建置生產版本"
    echo "  $0 clean   # 清理所有 Docker 資源"
}

dev() {
    echo "🚀 啟動 Jekyll 開發服務器..."
    docker-compose -f docker/docker-compose.dev.yml up --build
}

build() {
    echo "🏗️ 建置 Jekyll 網站..."
    docker-compose -f docker/docker-compose.yml --profile build up --build build
}

test() {
    echo "🧪 運行網站測試..."
    docker-compose -f docker/docker-compose.yml --profile build up --build build
    docker-compose -f docker/docker-compose.yml --profile test up test
}

clean() {
    echo "🧹 清理 Docker 資源..."
    docker-compose -f docker/docker-compose.dev.yml down -v --remove-orphans
    docker-compose -f docker/docker-compose.yml down -v --remove-orphans
    docker system prune -f
    echo "✅ 清理完成"
}

shell() {
    echo "🐚 進入容器 shell..."
    docker-compose -f docker/docker-compose.dev.yml exec jekyll-dev sh
}

logs() {
    echo "📋 查看服務日誌..."
    docker-compose -f docker/docker-compose.dev.yml logs -f
}

stop() {
    echo "⏹️ 停止所有服務..."
    docker-compose -f docker/docker-compose.dev.yml down
    docker-compose -f docker/docker-compose.yml down
}

restart() {
    echo "🔄 重啟開發服務器..."
    docker-compose -f docker/docker-compose.dev.yml down
    docker-compose -f docker/docker-compose.dev.yml up --build
}

# 主要邏輯
case "${1:-dev}" in
    dev)
        dev
        ;;
    build)
        build
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    shell)
        shell
        ;;
    logs)
        logs
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo "❌ 未知命令: $1"
        echo ""
        help
        exit 1
        ;;
esac
