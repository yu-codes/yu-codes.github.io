#!/bin/bash
# Docker 開發環境快捷腳本 - 調用 docker 目錄中的主腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

bash docker/docker-dev.sh "$@"
