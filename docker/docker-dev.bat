@echo off
setlocal

REM Jekyll Docker 開發環境管理腳本 (Windows 版本)

if "%1"=="" goto dev
if "%1"=="dev" goto dev
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="shell" goto shell
if "%1"=="logs" goto logs
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help

echo ❌ 未知命令: %1
echo.
goto help

:dev
echo 🚀 啟動 Jekyll 開發服務器...
docker-compose -f docker\docker-compose.dev.yml up --build
goto end

:build
echo 🏗️ 建置 Jekyll 網站...
docker-compose -f docker\docker-compose.yml --profile build up --build build
goto end

:test
echo 🧪 運行網站測試...
docker-compose -f docker\docker-compose.yml --profile build up --build build
docker-compose -f docker\docker-compose.yml --profile test up test
goto end

:clean
echo 🧹 清理 Docker 資源...
docker-compose -f docker\docker-compose.dev.yml down -v --remove-orphans
docker-compose -f docker\docker-compose.yml down -v --remove-orphans
docker system prune -f
echo ✅ 清理完成
goto end

:shell
echo 🐚 進入容器 shell...
docker-compose -f docker\docker-compose.dev.yml exec jekyll-dev sh
goto end

:logs
echo 📋 查看服務日誌...
docker-compose -f docker\docker-compose.dev.yml logs -f
goto end

:stop
echo ⏹️ 停止所有服務...
docker-compose -f docker\docker-compose.dev.yml down
docker-compose -f docker\docker-compose.yml down
goto end

:restart
echo 🔄 重啟開發服務器...
docker-compose -f docker\docker-compose.dev.yml down
docker-compose -f docker\docker-compose.dev.yml up --build
goto end

:help
echo Jekyll Docker 開發環境管理工具
echo.
echo 使用方式: %0 [命令]
echo.
echo 命令:
echo   dev        啟動開發服務器 (預設)
echo   build      建置網站
echo   test       運行測試
echo   clean      清理 Docker 資源
echo   shell      進入容器 shell
echo   logs       查看日誌
echo   stop       停止所有服務
echo   restart    重啟開發服務器
echo   help       顯示此說明
echo.
echo 範例:
echo   %0 dev     # 啟動開發服務器
echo   %0 build   # 建置生產版本
echo   %0 clean   # 清理所有 Docker 資源
goto end

:end
