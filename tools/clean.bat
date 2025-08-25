@echo off
setlocal

REM 專案清理腳本 (Windows 版本)
REM 用於清理不必要的生成文件和快取

echo 🧹 開始清理專案文件...

REM 清理 Jekyll 相關
echo 清理 Jekyll 建置文件和快取...
if exist "_site" rmdir /s /q "_site"
if exist ".jekyll-cache" rmdir /s /q ".jekyll-cache"
if exist ".jekyll-metadata" del /q ".jekyll-metadata"

REM 清理 Node.js 相關（可選）
set /p "cleanup_node=是否要清理 node_modules? (y/N): "
if /i "%cleanup_node%"=="y" (
    echo 清理 Node.js 依賴...
    if exist "node_modules" rmdir /s /q "node_modules"
    if exist "package-lock.json" del /q "package-lock.json"
)

REM 清理 Ruby Bundle 相關（可選）
set /p "cleanup_bundle=是否要清理 Ruby bundle 快取? (y/N): "
if /i "%cleanup_bundle%"=="y" (
    echo 清理 Ruby Bundle 快取...
    if exist ".bundle" rmdir /s /q ".bundle"
    if exist "Gemfile.lock" del /q "Gemfile.lock"
)

REM 清理 Docker 相關（可選）
set /p "cleanup_docker=是否要清理 Docker 資源? (y/N): "
if /i "%cleanup_docker%"=="y" (
    echo 清理 Docker 資源...
    docker system prune -f
    docker volume prune -f
)

REM 清理臨時文件
echo 清理臨時文件...
for /r %%i in (*.tmp *.temp *.swp *.swo *~ .DS_Store Thumbs.db) do (
    if exist "%%i" del /q "%%i"
)

echo ✅ 清理完成！
echo.
echo 建議接下來執行：
echo   npm install              # 重新安裝 Node.js 依賴
echo   bundle install           # 重新安裝 Ruby 依賴
echo   .\docker-dev.bat dev     # 啟動 Docker 開發環境

pause
