@echo off
REM Docker 開發環境快捷腳本 - 調用 docker 目錄中的主腳本

cd /d "%~dp0\.."
call docker\docker-dev.bat %*
