# 🐳 Docker 開發環境

這個目錄包含所有 Docker 相關的配置文件和腳本，用於建立一致的開發環境。

## 📁 文件結構

```
docker/
├── Dockerfile              # Docker 映像配置
├── docker-compose.yml      # 基本 Docker Compose 配置
├── docker-compose.dev.yml  # 開發環境專用配置
├── .dockerignore           # Docker 建置忽略文件
├── docker-dev.bat          # Windows 開發腳本
├── docker-dev.sh           # Unix 開發腳本
└── README.md               # 本說明文件
```

## 🚀 快速開始

### 方法 1: 使用根目錄的便捷腳本（推薦）

```bash
# Windows
.\dev.bat                   # 啟動開發環境

# macOS/Linux  
bash dev.sh                 # 啟動開發環境
```

### 方法 2: 直接使用 docker 目錄中的腳本

```bash
# Windows
.\docker\docker-dev.bat dev

# macOS/Linux
bash docker/docker-dev.sh dev
```

## 🛠️ 可用命令

| 命令      | 功能         | 說明                                      |
| --------- | ------------ | ----------------------------------------- |
| `dev`     | 啟動開發環境 | 預設命令，啟動 Jekyll 服務器 + LiveReload |
| `build`   | 建置生產版本 | 生產環境建置，輸出到 `_site/`             |
| `test`    | 運行測試     | 執行 HTML 校驗和其他測試                  |
| `shell`   | 進入容器     | 用於除錯和手動操作                        |
| `logs`    | 查看日誌     | 實時查看容器日誌                          |
| `stop`    | 停止服務     | 停止所有正在運行的容器                    |
| `restart` | 重啟服務     | 停止並重新啟動開發環境                    |
| `clean`   | 清理資源     | 清理 Docker 映像、容器和卷                |

## 🔧 配置說明

### Dockerfile
- 基於 `ruby:3.3-alpine` 輕量級映像
- 安裝 Ruby、Node.js 和必要的建置工具
- 預先安裝專案依賴，加快啟動速度

### docker-compose.dev.yml（開發環境）
- 掛載專案目錄，實現即時同步
- 啟用 LiveReload 功能
- 使用命名 volume 優化依賴安裝速度
- 設定開發專用的環境變數

### docker-compose.yml（生產/測試）
- 生產環境建置配置
- 包含測試服務配置
- 使用 profiles 控制服務啟動

## 🌐 網址訪問

當開發環境啟動後，可通過以下網址訪問：

- **主網站**: http://localhost:4000
- **LiveReload**: http://localhost:35729 (自動連接)

## 📋 故障排除

### 端口衝突
如果 4000 或 35729 端口被占用：

```bash
# 檢查端口使用情況
netstat -ano | findstr :4000    # Windows
lsof -i :4000                   # macOS/Linux

# 停止占用端口的進程或修改 docker-compose.yml 中的端口映射
```

### 權限問題
如果遇到文件權限問題：

```bash
# Windows: 確保 Docker Desktop 有足夠權限
# macOS/Linux: 確保 Docker 有訪問專案目錄的權限
sudo chown -R $USER:$USER .
```

### 依賴問題
如果依賴安裝失敗：

```bash
# 清理並重新建置
.\dev.bat clean
.\dev.bat dev
```

## 🔄 更新和維護

### 更新依賴
當 `Gemfile` 或 `package.json` 更新時：

```bash
# 重新建置 Docker 映像
.\dev.bat clean
.\dev.bat dev
```

### 清理空間
定期清理 Docker 資源：

```bash
# 清理未使用的映像和容器
.\dev.bat clean

# 或手動清理
docker system prune -a
docker volume prune
```

## 📊 效能優化

### 建置快取
- Docker 層級快取：相同的依賴不會重複安裝
- 命名 volume：`node_modules` 和 `.bundle` 在容器間持久化

### 文件同步
- 使用 `--force_polling` 確保在 Windows 上的文件變更檢測
- `.dockerignore` 排除不必要的文件，加快建置速度

## 🚀 生產部署

這個 Docker 環境主要用於本地開發。生產環境建議使用：
- GitHub Pages（自動建置）
- 或其他 Jekyll 專用的部署平台

## 📞 支援

如果遇到問題：
1. 檢查 Docker Desktop 是否正常運行
2. 確認網路連接正常（需要下載依賴）
3. 查看容器日誌：`.\dev.bat logs`
4. 進入容器除錯：`.\dev.bat shell`
