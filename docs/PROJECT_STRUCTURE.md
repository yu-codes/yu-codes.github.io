# 📁 專案文件結構整理指南

## 🎯 整理後的文件結構

```
yu-codes.github.io/
├── 📄 核心 Jekyll 文件
│   ├── _config.yml              # Jekyll 主配置
│   ├── index.html               # 網站首頁
│   ├── Gemfile                  # Ruby 依賴管理
│   ├── package.json             # Node.js 依賴管理
│   └── jekyll-theme-chirpy.gemspec  # Gem 主題規格
│
├── 📂 內容和佈局
│   ├── _posts/                  # 部落格文章（按日期）
│   ├── _articles/               # 技術文章（按分類）
│   │   ├── algorithm/           # 演算法文章
│   │   ├── backend/             # 後端技術
│   │   ├── cloud/               # 雲端技術
│   │   ├── data-engineering/    # 資料工程
│   │   ├── deep-learning/       # 深度學習
│   │   ├── machine-learning/    # 機器學習
│   │   ├── optimization/        # 優化技術
│   │   └── system-design/       # 系統設計
│   ├── _tabs/                   # 導航頁面
│   ├── _layouts/                # 頁面佈局模板
│   ├── _includes/               # 可重用組件
│   ├── _data/                   # 結構化資料
│   └── assets/                  # 靜態資源
│       ├── css/                 # 編譯後的 CSS
│       ├── js/                  # 編譯後的 JavaScript
│       └── img/                 # 圖片資源
│
├── 📂 源碼和樣式
│   ├── _sass/                   # SCSS 樣式源碼
│   └── _javascript/             # JavaScript 源碼
│
├── 🔧 開發工具和配置
│   ├── .config/                 # 整理後的配置文件
│   │   ├── .markdownlint.json   # Markdown 檢查規則
│   │   └── .stylelintrc.json    # CSS 檢查規則
│   ├── .vscode/                 # VS Code 配置
│   │   ├── settings.json        # 編輯器設定
│   │   ├── tasks.json           # 任務配置
│   │   └── extensions.json      # 推薦擴展
│   ├── .devcontainer/           # 開發容器配置
│   ├── tools/                   # 開發腳本
│   │   ├── run.sh               # 啟動腳本
│   │   └── test.sh              # 測試腳本
│   ├── rollup.config.js         # JavaScript 打包配置
│   ├── purgecss.js              # CSS 優化配置
│   ├── eslint.config.js         # JavaScript 檢查配置
│   └── .editorconfig            # 編輯器統一配置
│
├── 🐳 Docker 開發環境
│   ├── docker/                  # Docker 配置目錄 📁 新整理
│   │   ├── Dockerfile          # Docker 映像配置
│   │   ├── docker-compose.yml  # 基本 Docker Compose
│   │   ├── docker-compose.dev.yml # 開發環境配置
│   │   ├── .dockerignore       # Docker 忽略文件
│   │   ├── docker-dev.bat      # Windows 開發腳本
│   │   ├── docker-dev.sh       # Unix 開發腳本
│   │   └── README.md           # Docker 使用說明
│   ├── dev.bat                 # 快捷啟動腳本 (Windows) 🆕
│   └── dev.sh                  # 快捷啟動腳本 (Unix) 🆕
│
├── 📚 文件和說明
│   ├── README.md                # 專案說明
│   ├── LICENSE                  # MIT 授權
│   └── docs/                    # 專案文件
│       ├── CHANGELOG.md         # 更新日誌
│       ├── CONTRIBUTING.md      # 貢獻指南
│       ├── CODE_OF_CONDUCT.md   # 行為準則
│       └── SECURITY.md          # 安全政策
│
├── 🔍 版本控制和 CI/CD
│   ├── .git/                    # Git 版本控制
│   ├── .github/                 # GitHub 配置
│   │   └── workflows/           # GitHub Actions
│   ├── .gitignore               # Git 忽略文件
│   ├── .gitattributes           # Git 屬性配置
│   └── .gitmodules              # Git 子模組配置
│
├── 🏗️ 建置和快取（被 .gitignore 忽略）
│   ├── _site/                   # Jekyll 建置輸出
│   ├── .jekyll-cache/           # Jekyll 快取
│   ├── .bundle/                 # Ruby Bundle 快取
│   └── node_modules/            # Node.js 依賴
│
└── 🪝 Git Hooks（可選）
    └── .husky/                  # Git hooks 管理
```

## 🧹 已完成的整理動作

### ✅ 配置文件整理
- 建立 `.config/` 目錄統一管理配置文件
- 移動 `.markdownlint.json` → `.config/markdownlint.json`
- 移動 `.stylelintrc.json` → `.config/stylelintrc.json`

### ✅ .gitignore 優化
- 新增更完整的忽略規則
- 包含 OS 生成文件、臨時文件、日誌文件
- 新增 Docker 和環境變數忽略規則

### ✅ Docker 環境配置
- 建立完整的 Docker 開發環境
- 支援 Windows 和 Unix 系統的開發腳本
- 提供一鍵啟動的便捷工具

## 📋 文件分類說明

### 🟢 核心文件（必須保留）
- Jekyll 配置和內容文件
- Ruby 和 Node.js 依賴文件
- 網站內容和佈局模板

### 🟡 配置文件（已整理）
- 程式碼品質檢查配置
- 編輯器和開發工具設定
- 建置和部署配置

### 🔴 生成文件（可清理）
- `_site/` - Jekyll 建置輸出
- `.jekyll-cache/` - Jekyll 快取
- `node_modules/` - Node.js 依賴
- `.bundle/` - Ruby Bundle 快取

### 🔵 開發工具（功能性）
- Docker 相關配置
- VS Code 開發容器
- Git hooks 和 CI/CD

## 🚀 後續建議

### 1. 定期清理
```bash
# 清理 Jekyll 快取和建置文件
rm -rf _site .jekyll-cache

# 清理 Node.js 依賴（重新安裝）
rm -rf node_modules package-lock.json
npm install

# 清理 Ruby Bundle 快取
rm -rf .bundle Gemfile.lock
bundle install
```

### 2. 使用 Docker 開發（推薦）
```bash
# 一鍵啟動開發環境
.\docker-dev.bat dev

# 清理所有 Docker 資源
.\docker-dev.bat clean
```

### 3. 維護配置文件
- 定期檢查和更新 `.config/` 中的配置
- 保持 `.gitignore` 與專案需求同步
- 根據需要調整 Docker 配置

## 📊 文件大小統計

| 類別       | 文件數量 | 主要作用             |
| ---------- | -------- | -------------------- |
| 內容文件   | ~100+    | 部落格文章和網站內容 |
| 佈局模板   | ~20      | Jekyll 主題和佈局    |
| 樣式文件   | ~30      | SCSS 和 CSS 樣式     |
| JavaScript | ~15      | 前端互動功能         |
| 配置文件   | ~15      | 開發工具配置         |
| 文件說明   | ~5       | 專案文件和授權       |

## 🎯 整理效果

- ✅ **提高可讀性**：文件分類清晰，易於導航
- ✅ **簡化開發**：Docker 一鍵啟動開發環境
- ✅ **統一配置**：配置文件集中管理
- ✅ **減少混亂**：隱藏文件有序組織
- ✅ **提升效率**：快速找到需要的文件
