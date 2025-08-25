# Yu Codes Blog

這是一個個人技術部落格專案，專注於分享軟體開發知識、程式設計最佳實務、技術教學文章，並記錄個人成長與學習歷程。採用靜態網站生成器架構，確保快速載入與良好的SEO表現。

部落格內容涵蓋：
- **資料工程**：ETL/ELT 流程、資料格式、分散式運算
- **機器學習**：演算法實作、數學基礎、深度學習
- **後端開發**：API 設計、資料庫優化、系統架構
- **前端技術**：現代框架、使用者體驗設計
- **DevOops**：CI/CD、容器化、雲端部署
- **演算法與資料結構**：面試準備、解題思路



## 專案結構

```
yu-codes.github.io/
├── 📄 核心配置文件
│   ├── _config.yml          # 網站主要配置
│   ├── index.html           # 網站入口頁面
│   ├── Gemfile             # Ruby 依賴管理
│   ├── package.json        # Node.js 依賴管理
│ # 1. 啟動 Docker 開發環境
.\dev.bat               # Windows
# 或
bash dev.sh             # macOS/Linux─ jekyll-theme-chirpy.gemspec  # Gem 主題規格
│
├── 📂 內容管理 (分類目錄組織)
│   ├── _posts/             # 部落格文章 (按分類組織)
│   │   ├── data-engineering/    # 資料工程文章 (13篇)
│   │   │   ├── data-format-storage.md
│   │   │   ├── etl-vs-elt-pipeline.md
│   │   │   ├── distributed-compute-engine.md
│   │   │   └── ...
│   │   ├── machine-learning/    # 機器學習文章 (20+篇)
│   │   │   ├── core-concepts.md
│   │   │   ├── classification-algorithms.md
│   │   │   ├── linear-algebra-for-ai.md
│   │   │   └── ...
│   │   └── ...
|   |   
│   ├── _tabs/              # 導航頁面
│   │   ├── archives.md     # 文章歸檔
│   │   ├── categories.md   # 分類頁面
│   │   ├── tags.md         # 標籤頁面
│   │   └── resume.md       # 個人履歷
|   |
│   └── assets/             # 靜態資源
│       ├── img/            # 圖片資源
│       ├── css/            # 編譯後的 CSS
│       └── js/             # 編譯後的 JavaScript
│
├── 📂 網站資料配置
│   └── _data/              # 結構化資料文件
│       ├── authors.yml     # 作者資訊
│       ├── contact.yml     # 聯絡方式與社群連結
│       ├── share.yml       # 社群分享設定
│       ├── media.yml       # 媒體資源設定
│       ├── locales/        # 多語言支援配置
│       └── origin/         # 網站來源配置
│
├── 📂 前端開發
│   ├── _sass/              # SCSS 樣式源碼
│   │   ├── addon/          # 額外功能樣式
│   │   ├── colors/         # 主題色彩配置
│   │   ├── layout/         # 版面佈局樣式
│   │   └── ...
│   ├── _javascript/        # JavaScript 功能模組
│   │   ├── commons.js      # 共用功能 (搜尋、導航)
│   │   ├── home.js         # 首頁互動功能
│   │   ├── post.js         # 文章頁面功能
│   │   ├── theme.js        # 主題切換功能
│   │   ├── modules/        # 可重用模組
│   │   └── pwa/           # Progressive Web App 功能
│   ├── rollup.config.js    # JavaScript 打包配置
│   ├── purgecss.js         # CSS 優化配置
│   └── eslint.config.js    # 代碼品質檢查
│
├── 📂 開發與部署工具
│   ├── 🐳 Docker 開發環境
│   │   ├── docker/             # Docker 配置目錄 📁 新整理
│   │   │   ├── Dockerfile      # Docker 映像配置
│   │   │   ├── docker-compose.yml     # 基本 Docker Compose
│   │   │   ├── docker-compose.dev.yml # 開發環境配置
│   │   │   ├── .dockerignore   # Docker 忽略文件
│   │   │   ├── docker-dev.bat  # Windows 開發腳本
│   │   │   ├── docker-dev.sh   # Unix 開發腳本
│   │   │   └── README.md       # Docker 使用說明
│   │   ├── dev.bat             # 快捷啟動腳本 (Windows) 🆕
│   │   └── dev.sh              # 快捷啟動腳本 (Unix) 🆕
│   ├── .config/            # 開發工具配置 📁 新增
│   │   ├── .markdownlint.json  # Markdown 檢查規則
│   │   └── .stylelintrc.json   # CSS 檢查規則
│   ├── .vscode/            # VS Code 配置
│   │   ├── settings.json   # 編輯器設定
│   │   ├── tasks.json      # 任務配置
│   │   └── extensions.json # 推薦擴展
│   ├── tools/              # 開發輔助腳本
│   │   ├── run.sh          # 本地開發伺服器啟動
│   │   ├── test.sh         # 網站建置與測試
│   │   ├── clean.sh        # 專案清理腳本 🆕
│   │   └── clean.bat       # Windows 清理腳本 🆕
│   ├── docs/               # 專案文件與說明
│   │   └── PROJECT_STRUCTURE.md  # 專案結構說明 🆕
│   └── _plugins/           # 功能擴充插件
│       └── posts-lastmod-hook.rb
│
└── 📄 專案管理文件
    ├── README.md           # 專案說明文件
    ├── LICENSE             # MIT 開源授權
    └── .gitignore          # Git 忽略文件 (已優化) ✨
```

### 內容組織說明

#### 分類目錄管理 (_posts/)
專案採用 `_posts/` 目錄下的子目錄結構來組織文章，提供清晰的分類管理：

**📊 資料工程 (`_posts/data-engineering/`)**
- 資料格式與儲存、ETL/ELT 流程、分散式運算
- 資料品質治理、即時流處理、OLAP 系統
- 檔案範例：`data-format-storage.md`、`distributed-compute-engine.md`

**🤖 機器學習 (`_posts/machine-learning/`)**
- 演算法實作、數學基礎、特徵工程
- LLM 應用、RAG 架構、模型評估
- 檔案範例：`core-concepts.md`、`llm-rag.md`、`linear-algebra-for-ai.md`

**🧠 深度學習 (`_posts/deep-learning/`)**
- CNN、RNN、Transformer 架構
- 注意力機制、生成模型、多模態
- 檔案範例：`transformer-family.md`、`attention-mechanism.md`

**⚡ 優化技術 (`_posts/optimization/`)**
- 梯度下降、學習率調度、正規化
- 分散式訓練、數值穩定性、訓練技巧
- 檔案範例：`gradient-descent.md`、`distributed-training.md`

**🏗️ 系統設計 (`_posts/system-design/`)**
- 高併發架構、特徵存儲、模型服務
- CI/CD for ML、容器化、監控告警
- 檔案範例：`system-design-mindset.md`、`feature-store-design.md`

**☁️ 雲端技術 (`_posts/cloud/`)**
- AWS/Azure/GCP AI 生態系
- Kubernetes、自動擴展、成本優化
- 檔案範例：`aws-ai-ecosystem.md`、`kubernetes-management.md`

#### 檔案命名規則
- **✅ 新規則**: 直接使用描述性檔名，無需時間前綴
  - 範例：`transformer-family.md`、`data-format-storage.md`
- **❌ 舊規則**: ~~`YYYY-MM-DD-title.md`~~ (已移除強制要求)
- **📅 日期管理**: 透過 Front Matter 的 `date` 欄位控制

#### URL 結構
```
舊格式: /posts/title/
新格式: /posts/category/title/
範例：
- /posts/data-engineering/etl-vs-elt-pipeline/
- /posts/machine-learning/transformer-family/
- /posts/system-design/feature-store-design/
```
---
## Jekyll 靜態網站生成器架構

> **重要說明**: 本專案目前使用 Jekyll 作為靜態網站生成器。以下內容介紹 Jekyll 相關的檔案與目錄結構，方便日後遷移到其他生成器時進行替換。

### Jekyll 特定文件與目錄

```
Jekyll 依賴的核心文件:
├── _config.yml              # Jekyll 主配置文件
├── Gemfile                  # Ruby 依賴管理
├── jekyll-theme-chirpy.gemspec  # Gem 規格文件
├── _layouts/                # HTML 模板系統
├── _includes/               # 可重用 HTML 組件
├── _plugins/                # Ruby 插件擴展
└── _sass/                   # SCSS 樣式預處理
```

#### Jekyll 核心概念

1. **Liquid 模板語言**: 用於動態內容渲染
   - 變數輸出: `{{ site.title }}`
   - 邏輯控制: `{% if page.title %}...{% endif %}`
   - 過濾器: `{{ content | strip_html }}`

2. **Front Matter**: YAML 格式的元資料區塊
   ```yaml
   ---
   title: "文章標題"
   date: 2025-01-01
   categories: [技術]
   tags: [程式設計]
   ---
   ```

3. **Collection 集合**: 組織相關內容
   - `_posts`: 部落格文章集合
   - `_tabs`: 導航頁面集合

4. **Layout 繼承**: 模板層次結構
   ```
   default.html (基礎框架)
   ├── home.html (首頁佈局)
   ├── post.html (文章佈局)
   └── page.html (一般頁面)
   ```

#### 遷移考量

如果未來要遷移到其他靜態網站生成器 (如 Hugo、Next.js、Gatsby)，需要替換/轉換：

**必須替換的 Jekyll 特定內容:**
- `_config.yml` → 對應的配置文件格式
- `_layouts/` → 新框架的模板系統
- `_includes/` → 組件化方案
- Liquid 語法 → 新的模板語言
- `_plugins/` → 新框架的插件系統

**可保留的通用內容:**
- `_posts/` 中的 Markdown 文章 (需調整 Front Matter)
- `_data/` 中的 YAML 配置文件
- `assets/` 中的靜態資源
- JavaScript 功能模組 (需適配新的建置流程)

### Jekyll 建置流程

```bash
# 開發環境
bundle exec jekyll serve    # 啟動開發伺服器 + 即時重載
bundle exec jekyll build    # 建置靜態網站到 _site/

# 生產環境
JEKYLL_ENV=production bundle exec jekyll build
```

Jekyll 處理流程：
1. 讀取 `_config.yml` 配置
2. 處理 `_posts/`、`_pages/` 等集合
3. 編譯 Sass/SCSS 檔案
4. 渲染 Liquid 模板
5. 生成靜態 HTML 到 `_site/` 目錄

---

## 環境設置與開發指南

### 🐳 Docker 開發環境 (推薦)

> **推薦使用 Docker**：無需在系統上安裝 Ruby 環境，確保開發環境一致性

#### 系統需求
- Docker Desktop (Windows/macOS/Linux)
- Docker Compose (通常隨 Docker Desktop 一起安裝)

#### 快速開始
```bash
# 1. 啟動開發服務器
docker-dev.bat         # Windows
# 或
bash docker-dev.sh     # macOS/Linux

# 2. 訪問網站
# http://localhost:4000
```

#### Docker 命令參考
```bash
# 開發模式 (預設)
.\dev.bat dev

# 建置生產版本
.\dev.bat build

# 運行測試
.\dev.bat test

# 進入容器 shell (除錯用)
.\dev.bat shell

# 查看日誌
.\dev.bat logs

# 停止服務
.\dev.bat stop

# 清理 Docker 資源
.\dev.bat clean
```

#### Docker 特色
- ✅ 無需安裝 Ruby 環境
- ✅ 一鍵啟動開發服務器
- ✅ 即時重載 (LiveReload)
- ✅ 自動安裝依賴
- ✅ 環境隔離，不影響系統
- ✅ 跨平台一致性

---

### 📦 傳統本地環境設置

> **可選方式**：如果不想使用 Docker，可以直接在系統上安裝依賴

#### 系統需求

| 工具     | 版本需求 | 用途                |
| -------- | -------- | ------------------- |
| Ruby     | 3.0+     | Jekyll 執行環境     |
| RubyGems | 最新版   | Ruby 套件管理       |
| GCC      | 4.2+     | 編譯 native gems    |
| Make     | 最新版   | 建置工具            |
| Node.js  | 16+      | 前端工具鏈          |
| npm      | 8+       | JavaScript 套件管理 |

#### 本地開發環境設置

#### 1. Ruby 環境安裝

**macOS** (使用 Homebrew):
```bash
brew install ruby
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
```

**Windows** (使用 RubyInstaller):
```bash
# 下載並安裝 Ruby+Devkit from https://rubyinstaller.org/
# 安裝完成後執行 ridk install
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install ruby-full build-essential zlib1g-dev
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
```

#### 2. 專案依賴安裝

```bash
# 安裝 Bundler
gem install bundler

# 安裝專案 Ruby 依賴
bundle install

# 安裝 Node.js 依賴
npm install
```

#### 3. 本地開發伺服器

```bash
# 方法 1: 使用專案腳本 (推薦)
bash tools/run.sh
# 支援選項:
# -H, --host [HOST]     指定主機 (預設: 127.0.0.1)
# -p, --production      生產模式

# 方法 2: 直接使用 Jekyll
bundle exec jekyll serve
# 常用選項:
# --livereload          啟用即時重載
# --drafts             包含草稿文章
# --port 4001          自訂埠號

# 方法 3: 生產環境模式
JEKYLL_ENV=production bundle exec jekyll serve
```

#### 4. 前端資源建置

```bash
# 建置所有前端資源
npm run build

# 開發模式 - 監控 JavaScript 變更
npm run watch:js

# 單獨建置 CSS (包含 PurgeCSS 優化)
npm run build:css

# 程式碼品質檢查
npm run lint:scss          # SCSS 語法檢查
npm run lint:fix:scss      # 自動修復 SCSS 問題
npm test                   # 執行所有測試
```

### 開發工作流程

#### 日常開發
1. 啟動開發伺服器: `bash tools/run.sh`
2. 開啟瀏覽器訪問: `http://localhost:4000`
3. 編輯文章或代碼，觀察即時變更
4. 提交前執行測試: `npm test`

#### 新增文章流程
1. 選擇對應的分類目錄: `_posts/machine-learning/`、`_posts/data-engineering/` 等
2. 建立文件: 使用描述性檔名，無需時間前綴
3. 添加 Front Matter:
   ```yaml
   ---
   title: "文章標題"
   date: 2025-08-18 10:00:00 +0800
   categories: [分類名稱]
   tags: [標籤1, 標籤2, 標籤3]
   ---
   ```
4. 使用 Markdown 撰寫內容
5. 本地預覽確認格式正確: `bash tools/run.sh`
6. 提交到版本控制

#### 自訂樣式與功能
1. CSS 修改: 編輯 `_sass/` 目錄下的 SCSS 文件
2. JavaScript 功能: 修改 `_javascript/` 目錄下的模組
3. 頁面佈局: 調整 `_layouts/` 和 `_includes/` 的 HTML 模板
4. 網站配置: 修改 `_config.yml` 和 `_data/` 目錄的設定

### 🧹 專案維護

#### 清理命令
```bash
# 使用清理腳本
.\tools\clean.bat           # Windows
bash tools/clean.sh         # macOS/Linux

# 手動清理
rm -rf _site .jekyll-cache node_modules .bundle
```

#### 重新安裝依賴
```bash
# 重新安裝所有依賴
npm install
bundle install

# 或使用 Docker（推薦）
.\dev.bat clean
.\dev.bat dev
```

### 部署與測試

#### 本地測試
```bash
# 完整建置與測試
bash tools/test.sh

# 手動測試步驟
JEKYLL_ENV=production bundle exec jekyll build
bundle exec htmlproofer _site --disable-external
```

#### GitHub Pages 自動部署
專案已配置 GitHub Actions，推送到 `main` 分支時自動觸發:
1. 建置 Jekyll 網站
2. 執行 HTML 驗證
3. 部署到 GitHub Pages

查看部署狀態: [Actions 頁面](https://github.com/yu-codes/yu-codes.github.io/actions)

## 技術棧

### 核心技術
- **靜態網站生成**: Jekyll 4.x (Ruby 生態系)
- **模板引擎**: Liquid 模板語言
- **內容格式**: Markdown + YAML Front Matter
- **樣式預處理**: Sass/SCSS
- **模組化設計**: 可重用組件與佈局系統

### 前端工具鏈
- **UI 框架**: Bootstrap 5.3.3
- **JavaScript**:
  - 模組打包: Rollup.js
  - 語法轉譯: Babel (ES6+ → ES5)
  - 程式碼檢查: ESLint
- **CSS 優化**:
  - 樣式檢查: Stylelint
  - 未使用樣式清理: PurgeCSS
  - 自動前綴: Autoprefixer

### 開發與部署
- **版本控制**: Git + GitHub
- **CI/CD**: GitHub Actions
- **託管服務**: GitHub Pages
- **CDN**: jsDelivr
- **測試工具**: HTML Proofer
- **效能優化**: 
  - 圖片壓縮與最佳化
  - CSS/JS 壓縮與合併
  - Progressive Web App (PWA) 支援

### 專案特色
- **響應式設計**: 支援桌面、平板、手機
- **主題系統**: 深色/淺色模式切換
- **搜尋功能**: 全站即時搜尋
- **SEO 優化**: 結構化資料、meta 標籤
- **社群整合**: 分享按鈕、留言系統
- **無障礙設計**: 語意化 HTML、鍵盤導航

## 快速開始

### 🐳 Docker 方式 (推薦)

```bash
# 1. 複製專案
git clone https://github.com/yu-codes/yu-codes.github.io.git
cd yu-codes.github.io

# 2. 啟動 Docker 開發環境
.\dev.bat               # Windows
# 或
bash dev.sh             # macOS/Linux

# 3. 開啟瀏覽器
# 訪問 http://localhost:4000
# LiveReload: http://localhost:35729
```

### 📦 傳統方式

```bash
# 1. 複製專案
git clone https://github.com/yu-codes/yu-codes.github.io.git
cd yu-codes.github.io

# 2. 安裝依賴
bundle install    # Ruby 依賴
npm install      # Node.js 依賴

# 3. 啟動開發伺服器
bash tools/run.sh
# 或者: bundle exec jekyll serve

# 4. 開啟瀏覽器
# 訪問 http://localhost:4000
```

### 內容創作

#### 新增文章
```bash
# 1. 選擇對應的分類目錄
cd _posts/machine-learning/    # 或其他分類目錄

# 2. 建立新文章（無需時間前綴）
vim transformer-advanced-techniques.md

# 3. 新增 Front Matter
---
title: "Transformer 進階技術：Multi-Head Attention 深度解析"
date: 2025-08-18 10:00:00 +0800
categories: [Machine Learning]
tags: [Transformer, Attention, Deep Learning]
---

# 4. 即時預覽
bash tools/run.sh --livereload

# 5. 建置與測試
npm run build
bash tools/test.sh
```

#### 文章範本
各分類目錄的標準 Front Matter：

**機器學習文章**:
```yaml
---
title: "文章標題"
date: 2025-08-18 10:00:00 +0800
categories: [Machine Learning]
tags: [演算法, 特徵工程, 模型評估]
image: /assets/img/posts/ml-feature.jpg  # 可選
---
```

**資料工程文章**:
```yaml
---
title: "文章標題"
date: 2025-08-18 10:00:00 +0800
categories: [Data Engineering]
tags: [ETL, 資料格式, 分散式系統]
---
```

**系統設計文章**:
```yaml
---
title: "文章標題"
date: 2025-08-18 10:00:00 +0800
categories: [System Design]
tags: [高併發, 微服務, 架構設計]
---
```

## 貢獻指南

歡迎任何形式的貢獻！如果你有建議、發現錯誤或想要新增功能，請：

1. Fork 這個專案
2. 建立你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的變更 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟一個 Pull Request

### 文章貢獻

如果你想分享技術文章：
1. 確保內容原創且有價值
2. 遵循現有的文章格式與風格
3. 包含適當的程式碼範例
4. 新增必要的標籤與分類

## 授權條款

這個專案使用 [MIT License][license] 授權。

## 聯絡方式

- **部落格**: [yu-codes.github.io][demo]
- **GitHub**: [@yu-codes](https://github.com/yu-codes)
- **Email**: dylan.jhou1120@gmail.com
- **LinkedIn**: [YuHan Jhou](https://www.linkedin.com/in/yuhan-jhou-a0962b264/)

---

⭐ 如果這個專案對你有幫助，請給個星星支持！

[ci]: https://github.com/yu-codes/yu-codes.github.io/actions/workflows/ci.yml
[license]: https://github.com/yu-codes/yu-codes.github.io/blob/main/LICENSE
[demo]: https://yu-codes.github.io/
[jekyllrb]: https://jekyllrb.com/
