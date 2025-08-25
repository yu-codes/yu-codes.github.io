# Yu's Tech Blog

[![Build and Deploy](https://github.com/yu-codes/yu-codes.github.io/actions/workflows/pages-deploy.yml/badge.svg)](https://github.com/yu-codes/yu-codes.github.io/actions/workflows/pages-deploy.yml)

> 一個專注於技術分享的個人部落格，涵蓋機器學習、深度學習、系統設計、演算法等領域。

🌐 **網站**: https://yu-codes.github.io

## 📚 內容領域

- **機器學習** - 核心概念、演算法實現
- **深度學習** - 模型架構、訓練技巧
- **系統設計** - 高可用性、可擴展性架構
- **演算法** - 資料結構與演算法分析
- **雲端技術** - AWS、Docker、Kubernetes
- **後端開發** - API 設計、資料庫優化

## � 專案結構

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
│   │   ├── archives.md          # 歸檔頁面
│   │   ├── categories.md        # 分類頁面
│   │   ├── tags.md              # 標籤頁面
│   │   ├── articles.md          # 技術文章頁面
│   │   └── resume.md            # 個人履歷
│   ├── _layouts/                # 頁面佈局模板
│   │   ├── home.html            # 首頁佈局
│   │   ├── articles-home.html   # 文章列表佈局
│   │   ├── post.html            # 文章頁面佈局
│   │   └── ...
│   ├── _includes/               # 可重用組件
│   ├── _data/                   # 結構化資料
│   │   ├── authors.yml          # 作者資訊
│   │   ├── contact.yml          # 聯絡方式
│   │   └── locales/             # 多語言支援
│   └── assets/                  # 靜態資源
│       ├── css/                 # 編譯後的 CSS
│       ├── js/                  # 編譯後的 JavaScript
│       └── img/                 # 圖片資源
│
├── 📂 源碼和樣式
│   ├── _sass/                   # SCSS 樣式源碼
│   │   ├── addon/               # 額外功能樣式
│   │   ├── layout/              # 版面佈局樣式
│   │   └── themes/              # 主題色彩配置
│   └── _javascript/             # JavaScript 源碼
│       ├── commons.js           # 通用功能
│       ├── home.js              # 首頁功能
│       └── modules/             # 可重用模組
│
├── 🔧 開發工具和配置
│   ├── .config/                 # 配置文件目錄
│   │   ├── .editorconfig        # 編輯器統一配置
│   │   ├── .markdownlint.json   # Markdown 檢查規則
│   │   └── .stylelintrc.json    # CSS 檢查規則
│   ├── .vscode/                 # VS Code 配置
│   │   ├── settings.json        # 編輯器設定
│   │   ├── tasks.json           # 任務配置
│   │   └── extensions.json      # 推薦擴展
│   ├── scripts/                 # 開發腳本
│   │   ├── dev.bat              # Windows 快捷腳本
│   │   └── dev.sh               # Unix 快捷腳本
│   ├── tools/                   # 工具腳本
│   │   ├── run.sh               # 啟動腳本
│   │   ├── test.sh              # 測試腳本
│   │   ├── clean.bat            # Windows 清理腳本
│   │   └── clean.sh             # Unix 清理腳本
│   ├── rollup.config.js         # JavaScript 打包配置
│   ├── purgecss.js              # CSS 優化配置
│   └── eslint.config.js         # JavaScript 檢查配置
│
├── 🐳 Docker 開發環境
│   └── docker/                  # Docker 配置目錄
│       ├── Dockerfile           # Docker 映像配置
│       ├── docker-compose.yml   # 基本 Docker Compose
│       ├── docker-compose.dev.yml # 開發環境配置
│       ├── .dockerignore        # Docker 忽略文件
│       ├── docker-dev.bat       # Windows 開發腳本
│       ├── docker-dev.sh        # Unix 開發腳本
│       └── README.md            # Docker 使用說明
│
├── 🔍 版本控制和自動化
│   ├── .github/                 # GitHub 配置
│   │   └── workflows/           # GitHub Actions
│   │       └── pages-deploy.yml # 自動部署工作流
│   ├── _plugins/                # Jekyll 插件
│   │   └── articles-pages-generator.rb # 自定義頁面生成器
│   ├── .gitignore               # Git 忽略文件
│   ├── .gitattributes           # Git 屬性配置
│   └── .gitmodules              # Git 子模組配置
│
├── 📚 文件和說明
│   ├── README.md                # 專案說明（本檔案）
│   ├── LICENSE                  # MIT 授權
│   └── docs/                    # 專案文件
│       └── PROJECT_STRUCTURE.md # 詳細結構說明
│
└── 🏗️ 建置和快取（被 .gitignore 忽略）
    ├── _site/                   # Jekyll 建置輸出
    ├── .jekyll-cache/           # Jekyll 快取
    ├── .bundle/                 # Ruby Bundle 快取
    └── node_modules/            # Node.js 依賴
```

## �🚀 本地開發

### 使用 Docker（推薦）

```bash
# Windows
.\scripts\dev.bat

# macOS/Linux
bash scripts/dev.sh
```

### 傳統方式

```bash
# 安裝依賴
bundle install
npm install

# 啟動開發服務器
bundle exec jekyll serve --livereload
```

## 📝 寫作

### 新增文章

```bash
# 部落格文章（需要日期）
touch _posts/YYYY-MM-DD-title.md

# 技術文章（按分類組織）
touch _articles/category/title.md
```

### Front Matter 範例

```yaml
---
title: "文章標題"
date: 2025-01-01 12:00:00 +0800
categories: [Category, Subcategory]
tags: [tag1, tag2, tag3]
description: "簡潔的文章描述，用於 SEO 和社群分享"
pin: true          # 設為精選文章
image:
  path: /assets/img/cover.jpg
  alt: "封面圖片說明"
---
```

### 內容組織

- **文章 (_articles/)**: 適合技術教學和參考文檔，按主題分類
- **部落格 (_posts/)**: 適合時間性內容，按日期排序

## 🎨 主要功能

- **個人化首頁**: Hero section、統計卡片、精選文章
- **雙重內容系統**: Posts + Articles 統一管理
- **自動分類標籤**: 支援混合內容的分類和標籤頁面
- **Docker 開發環境**: 一鍵啟動，環境一致性
- **自動化部署**: GitHub Actions 自動部署到 GitHub Pages

## 🛠️ 技術架構

- **Jekyll 4.4** - 靜態網站生成器
- **Chirpy Theme** - 自定義修改版本
- **Docker** - 開發環境容器化
- **GitHub Pages** - 自動部署平台
- **Ruby 3.3** - 後端語言
- **Node.js** - 前端工具鏈
- **SCSS + Bootstrap 5** - 樣式框架

## 🚀 部署

推送到 `main` 分支會自動觸發 GitHub Actions 進行部署。

### 開發命令

```bash
# Docker 方式
.\scripts\dev.bat dev        # 啟動開發服務器
.\scripts\dev.bat build      # 建置生產版本
.\scripts\dev.bat clean      # 清理資源

# 傳統方式
bundle exec jekyll serve     # 啟動服務器
npm run build               # 建置前端
```

## 📄 授權

本專案採用 [MIT 授權](LICENSE)。