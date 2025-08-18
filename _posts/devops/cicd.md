---
title: "從零開始實作 CI/CD：用 GitHub Actions 建立自動測試與部署流程"
date: 2025-05-14 14:0:00 +0800
categories: [DevOps]
tags: [GitHub Actions, CI/CD, 自動化部署, DevOps, Python]
---

# 從零開始實作 CI/CD：用 GitHub Actions 建立自動測試與部署流程

CI/CD 是現代軟體開發中的核心流程之一，能讓你從寫完程式到自動測試、自動部署都一氣呵成，不需要手動操作。  
本篇文章將以 **GitHub Actions 為主軸**，從零開始建立一套 CI/CD 流程，搭配簡單的 Python 應用進行實作。

---

## 🧠 CI/CD 是什麼？

- **CI（Continuous Integration）**：當你 push 程式碼時，自動執行建構、測試，確保每次整合都不會壞掉。
- **CD（Continuous Delivery / Deployment）**：
  - Delivery：可一鍵部署（人工觸發）
  - Deployment：自動部署（push 就上線）

---

## 🧱 CI 專案結構範例（Python App）

```

myapp/
├── main.py
├── requirements.txt
└── .github/
└── workflows/
└── ci.yml

```

`main.py`：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, CI/CD!"}
```

### ⚙️ Step 1：建立 `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          echo "可加入 pytest 等測試工具"
```

### ☁️ Step 2：部署到 Railway（或其他平台）

1. 將你的程式部署至 Railway，取得對應專案名稱與 Token

2. 在 GitHub → 專案 Settings → Secrets → 新增：

   * `RAILWAY_TOKEN`
   * `RAILWAY_PROJECT`

3. 更新 `ci.yml` 加入部署指令：

```yaml
      - name: Deploy to Railway
        run: |
          curl -sSL https://railway.app/install.sh | sh
          railway login --token ${{ secrets.RAILWAY_TOKEN }}
          railway up --project ${{ secrets.RAILWAY_PROJECT }}
```

---

### 🧪 進階功能建議（Optional）

* 使用 `matrix` 執行多版本測試（如 Python 3.8 / 3.10）
* 加上 Lint 工具（如 flake8）
* 加上通知（Slack、Telegram、Discord Webhook）
* 設定排程執行（on: schedule）

---

### ✅ 測試流程驗證方式

1. 修改程式碼並 commit
2. Push 至 main 分支
3. 前往 GitHub → Actions → 查看流程是否成功
4. 成功後自動部署，URL 可在 Railway 查看

---

### 🔐 管理 Secrets 建議

* 所有敏感資訊（Token、API Key）都用 GitHub Secrets 儲存
* 不要直接寫在 `yml` 或 `env` 檔案中
* 可用 `dotenv` 輔助本地測試環境

---

## 🚀 Continuous Delivery / Deployment（CD 是什麼？）

### 📦 Continuous Delivery（持續交付）

> 將測試通過的程式碼**自動建構與封裝好部署物**，但仍需「手動」觸發部署。

* 適合需要人工審查的上線流程（如 PR review、手動點選「Deploy」按鈕）
* 使用場景：銀行、法規敏感服務、需手動觸發的 staging/production deploy

### ⚡ Continuous Deployment（持續部署）

> 一旦程式碼通過測試，自動**部署到正式環境**，完全無需人手介入。

* 優點：開發 → 部署全自動，縮短回饋時間
* 缺點：需高度信任測試流程（測試 coverage 要夠）

### ✅ 兩者差異比較：

| 項目        | Continuous Delivery | Continuous Deployment |
| --------- | ------------------- | --------------------- |
| 部署是否自動    | ❌（人工觸發）             | ✅（全自動）                |
| 測試後是否立即上線 | ❌                   | ✅                     |
| 適合場景      | 審查、嚴謹控管             | 快速迭代、每日多次上線           |

---

### 👷 如何在 GitHub Actions 實作 CD？

#### ✅ 實作 Continuous Deployment（自動部署）

你在 `.github/workflows/ci.yml` 中直接加上「部署階段」，像這樣：

```yaml
      - name: Deploy to Railway
        if: success()  # 測試成功後才部署
        run: |
          curl -sSL https://railway.app/install.sh | sh
          railway login --token ${{ secrets.RAILWAY_TOKEN }}
          railway up --project ${{ secrets.RAILWAY_PROJECT }}
```

#### ✅ 實作 Continuous Delivery（人工觸發部署）

你可以改為使用 `workflow_dispatch`：

```yaml
on:
  push:
    branches:
      - main
  workflow_dispatch:  # 加上手動觸發入口
```

或者將 deploy job 設為條件啟用：

```yaml
jobs:
  deploy:
    if: github.event_name == 'workflow_dispatch'
    ...
```
---

## 📘 延伸資源推薦

* [GitHub Actions 官方教學](https://docs.github.com/en/actions)
* [Railway 官方文件](https://docs.railway.app/)
* [FastAPI + CI/CD 教學](https://fastapi.tiangolo.com/deployment/)

---

## ✅ 結語

CI/CD 能讓你的開發流程更穩定、更快、更可靠，特別是團隊合作、頻繁更新或產品上線時，幾乎是不可或缺的配備。
透過 GitHub Actions，搭配適合的部署平台，你可以用最低門檻，建立自動化的軟體交付流程。
