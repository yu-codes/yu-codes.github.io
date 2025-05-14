---
title: "LINE Chatbot 完整開發指南：從 Channel 設定到 FastAPI 實作與部署"
date: 2025-05-14 12:00:00 +0800
categories: [Backend]
tags: [LINE, FastAPI, Webhook, Bot, Python]
---

# LINE Chatbot 完整開發指南：從 Channel 設定到 FastAPI 實作與部署

LINE 是台灣最受歡迎的通訊工具之一，而 LINE Messaging API 提供了強大的聊天機器人開發能力。本篇文章將帶你完整實作一個 LINE Chatbot，從後台設定到 Webhook 實作與部署，全流程拆解，讓你快速上線屬於自己的 LINE 機器人。

---

## 🧱 開發前準備

### ✅ 你需要準備：

- 一個 LINE 帳號
- [LINE Developers Console](https://developers.line.biz/)
- Python 開發環境（推薦搭配 FastAPI）
- Ngrok（用於本地開發時暴露 Webhook）

---

## 📡 Step 1：註冊與設定 LINE Bot Channel

1. 前往 [LINE Developers Console](https://developers.line.biz/)
2. 建立一個 Provider
3. 在 Provider 底下新增一個 Messaging API Channel
4. 記下以下資訊：
   - **Channel Secret**
   - **Channel Access Token**
5. 啟用 Webhook 並打開「允許訊息」開關

---

## 🌐 Step 2：建立 Webhook 接收端（使用 FastAPI）

### 安裝套件：

```bash
pip install fastapi uvicorn line-bot-sdk
```

---

### 建立 `main.py`：

```python
from fastapi import FastAPI, Request, Header
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# 初始化
app = FastAPI()
line_bot_api = LineBotApi("你的 Channel Access Token")
handler = WebhookHandler("你的 Channel Secret")

# Webhook 接收點
@app.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(...)):
    body = await request.body()
    handler.handle(body.decode("utf-8"), x_line_signature)
    return "OK"

# 訊息處理邏輯
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_msg = event.message.text
    reply_msg = f"你說的是：{user_msg}"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_msg)
    )
```

---

### 使用 Ngrok 曝露本地網址：

```bash
ngrok http 8000
```

> 複製 Ngrok 的 HTTPS URL，貼到 LINE Bot 的 Webhook URL 設定中（例如 `https://xxxx.ngrok.io/webhook`）

---

## 🧪 Step 3：測試機器人

1. 加入你的 Bot 為好友（掃 Channel 裡的 QR Code）
2. 發送任意訊息給 Bot
3. 應收到自動回覆：「你說的是：...」

---

## ⚙️ 延伸功能與進階應用

### 📌 回覆貼圖、圖片、按鈕樣板等

```python
from linebot.models import StickerMessage, ImageMessage, TemplateSendMessage

# 回覆按鈕樣板
line_bot_api.reply_message(
    event.reply_token,
    TemplateSendMessage(
        alt_text='選單',
        template=ButtonsTemplate(
            title='選擇功能',
            text='請選擇',
            actions=[
                MessageAction(label='說明', text='說明'),
                URIAction(label='官方網站', uri='https://example.com')
            ]
        )
    )
)
```

---

### 🧠 加入對話邏輯 / AI 模型 / 後端 API 整合

你可以將收到的訊息轉送到：

* Hugging Face 模型 API（串 GPT、問答等）
* 自建 Flask/FastAPI 後端
* OpenAI / Gemini / Claude 等 AI 模型

---

## 🚀 Step 4：正式部署建議（選用）

* 使用 [Render](https://render.com/)、[Railway](https://railway.app/)、或 [Fly.io](https://fly.io/) 快速部署 FastAPI
* 使用 Docker 包裝應用並部署到雲端主機（如 EC2、VPS）

---

## 🔒 安全性與驗證建議

* 驗證 `x-line-signature` 是否有效
* 設定 IP 白名單（如部署於私有伺服器）
* 使用 `.env` 儲存密鑰（使用 dotenv 套件）

---

## 📘 延伸資源推薦

* [LINE Messaging API 官方文件](https://developers.line.biz/en/reference/messaging-api/)
* [line-bot-sdk-python GitHub](https://github.com/line/line-bot-sdk-python)
* [FastAPI 教學系列](https://fastapi.tiangolo.com/)
* [Ngrok 官方](https://ngrok.com/)

---

## ✅ 結語

LINE Chatbot 的開發流程不難，只要掌握 Webhook、事件處理與 FastAPI 架構，就能快速開發出實用的聊天服務。無論是客服、問答助手、資料回報系統，甚至整合 AI 模型，都能透過 LINE Bot 實現。