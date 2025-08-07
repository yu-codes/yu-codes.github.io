---
title: "FastAPI 完全指南：從開發實戰到面試問答，一文掌握"
date: 2025-05-12 20:10:00 +0800
categories: [Backend]
tags: [FastAPI, Interview]
---

# FastAPI 完全指南：從開發實戰到面試問答，一文掌握

FastAPI 是近年來最受歡迎的 Python Web 框架之一，結合 **速度、簡潔與現代開發體驗**，尤其適合打造 **RESTful API 與 AI 應用後端**。其高效能與自動產生文件能力，使它成為眾多新創與企業的首選。

> 本文從最基礎的路由設計，到進階的驗證、安全性與部署，最後還附上面試常見問題解析，讓你從學習到實戰一次到位。

---

## 🚀 FastAPI 是什麼？

FastAPI 是一個以 **Python 3.6+ Type Hint** 為基礎打造的現代化 Web 框架。它的特色如下：

- ⚡ **超快效能**（可媲美 Node.js / Go）
- 🧠 **支援自動文件生成**（Swagger / ReDoc）
- 🛡️ **資料驗證內建**（使用 Pydantic）
- 🔗 **支援 OAuth2 / JWT 認證**
- ✅ **與 async / await 完美整合**

---

## 🧱 FastAPI 架構概覽

```

\[Client Request]
↓
\[FastAPI 路由處理]
↓
\[Pydantic 參數驗證]
↓
\[商業邏輯（可 async）]
↓
\[回應 JSON + 自動生成 API 文件]

```

---

## 🔧 快速起步：Hello FastAPI

### 安裝 FastAPI + uvicorn

```bash
pip install fastapi uvicorn
```

### 建立一個簡單 API 檔案 `main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
```

### 啟動伺服器

```bash
uvicorn main:app --reload
```

* 預設會啟動在 [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger 文件：`http://127.0.0.1:8000/docs`
* ReDoc 文件：`http://127.0.0.1:8000/redoc`

---

## 📥 路由與參數處理

### 路徑參數

```python
@app.get("/users/{user_id}")
def read_user(user_id: int):
    return {"user_id": user_id}
```

### 查詢參數

```python
@app.get("/items/")
def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

---

## 🧪 請求驗證：使用 Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

@app.post("/items/")
def create_item(item: Item):
    return item
```

* FastAPI 會自動驗證資料結構
* 錯誤會自動回傳 422 錯誤訊息與說明

---

## 🔄 回應模型與篩選欄位

```python
class ItemResponse(BaseModel):
    name: str
    price: float

@app.post("/items/", response_model=ItemResponse)
def create_item(item: Item):
    return item
```

---

## 🧵 非同步支援

```python
import asyncio

@app.get("/wait/")
async def wait_3s():
    await asyncio.sleep(3)
    return {"done": True}
```

* FastAPI 完整支援 async def 與非同步 I/O 操作（例如連資料庫）

---

## 🔒 JWT 驗證與 OAuth2

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

FastAPI 內建 OAuth2PasswordBearer 與 JWT 結合，可實作登入、授權流程。此處略述，可詳見官方文件或後續文章介紹。

---

## 🗂️ 路由分模組與專案架構

```
project/
│
├── main.py
├── routers/
│   └── users.py
├── models/
│   └── item.py
└── services/
    └── database.py
```

在 `main.py` 中掛載：

```python
from routers import users
app.include_router(users.router)
```

---

## 💾 整合資料庫（以 SQLite + SQLAlchemy 為例）

```bash
pip install sqlalchemy
```

### models.py

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
```

### 建立連線

```python
engine = create_engine("sqlite:///./test.db")
Base.metadata.create_all(bind=engine)
```

你也可以使用 ORM 工具如 SQLModel、Tortoise ORM 或 async SQLAlchemy。

---

## 🧪 自動化測試（使用 pytest）

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}
```

---

## 🛡️ CORS、例外處理與中介層

### CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 自定例外處理

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"message": str(exc)})
```

---

## 🚀 FastAPI 部署選項

* `uvicorn main:app --host 0.0.0.0 --port 8000`
* 使用 Gunicorn + Uvicorn workers
* Dockerize（建議）

### Dockerfile 範例

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 💼 面試常見問題與回答建議

1. **為何選用 FastAPI 而非 Flask？**

   > FastAPI 有型別支援、自動文件、非同步、速度快，對大型 API 更友好。

2. **FastAPI 如何處理輸入驗證？**

   > 透過 Pydantic 模型，類似 schema 驗證。

3. **如何測試 FastAPI API？**

   > 使用 `TestClient` 結合 `pytest`，可撰寫單元與整合測試。

4. **你曾如何在 FastAPI 中設計模組化結構？**

   > 使用 `routers/`、`models/`、`services/` 等目錄劃分功能層，並用 DI 傳遞共用資源。

5. **FastAPI 中的 async 特性為何重要？**

   > 支援 async I/O 可提升處理效率，對高併發與 API 效能有顯著幫助。

---

## 📘 延伸學習資源

* [FastAPI 官方文件](https://fastapi.tiangolo.com/)
* [SQLModel 專案](https://sqlmodel.tiangolo.com/)
* [Typer CLI 工具（FastAPI 作者另一作品）](https://typer.tiangolo.com/)

---

## ✅ 結語

FastAPI 提供了現代 Python API 開發的理想選擇，無論你是快速原型、部署微服務，還是打造 AI 系統的中介層，FastAPI 都能提供優雅且高效的解決方案。掌握本文內容，你不僅能立即上手實作，亦能在面試中從容應對各種提問。
