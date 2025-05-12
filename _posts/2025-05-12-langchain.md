---
title: "一文搞懂 LangChain：架構、模組、實作與應用全解析"
date: 2025-05-12 19:40:00 +0800
categories: [AI, LLM]
tags: [LangChain, LLM, AI 應用, Python, Chain of Thought]
---

# 一文搞懂 LangChain：架構、模組、實作與應用全解析

在 LLM（大型語言模型）應用爆炸性成長的今天，**LangChain** 成為最受歡迎的 Python 框架之一。它讓開發者能夠快速串接 OpenAI、Hugging Face、資料庫、搜尋引擎等模組，打造具備記憶、推理與資料擷取能力的 AI 應用。

> 本文將從 LangChain 的設計理念與模組出發，涵蓋實作範例與開發心法，讓你一篇就能掌握它的使用與擴展方式。

---

## 📦 什麼是 LangChain？

LangChain 是一套開源 Python 框架，專門用來構建由 LLM 駆動的應用程式（LLM-powered apps）。它主打：

- 將 **語言模型變成一個模組化可組裝的系統**
- 解決「從 prompt 到應用」中繁瑣的串接邏輯
- 支援記憶（memory）、工具（tools）、檔案搜尋（retriever）與 agent 機制

---

## 🧠 設計理念與核心架構

LangChain 的架構由下列核心組件構成：

### 1. **LLMs / ChatModels**
- 封裝如 OpenAI、Anthropic、HuggingFace、Google 等模型 API。
- 支援文字或對話輸出。

### 2. **Prompts**
- 提供可參數化的提示模板，如：
```python
  PromptTemplate.from_template("Translate {text} to {lang}")
```

### 3. **Chains**

* 將 LLM + Prompt + 後處理組成可重複使用的邏輯流程（如翻譯器、分類器）。
* 可用 `SimpleChain`, `SequentialChain`, `LLMChain` 等。

### 4. **Tools**

* 將搜尋引擎、計算機、API 包裝成可供 Agent 使用的函式介面。

### 5. **Agents**

* 有邏輯推理能力的執行引擎，能根據需求決定要呼叫哪些工具。
* 如 `ReAct Agent`, `Conversational Agent`, `OpenAI Functions Agent` 等。

### 6. **Memory**

* 為 Chain 或 Agent 增加長短期記憶，例如：

  * ConversationBufferMemory
  * ConversationSummaryMemory
  * VectorStoreRetrieverMemory

---

## 🔧 快速上手：打造一個 ChatGPT 工具機器人

### 安裝 LangChain 與依賴

```bash
pip install langchain openai
```

### 建立一個簡單 LLMChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="用簡單中文介紹：{topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("量子糾纏")
print(response)
```

---

## 🔍 更進階：使用 Agent 進行推理與工具調用

```python
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

agent.run("查一下台北天氣再算出明天溫度加上 5 的平方是多少？")
```

---

## 🧠 Memory 實作：對話紀錄與摘要記憶

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chat = ConversationChain(llm=llm, memory=memory)

chat.run("你好，我叫小明。")
chat.run("請記得我叫什麼。")  # LLM 會使用記憶資料產生回答
```

---

## 📚 Retriever + VectorStore + RAG（檔案檢索與生成）

LangChain 非常適合建構 Retrieval-Augmented Generation 系統。結合向量資料庫如 FAISS、Pinecone 或 Chroma，你可以實現 ChatPDF、問知識庫等功能。

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# 建立向量索引
loader = TextLoader("data.txt")
docs = loader.load()
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)

# 問答系統
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
qa_chain.run("這份文件的作者是誰？")
```

---

## ⚙️ LangChain 生態整合（可搭配）

| 工具 / 系統                       | 說明                            |
| ----------------------------- | ----------------------------- |
| **FAISS / Chroma / Weaviate** | 建立文件索引與向量查詢                   |
| **Streamlit / Gradio**        | 快速部署互動介面                      |
| **FastAPI / Flask**           | 建立 LangChain API              |
| **LangServe**                 | LangChain 官方部署工具              |
| **LangSmith**                 | Debug 與記錄 Chain 的追蹤工具（類似 APM） |

---

## 💡 常見應用場景

| 類型        | 說明                        |
| --------- | ------------------------- |
| Chatbot   | 建立具記憶與推理能力的對話機器人          |
| ChatPDF   | 將 PDF 轉成向量資料，供問答系統使用      |
| 資料助手      | 擷取知識庫內容並以自然語言回答           |
| 多工具 Agent | 可控制計算機、API、網頁查詢的智慧助手      |
| 自訂 Chain  | 包裝任務流程：例如「摘要 + 翻譯 + 寫信」流程 |

---

## 🎯 LangChain 面試常見問題（附回答建議）

1. **什麼情況下用 Chain 而不是 Agent？**

   > Chain 適合固定流程，Agent 適合根據語意判斷下一步行動。

2. **LangChain 如何串接私有知識庫？**

   > 透過 Retriever 模組與向量資料庫整合（RAG 架構）。

3. **Agent 如何選擇要使用哪個 Tool？**

   > 根據 prompt 訓練的推理策略，如 ReAct 原則。

4. **LangChain 與 LlamaIndex 差在哪？**

   > LangChain 為通用組合式框架，LlamaIndex 專注於文件摘要與索引查詢。

---

## 📘 推薦延伸資源

* [LangChain 官方網站](https://www.langchain.com/)
* [LangChain Docs](https://docs.langchain.com/)
* [LangChain Templates](https://github.com/langchain-ai/)

---

## 結語

LangChain 將 LLM 的應用推向了新的層次。無論是快速原型、RAG 系統、資料助手或智能工作流程，你都能用 LangChain 模組化地開發並調試完整流程。希望本文能成為你打造下一個 LLM 應用的起點。
