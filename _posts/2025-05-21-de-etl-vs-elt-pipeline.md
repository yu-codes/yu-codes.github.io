---
title: "ETL vs. ELT Pipeline 全解析：轉換時機、數據建模、Orchestrator 與實戰設計"
date: 2025-05-21 14:00:00 +0800
categories: [Data Engineering]
tags: [ETL, ELT, Data Pipeline, Star Schema, Snowflake Schema, SCD, Airflow, Dagster, Prefect, Orchestrator]
---

# ETL vs. ELT Pipeline 全解析：轉換時機、數據建模、Orchestrator 與實戰設計

數據管線的設計直接影響資料質量、查詢效率與維運成本。ETL（Extract-Transform-Load）與 ELT（Extract-Load-Transform）是現代數據工程的兩大主流。從轉換時機、數據建模（Star/Snowflake Schema、SCD）、到 Orchestrator（Airflow、Dagster、Prefect）選型與 DAG 設計，本章將結合理論、圖解、實戰、面試熱點與常見誤區，幫助你打造高效可維護的數據管線。

---

## ETL vs. ELT Pipeline：轉換時機與架構差異

### ETL（Extract-Transform-Load）

- 先抽取（Extract）→ 轉換（Transform）→ 載入（Load）到目標資料庫
- 適合傳統資料倉庫、轉換邏輯複雜、需嚴格資料治理

### ELT（Extract-Load-Transform）

- 先抽取→ 載入到資料湖/倉庫→ 再轉換
- 適合現代雲端倉庫（BigQuery、Snowflake），利用倉庫運算力做轉換
- 支援即時查詢、彈性探索、Schema on Read

| 管線類型 | 轉換時機 | 適用場景           | 優點           | 缺點             |
| -------- | -------- | ------------------ | -------------- | ---------------- |
| ETL      | 載入前   | 傳統倉庫、嚴格治理 | 資料一致性高   | 彈性低、維運重   |
| ELT      | 載入後   | 雲端倉庫、資料湖   | 彈性高、易探索 | 治理難、轉換延遲 |

---

## Star / Snowflake Schema、維度表更新 (SCD-I/II)

### Star Schema（星型模式）

- 以事實表為中心，連接多個維度表
- 查詢效率高，結構簡單，適合 BI 報表

### Snowflake Schema（雪花模式）

- 維度表再細分為子維度表，正規化更高
- 節省儲存空間，查詢需多次 Join

### 維度表更新（SCD, Slowly Changing Dimension）

- SCD-I：直接覆蓋舊資料，不保留歷史
- SCD-II：保留歷史版本，新增有效/失效欄位，支援時間旅行查詢
- SCD-III：僅保留部分歷史（如前一狀態）

```sql
-- SCD-II 範例
ALTER TABLE dim_customer ADD COLUMN valid_from DATE, valid_to DATE;
-- 新增新版本時，舊資料 valid_to 設為當前日期，新資料 valid_from 設為當前日期
```

---

## Orchestrator：Airflow／Dagster／Prefect

### Airflow

- 最主流的 Workflow Orchestrator，支援 DAG、排程、重試、監控
- Python 編寫 DAG，支援多種 Operator（Bash、Python、Spark、BigQuery 等）
- 社群活躍，易於擴展

### Dagster

- 強調型別安全、資產導向、開發體驗佳
- 支援資料資產追蹤、測試、分區管理

### Prefect

- 雲端原生，支援動態 DAG、易於本地開發與部署
- 強調易用性與彈性，適合中小型團隊

```python
# Airflow DAG 範例
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract():
    # ...資料抽取邏輯...
    pass

def transform():
    # ...資料轉換邏輯...
    pass

def load():
    # ...資料載入邏輯...
    pass

with DAG('etl_example', start_date=datetime(2023,1,1), schedule_interval='@daily') as dag:
    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='transform', python_callable=transform)
    t3 = PythonOperator(task_id='load', python_callable=load)
    t1 >> t2 >> t3
```

---

## Pipeline 設計實戰與最佳實踐

- 根據資料量、查詢需求選擇 ETL 或 ELT
- 數據建模時優先考慮 Star Schema，需正規化再用 Snowflake
- SCD-II 適合需追蹤歷史的維度表（如用戶狀態、產品價格）
- Orchestrator DAG 設計需考慮依賴、重試、監控、通知
- 測試與 CI/CD：DAG 單元測試、資料驗證、版本控管

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融報表、電商分析、IoT 數據湖、即時 BI、資料治理平台

### 常見誤區

- ELT 濫用導致資料湖變資料沼澤（Data Swamp）
- SCD 實作不當導致歷史資料錯亂
- Orchestrator DAG 無監控，失敗難追蹤
- Star/Snowflake Schema 混用導致查詢複雜

---

## 面試熱點與經典問題

| 主題                    | 常見問題             |
| ----------------------- | -------------------- |
| ETL vs ELT              | 何時選用？優缺點？   |
| Star vs Snowflake       | 結構差異與查詢效率？ |
| SCD-I/II                | 如何設計與應用？     |
| Airflow/Dagster/Prefect | 選型與實戰經驗？     |
| DAG 設計                | 如何處理依賴與重試？ |

---

## 使用注意事項

* Pipeline 設計需兼顧資料質量、查詢效率與維運成本
* Orchestrator 選型需考慮團隊規模、技術棧與監控需求
* SCD-II 維度表需設計有效/失效欄位與主鍵

---

## 延伸閱讀與資源

* [Airflow 官方文件](https://airflow.apache.org/docs/)
* [Dagster 官方文件](https://docs.dagster.io/)
* [Prefect 官方文件](https://docs.prefect.io/)
* [Star vs Snowflake Schema](https://www.guru99.com/star-snowflake-data-warehousing.html)
* [SCD 維度表設計](https://www.sqlshack.com/slowly-changing-dimensions-explained-with-examples/)

---

## 經典面試題與解法提示

1. ETL 與 ELT 差異與選型原則？
2. Star/Snowflake Schema 適用場景？
3. SCD-I/II/III 設計與查詢方式？
4. Airflow DAG 如何設計依賴與監控？
5. ELT Pipeline 如何防止 Data Swamp？
6. Orchestrator 選型經驗與優缺點？
7. Pipeline 失敗如何自動補償與通知？
8. SCD-II 如何查詢歷史狀態？
9. DAG 單元測試與 CI/CD 如何實作？
10. Pipeline 設計如何兼顧彈性與治理？

---

## 結語

ETL/ELT Pipeline 是數據工程的核心。熟悉轉換時機、數據建模、Orchestrator 與 DAG 設計，能讓你打造高效可維護的數據管線。下一章將進入資料格式與儲存，敬請期待！
