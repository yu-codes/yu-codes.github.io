---
title: "數據工程挑戰題庫：13 章經典面試題與解法提示"
date: 2025-05-21 23:59:00 +0800
categories: [Data Engineering]
tags: [面試題, 數據工程, 解題技巧, 白板題, 口試]
---

# 數據工程挑戰題庫：13 章經典面試題與解法提示

本章彙整前述 12 章數據工程主題的經典面試題，每章精選 10-15 題，涵蓋理論推導、實作、直覺解釋與白板題。每題附上解法提示與常見誤區，幫助你在面試與實戰中脫穎而出。

---

## DE1 數據工程大局觀

1. Data Engineer、ML Engineer、Analytics Engineer 差異與合作？
2. Batch 與 Streaming 架構選型原則？
3. 如何設計一條高可用的數據生命線？
4. 生命線各環節常見瓶頸與解法？
5. Streaming 架構如何保證 Exactly-once？
6. 如何協作跨部門數據專案？
7. 生命線設計如何兼顧延遲與資料質量？
8. Batch/Streaming 混合架構應用場景？
9. 生命線設計如何支援多消費者？
10. 如何用圖解說明數據流全流程？

---

## DE2 資料採集 & Ingestion

1. API、Webhook、CDC、File Drop 各自適用場景？
2. Kafka Partition Key 如何設計避免熱點？
3. Schema on Read/Write 差異與選型？
4. Webhook 如何設計去重與重試？
5. CDC 如何確保資料一致性與低延遲？
6. 多源資料流入如何統一治理？
7. Partition Key 動態調整策略？
8. Kafka 消費者如何實現高吞吐？
9. Schema on Read 濫用會有什麼風險？
10. 如何設計資料流入的監控與補償？

---

## DE3 ETL vs. ELT Pipeline

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

## DE4 資料格式 & 儲存

1. Parquet/ORC/Avro/CSV/JSON 適用場景？
2. Columnar 格式為何查詢快？
3. RLE/ZSTD 壓縮原理與優缺點？
4. Arrow 如何實現零複製？
5. 格式選型對下游查詢有何影響？
6. Parquet/ORC 如何設壓縮參數？
7. Arrow 與 Parquet 差異？
8. Memory Mapping 有哪些應用？
9. 格式選型如何兼顧效能與相容性？
10. 如何用 Python 操作 Parquet/Arrow？

---

## DE5 資料湖・倉庫・湖倉

1. Delta Lake/Iceberg/Hudi 差異與適用場景？
2. ACID 保證如何實現於資料湖？
3. Time-Travel 有哪些應用？
4. Redshift/BigQuery/Snowflake 儲存分層設計？
5. Hive Metastore 在多引擎架構的角色？
6. 湖倉如何支援即時分析與 ML？
7. 多引擎存取如何統一治理？
8. Time-Travel 如何設保存策略？
9. 湖倉查詢效能優化技巧？
10. 如何用 SQL 查詢歷史版本？

---

## DE6 分散式計算引擎

1. Spark/Flink/Beam 架構與適用場景？
2. Structured Streaming 如何實現 Exactly-once？
3. Shuffle 原理與效能瓶頸？
4. Broadcast Join 何時適用？風險？
5. Skew 偵測與解決方法？
6. 分區數如何設置？
7. Spark/Flink 資源管理與調優技巧？
8. Shuffle/Broadcast/Skew 如何聯合調優？
9. Spark UI/Flink Dashboard 如何定位瓶頸？
10. 如何用 Python 實作 Broadcast Join？

---

## DE7 Pandas & 新世代加速

1. Pandas 處理大檔案的限制與解法？
2. Dask/Modin/Polars 適用場景與優缺點？
3. Categorical 型態如何加速運算？
4. 10TB Click-stream 清洗流程設計？
5. Chunk 讀檔與分散式運算如何結合？
6. Vaex/Polars 記憶體外運算原理？
7. 分散式 DataFrame 工具的資源配置？
8. Pandas 向量化與 for 迴圈效能差異？
9. 分散式 ETL 常見踩坑與解法？
10. 如何用 Python 實作高效資料清洗？

---

## DE8 流式處理 & 實時分析

1. Exactly-once 如何實現？與 At-least-once 差異？
2. Tumbling/Sliding Window 適用場景？
3. Pinot/Druid/ClickHouse 架構與選型？
4. Checkpoint/Offset 管理細節？
5. 流式處理如何保證低延遲？
6. OLAP 系統如何支援即時查詢？
7. Window 聚合如何設計多層級指標？
8. Kafka/Flink 如何聯動保證資料一致？
9. 流式架構常見瓶頸與解法？
10. 如何用 Python 實作簡單流式聚合？

---

## DE9 特徵工程工坊

1. Lag/Rolling/Expanding 特徵設計與風險？
2. Target/Hash/Leave-one-out Encoding 差異？
3. Feature Store 如何設計與落地？
4. 線上/離線特徵一致性驗證？
5. 特徵自動化與資料驗證如何結合？
6. Category Encoding 如何防止洩漏？
7. Feature Store 版本控管與追溯？
8. 特徵工程常見踩坑與解法？
9. 如何用 Python 實作 Lag/Rolling/Target Encoding？
10. Feature Store 在推薦/金融的應用？

---

## DE10 資料品質 & 治理

1. Data Contract 如何設計與落地？
2. Schema Evolution 兼容策略有哪些？
3. Great Expectations/Deequ 如何自動化驗證？
4. Data Lineage 有哪些應用場景？
5. 監控與告警如何設計？
6. Lineage 與法規遵循的關係？
7. Schema Evolution 失敗會有什麼後果？
8. 如何用 Python 實作資料驗證？
9. Lineage 平台如何整合 Orchestrator？
10. 資料品質治理常見挑戰與解法？

---

## DE11 版本控管 & 測試

1. LakeFS/DVC 與 Git 的差異與優勢？
2. DAG 單元/整合測試如何設計？
3. CI/CD on Data 如何自動化資料驗證？
4. 資料/模型版本控管常見挑戰？
5. 測試覆蓋率如何提升？
6. LakeFS 如何支援資料回滾與分支？
7. DVC 適合哪些 ML pipeline？
8. Airflow/Prefect/Dagster 測試策略？
9. CI/CD pipeline 如何整合資料品質檢查？
10. 如何用 Python 撰寫 DAG 測試？

---

## DE12 安全・隱私・合規

1. PII 分類與分級管理如何設計？
2. Tokenization 與加密差異？
3. K-Anonymity 如何實作與評估？
4. GDPR/CCPA 對數據管線的實際要求？
5. RLS/IAM 權限設計原則？
6. 敏感資料刪除/匿名化流程？
7. 合規審計如何落地？
8. Tokenization 實作挑戰？
9. RLS 在 SQL/BI 工具的應用？
10. 數據平台如何兼顧安全、隱私與合規？

---

## 解題技巧與常見誤區

- **計算題**：先寫公式再帶數字，避免粗心。
- **推導題**：分步驟寫清楚，標明假設。
- **直覺題**：用圖解、生活例子輔助說明。
- **實作題**：熟悉 pandas、pyarrow、dask、spark 等常用 API。
- **常見誤區**：混淆定義、忽略資料治理、過度依賴單一工具。

---

## 結語

本題庫涵蓋數據工程經典面試題與解法直覺。建議每題都動手推導、實作與解釋，並多練習口頭表達。祝你面試順利、學習愉快！
