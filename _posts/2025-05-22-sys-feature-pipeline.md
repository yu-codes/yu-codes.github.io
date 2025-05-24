---
title: "Feature Pipeline 實戰：Batch/Streaming 特徵、CEP、特徵一致性解法"
date: 2025-05-22 15:00:00 +0800
categories: [大型系統設計與MLOps]
tags: [Feature Pipeline, Batch Feature, Streaming Feature, CEP, 特徵一致性, Training-Serving Skew]
---

# Feature Pipeline 實戰：Batch/Streaming 特徵、CEP、特徵一致性解法

特徵管線是 AI 系統的核心，直接影響模型表現與線上推論穩定性。本章將深入 Batch 特徵 ETL、Streaming 特徵（Click-stream CEP）、特徵一致性（Training-Serving Skew）解法，結合理論、實作、面試熱點與常見誤區，幫助你打造高效可維護的特徵平台。

---

## Batch 特徵：ETL → 話題特徵產生

- 利用 ETL 工具（如 Airflow、Spark）批次計算全量特徵
- 常見特徵：用戶統計、歷史行為、聚合指標
- 特徵產生流程：資料抽取 → 清洗 → 聚合 → 特徵存儲（Feature Store/DB）
- 支援定時更新（小時/天級），適合穩定特徵

```python
# Spark 批次特徵聚合範例
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("events.parquet")
user_feat = df.groupBy("user_id").agg({"amount": "sum", "event_time": "max"})
user_feat.write.parquet("user_features.parquet")
```

---

## Streaming 特徵：Click-stream CEP

- 利用流式引擎（Flink、Kafka Streams）即時計算特徵
- CEP（Complex Event Processing）：偵測複雜事件模式（如連續點擊、異常行為）
- 適合即時推薦、風控、監控

```python
# Flink CEP 偵測連續點擊事件（Python 偽碼）
from pyflink.datastream import StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()
# ...定義事件流與 CEP 規則...
# pattern: 5 分鐘內同用戶連續 3 次點擊
```

---

## 特徵一致性 (Training-Serving Skew) 解法

- Training-Serving Skew：訓練與推論特徵計算不一致，導致模型表現下降
- 解法：
  - 特徵邏輯共用（同一程式碼/模組）
  - Feature Store 管理特徵產生與版本
  - 線上/離線特徵驗證與監控
  - 定期回測線上特徵與離線資料分布

---

## Pipeline 設計與最佳實踐

- 批次/流式特徵分層管理，根據時效性選型
- CEP 適合複雜行為偵測，需設計狀態管理與容錯
- 特徵一致性驗證建議自動化，異常自動告警
- 特徵產生流程建議結合 Orchestrator（Airflow、Dagster）

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 推薦系統、廣告排序、金融風控、即時監控、用戶行為分析

### 常見誤區

- 批次/流式特徵混用未管理版本，導致推論錯誤
- CEP 狀態未持久化，重啟後丟失事件
- 特徵一致性驗證僅手動，異常難追蹤

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Batch vs Streaming 特徵 | 差異與選型？ |
| CEP         | 原理與應用場景？ |
| Training-Serving Skew | 如何解決？ |
| 特徵一致性驗證 | 如何設計？ |
| Pipeline 設計 | 如何兼顧時效與穩定？ |

---

## 使用注意事項

* 特徵邏輯建議共用模組，減少重複與偏差
* CEP 狀態需持久化與監控
* 特徵驗證與監控建議自動化

---

## 延伸閱讀與資源

* [Feature Engineering at Scale](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
* [Flink CEP 官方文件](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/libs/cep/)
* [Training-Serving Skew 解法](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#feature_store)
* [Feature Store 實踐](https://docs.feast.dev/)

---

## 經典面試題與解法提示

1. 批次/流式特徵差異與選型？
2. CEP 原理與應用場景？
3. Training-Serving Skew 如何解決？
4. 特徵一致性驗證如何設計？
5. Pipeline 設計如何兼顧時效與穩定？
6. 如何用 Python/Spark/Flink 實作特徵管線？
7. CEP 狀態管理與容錯？
8. 特徵驗證自動化策略？
9. Feature Store 如何輔助特徵一致性？
10. 特徵管線常見踩坑與解法？

---

## 結語

Feature Pipeline 是 AI 系統穩健運營的核心。熟悉 Batch/Streaming 特徵、CEP 與特徵一致性解法，能讓你打造高效可維護的特徵平台。下一章將進入 Feature Store 設計，敬請期待！
