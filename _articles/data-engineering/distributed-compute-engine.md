---
title: "分散式計算引擎全攻略：Spark、Flink、Beam、Shuffle、Skew 與調優實戰"
date: 2025-05-21 17:00:00 +0800
categories: [Data Engineering]
tags: [分散式計算, Spark, Flink, Beam, Structured Streaming, Shuffle, Broadcast, Skew, 調優]
---

# 分散式計算引擎全攻略：Spark、Flink、Beam、Shuffle、Skew 與調優實戰

分散式計算引擎是大數據處理的核心。從 Spark Core/SQL/Structured Streaming，到 Flink、Beam 的流批一體，再到 Shuffle、Broadcast、Skew 等調優技巧，這些技術決定了資料處理的效率、可擴展性與穩定性。本章將深入原理、架構圖解、實戰調優、面試熱點與常見誤區，幫助你打造高效能數據平台。

---

## Spark Core / SQL / Structured Streaming

### Spark Core

- RDD（彈性分散式資料集）為基礎，支援 Map/Reduce、分區、容錯
- 適合複雜轉換、低階控制

### Spark SQL

- DataFrame/Dataset API，支援 SQL 查詢、優化器（Catalyst）
- 適合 ETL、資料探索、BI 報表

### Structured Streaming

- 以 DataFrame 為基礎的流式處理，支援 Exactly-once、Window 聚合
- 支援與 Kafka、Kinesis、文件系統整合

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("demo").getOrCreate()
df = spark.read.parquet("data.parquet")
df.groupBy("col").count().show()
```

---

## Flink vs. Beam：Event Time & Watermark

### Flink

- 原生流批一體，支援 Event Time、Window、狀態管理
- Watermark 機制處理亂序資料，保證準確性
- 適合低延遲、複雜流式分析

### Beam

- 統一批次/流式 API，支援多執行引擎（Flink、Spark、Dataflow）
- 強調可攜性與跨平台

| 引擎  | 流批一體 | Event Time | Watermark | 適用場景           |
| ----- | -------- | ---------- | --------- | ------------------ |
| Spark | 部分     | 支援       | 有限      | ETL、批次分析      |
| Flink | 原生     | 強         | 強        | 實時流式、複雜事件 |
| Beam  | 統一 API | 強         | 強        | 跨平台、雲端       |

---

## Shuffle、Broadcast、Skew 調優

### Shuffle

- 分散式運算中資料重分區，常見於 groupBy、join
- Shuffle 過多會導致磁碟 I/O、網路壅塞

### Broadcast

- 小表廣播到所有節點，避免大表 shuffle
- 適合小型維度表 join

### Skew（資料傾斜）

- 單一分區資料量過大，導致部分節點瓶頸
- 解法：Salting、動態分區、Skew Join

```python
# Spark Broadcast Join
small_df = spark.read.parquet("dim.parquet")
large_df = spark.read.parquet("fact.parquet")
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), "key")
```

---

## 實戰調優與資源管理

- 合理設置分區數（partition），避免過多/過少
- 調整 shuffle buffer、executor memory、core 數
- 監控 DAG、Stage、Task 執行情況，定位瓶頸
- 使用 Spark UI/Flink Dashboard 進行資源與任務監控

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- ETL、批次分析、即時流式處理、複雜事件監控、資料湖建設

### 常見誤區

- Shuffle 過多導致效能瓶頸
- Broadcast Join 濫用導致記憶體爆炸
- Skew 未處理導致部分節點拖慢全局
- 分區數設置不當，資源利用率低

---

## 面試熱點與經典問題

| 主題                 | 常見問題                |
| -------------------- | ----------------------- |
| Spark vs Flink       | 架構與適用場景？        |
| Structured Streaming | Exactly-once 如何實現？ |
| Shuffle              | 原理與效能影響？        |
| Broadcast            | 何時用？有何風險？      |
| Skew                 | 如何偵測與解決？        |

---

## 使用注意事項

* 分區、Shuffle、Broadcast 需根據資料量與節點資源調整
* 流式處理需設計 Watermark 與容錯機制
* 調優建議結合監控工具與實驗

---

## 延伸閱讀與資源

* [Apache Spark 官方文件](https://spark.apache.org/docs/latest/)
* [Apache Flink 官方文件](https://nightlies.apache.org/flink/flink-docs-release-1.17/)
* [Apache Beam 官方文件](https://beam.apache.org/documentation/)
* [Spark 調優指南](https://spark.apache.org/docs/latest/tuning.html)
* [Flink 調優指南](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/ops/tuning/)

---

## 經典面試題與解法提示

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

## 結語

分散式計算引擎是大數據處理的核心。熟悉 Spark、Flink、Beam、Shuffle、Broadcast、Skew 與調優技巧，能讓你打造高效能、可擴展的數據平台。下一章將進入 Pandas 與新世代加速工具，敬請期待！
