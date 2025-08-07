---
title: "資料採集與 Ingestion 全攻略：API、Webhook、CDC、Kafka、Schema 設計與實戰"
date: 2025-05-21 13:00:00 +0800
categories: [Data Engineering]
tags: [Data Ingestion, API, Webhook, CDC, Kafka, Kinesis, PubSub, Partition Key, Schema on Read, Schema on Write]
---

# 資料採集與 Ingestion 全攻略：API、Webhook、CDC、Kafka、Schema 設計與實戰

資料採集與 Ingestion 是數據工程的第一哩路，直接影響資料質量、延遲與下游處理效率。本章將深入 API/Webhook/CDC/File Drop 等資料來源，Kafka/Kinesis/PubSub 的 Partition Key 設計，Schema on Read/Write 策略，並結合實戰案例、圖解、面試熱點與常見誤區，幫助你打造高效穩健的資料流入管線。

---

## API / Webhook / CDC / File Drop

### API

- 主動拉取資料，適合定時同步、第三方服務整合
- 支援 REST、GraphQL、gRPC 等協議
- 需考慮認證、速率限制、錯誤重試
- 可用 ETL 工具（如 Airbyte、Fivetran）自動化 API 抽取
- 支援分頁、增量同步、異常監控與自動補償
- 常見場景：金融報價、社群資料、第三方 SaaS 整合

### Webhook

- 被動接收資料，事件驅動，適合即時通知
- 常用於支付、訂單、IoT 事件
- 需設計重試、簽名驗證、去重機制
- 可結合 Serverless（如 AWS Lambda）自動處理
- 支援事件溯源、重放、異常告警
- 常見場景：金流通知、物流狀態、即時警報

### Change Data Capture（CDC）

- 監控資料庫變更（如 binlog），實時同步到下游
- 工具：Debezium、AWS DMS、Oracle GoldenGate
- 適合資料庫遷移、實時 ETL、資料湖同步
- 支援全量+增量同步，需監控延遲與資料一致性
- 需處理 schema 變更、主鍵衝突、補償機制
- 常見場景：多活資料同步、即時報表、資料湖建設

### File Drop

- 透過 SFTP、雲端儲存（S3/GCS）上傳檔案
- 適合批次資料、外部供應商交換
- 需設計檔案命名規則、到檔通知、重複檢查
- 可結合 Lambda/SNS 實現自動觸發
- 支援檔案驗證（checksum）、自動歸檔、異常補償
- 常見場景：金融對帳、批次資料交換、外部資料導入

---

## Kafka／Kinesis／Pub/Sub 設計 Partition Key

### Partition Key 設計原則

- 決定資料分佈與消費順序，影響負載均衡與吞吐量
- 常見設計：用戶 ID、訂單號、地區等
- 熱點分佈（如熱門用戶）需避免單分區過載
- 可用 hash、range、複合 key 動態分配
- 動態調整分區數，需考慮資料重分佈與消費者重平衡
- 監控分區延遲、消費速率，及時調整策略

### Kafka 實戰

- Producer 發送訊息時指定 key，確保同 key 資料進同一分區
- 消費者可根據分區並行處理，提升吞吐
- 支援多 topic、跨資料中心同步、Exactly-once 語意
- 需設計死信佇列（DLQ）、重試與補償機制

```python
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('orders', key=b'user_123', value=b'order_data')
```

### Kinesis / PubSub

- Kinesis：Partition Key 決定 Shard，需考慮熱點與分片數
- 支援自動擴容、分片合併，需監控 Shard 利用率
- Pub/Sub：支援 Ordering Key，確保同 key 有序消費
- 支援 Dead Letter Topic、重試與監控

---

## Schema on Read vs. Write

### Schema on Write

- 資料寫入時即驗證格式，保證資料一致性
- 適合強結構化需求（如資料倉庫、金融交易）
- 支援 schema registry、版本控管、資料驗證
- 需設計 schema 變更流程與相容性檢查

### Schema on Read

- 資料寫入時不驗證格式，讀取時再解析
- 適合半結構化/多樣性資料（如資料湖、IoT）
- 支援多格式共存、彈性探索、資料湖治理
- 需設計讀取時 schema 驗證、異常補償

| 策略            | 優點                 | 缺點                 | 適用場景          |
| --------------- | -------------------- | -------------------- | ----------------- |
| Schema on Write | 資料一致性高，易治理 | 彈性低，需預先定義   | 倉庫、金融、報表  |
| Schema on Read  | 彈性高，支援多格式   | 讀取時易出錯，治理難 | 資料湖、IoT、探索 |

---

## 實戰案例：多源資料流入設計

- 結合 API 拉取、Webhook 事件、CDC 實時同步、File Drop 批次補數
- Kafka 作為統一訊息匯流排，設計合理 Partition Key 與多 topic
- 下游 ETL 根據資料型態選擇 Schema on Read/Write 策略
- 監控資料延遲、丟失、重複，設計告警、補償與死信佇列
- 實作資料 lineage、資料驗證、異常自動補救

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融交易、IoT 資料流、電商訂單、第三方 API 整合、資料湖建設、跨國多活同步

### 常見誤區

- Partition Key 設計不當導致分區傾斜、消費瓶頸
- Webhook 未設計重試與去重，資料丟失或重複
- CDC 未監控延遲與資料一致性，schema 變更未驗證
- File Drop 檔案命名/驗證不嚴謹，導致資料遺失
- Schema on Read 濫用，導致資料治理困難與查詢失敗

---

## 面試熱點與經典問題

| 主題           | 常見問題               |
| -------------- | ---------------------- |
| API vs Webhook | 適用場景與設計差異？   |
| CDC            | 如何確保資料一致性？   |
| Partition Key  | 如何避免分區熱點？     |
| Schema 策略    | 何時選用 Read/Write？  |
| Kafka          | 如何設計高吞吐資料流？ |

---

## 使用注意事項

* 資料流入需設計監控、告警與補償流程
* Partition Key 需根據資料分佈動態調整
## 延伸閱讀與資源

* [Kafka 官方文件](https://kafka.apache.org/documentation/)
* [Debezium CDC 教學](https://debezium.io/documentation/)
* [Schema on Read vs Write](https://www.confluent.io/blog/schema-on-read-vs-schema-on-write/)
* [AWS Kinesis Best Practices](https://docs.aws.amazon.com/streams/latest/dev/best-practices.html)

---

## 經典面試題與解法提示

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

## 結語

資料採集與 Ingestion 是數據工程的基石。熟悉多種資料來源、Partition Key 設計與 Schema 策略，能讓你打造高效穩健的資料流入管線。下一章將進入 ETL vs. ELT Pipeline，敬請期待！
