---
title: "資料品質與治理全攻略：Data Contract、Schema Evolution、Lineage、驗證與監控"
date: 2025-05-21 22:00:00 +0800
categories: [數據工程]
tags: [資料品質, Data Contract, Schema Evolution, Data Lineage, Great Expectations, Deequ, OpenLineage, Marquez, 驗證, 監控]
---

# 資料品質與治理全攻略：Data Contract、Schema Evolution、Lineage、驗證與監控

資料品質與治理是數據工程的核心，直接影響下游分析、AI 與業務決策的可靠性。本章將深入 Data Contract、Schema Evolution、資料驗證（Great Expectations、Deequ）、Data Lineage（OpenLineage、Marquez）、監控與自動化治理，結合理論、實作、面試熱點與常見誤區，幫助你打造高信賴的數據平台。

---

## Data Contract / Schema Evolution

### Data Contract

- 明確定義資料格式、欄位型別、約束、品質標準
- 生產者/消費者協議，防止 breaking change
- 支援自動驗證、Schema 驗證、資料 SLA

### Schema Evolution

- 支援資料格式隨需求變更（如新增欄位、型別變更）
- 格式：Avro、Parquet、ORC 皆支援 Schema 演進
- 策略：Backward/Forward/Full Compatibility

```python
# Avro Schema Evolution 範例
import fastavro
from fastavro.schema import load_schema

old_schema = load_schema('user_v1.avsc')
new_schema = load_schema('user_v2.avsc')
# fastavro 會自動處理兼容性
```

---

## Great Expectations / Deequ 規則撰寫

### Great Expectations

- Python 生態，支援資料驗證、型別檢查、缺失值、唯一性、分布檢查
- 可自動產生驗證報告，整合 Airflow、dbt

```python
import great_expectations as ge
df = ge.from_pandas(your_dataframe)
df.expect_column_values_to_not_be_null('user_id')
df.expect_column_values_to_be_unique('order_id')
```

### Deequ

- Scala/Java 生態，適合 Spark 大數據驗證
- 支援自動化規則、分布檢查、異常偵測

---

## Data Lineage：OpenLineage & Marquez

### Data Lineage

- 追蹤資料從來源到消費的全流程，支援審計、回溯、影響分析
- 關鍵於資料治理、法規遵循、異常排查

### OpenLineage

- 開放標準，支援多種 Orchestrator（Airflow、Spark、dbt）
- 可視化資料流、任務依賴、欄位級 lineage

### Marquez

- 開源 Data Lineage 平台，整合 OpenLineage
- 支援 API、UI、事件追蹤、審計

---

## 監控與自動化治理

- 結合 Orchestrator（Airflow、Dagster）自動化資料驗證
- 設計資料 SLA、異常告警、品質報表
- 定期驗證 Schema、Lineage，防止資料腐敗

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融、醫療、電商、AI 平台、法規遵循、資料治理專案

### 常見誤區

- 無 Data Contract，資料格式頻繁變動
- Schema Evolution 未設兼容策略，導致下游崩潰
- 資料驗證僅做一次，未持續監控
- Lineage 未落地，異常難追蹤

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Data Contract | 作用與設計要點？ |
| Schema Evolution | 如何兼容？ |
| Great Expectations/Deequ | 規則撰寫與自動化？ |
| Data Lineage | 如何落地與應用？ |
| 監控         | 如何設計資料品質告警？ |

---

## 使用注意事項

* Data Contract 需雙方協議並自動驗證
* Schema Evolution 建議設兼容等級與審核流程
* 資料驗證與 Lineage 建議自動化並定期審查

---

## 延伸閱讀與資源

* [Great Expectations 官方文件](https://docs.greatexpectations.io/docs/)
* [Deequ 官方文件](https://deequ.github.io/deequ/)
* [OpenLineage 官方文件](https://openlineage.io/docs/)
* [Marquez 官方文件](https://marquezproject.github.io/marquez/)
* [Schema Evolution in Avro/Parquet](https://docs.confluent.io/platform/current/schema-registry/avro.html#schema-evolution)

---

## 經典面試題與解法提示

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

## 結語

資料品質與治理是數據平台的生命線。熟悉 Data Contract、Schema Evolution、Lineage、驗證與監控，能讓你打造高信賴、可追溯的數據平台。下一章將進入版本控管與測試，敬請期待！
