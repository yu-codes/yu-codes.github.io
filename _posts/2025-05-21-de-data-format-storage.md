---
title: "資料格式與儲存全解析：CSV, JSON, Parquet, Arrow, Columnar 優勢與壓縮"
date: 2025-05-21 15:00:00 +0800
categories: [數據工程]
tags: [資料格式, CSV, JSON, Avro, Parquet, ORC, Arrow, Columnar, 壓縮, RLE, ZSTD, Memory Mapping]
---

# 資料格式與儲存全解析：CSV, JSON, Parquet, Arrow, Columnar 優勢與壓縮

資料格式與儲存設計直接影響數據管線的效能、查詢速度與成本。從傳統的 CSV、JSON，到現代的 Avro、Parquet、ORC、Arrow，這些格式各有優缺點與適用場景。本章將深入格式比較、Columnar 優勢、壓縮編碼（RLE, ZSTD）、Arrow Memory Mapping、零複製等主題，結合理論、實作、面試熱點與常見誤區，幫助你打造高效能數據平台。

---

## 常見資料格式比較：CSV, JSON, Avro, Parquet, ORC

| 格式    | 結構       | 壓縮 | 架構 | 適用場景           | 優點                | 缺點                |
|---------|------------|------|------|--------------------|---------------------|---------------------|
| CSV     | 純文字     | 無   | Row  | 簡單批次、匯入匯出 | 易讀、通用          | 無型別、無壓縮      |
| JSON    | 樹狀/半結構| 無   | Row  | API、半結構資料    | 彈性高、易解析      | 無型別、無壓縮      |
| Avro    | 二進位     | 有   | Row  | Kafka、CDC         | 支援 schema、壓縮   | 不易人工檢查        |
| Parquet | 二進位     | 有   | Columnar | 數據湖、分析查詢 | 查詢快、壓縮佳      | 不適合頻繁寫入      |
| ORC     | 二進位     | 有   | Columnar | Hive、Spark       | 查詢快、壓縮佳      | 工具支援較少        |

---

## Columnar 優勢與壓縮編碼（RLE, ZSTD）

### Columnar 優勢

- 同欄位資料連續存放，查詢特定欄位時 I/O 最小化
- 適合分析型查詢（OLAP）、大數據平台（Spark、BigQuery、Redshift）
- 支援高效壓縮、向量化運算

### 壓縮編碼

- RLE（Run-Length Encoding）：連續重複值壓縮，適合低基數欄位
- ZSTD（Zstandard）：現代高效壓縮演算法，兼顧壓縮率與速度
- Parquet/ORC 支援多種壓縮（Snappy, Gzip, ZSTD）

```python
import pandas as pd

df = pd.DataFrame({'a': [1]*1000 + [2]*1000})
df.to_parquet('data.parquet', compression='zstd')
```

---

## Arrow Memory Mapping & 零複製

### Apache Arrow

- 記憶體中統一欄式格式，支援多語言零複製資料交換
- 適合高效資料傳輸、機器學習、即時查詢

### Memory Mapping

- Arrow 支援 mmap，直接在記憶體中操作大檔案，無需整檔載入
- 零複製（Zero-Copy）：不同系統/語言間共享資料，無需序列化/反序列化

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({'a': [1, 2, 3]})
pq.write_table(table, 'demo.parquet')
table2 = pq.read_table('demo.parquet', memory_map=True)
```

---

## 格式選型與儲存分層實戰

- 批次交換/外部供應商：CSV、JSON
- 流式/CDC：Avro（支援 schema 演進）
- 數據湖/分析查詢：Parquet、ORC（欄式、壓縮佳）
- 即時運算/多語言：Arrow
- 儲存分層：原始區（Raw）、清洗區（Staging）、分析區（Analytics）

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 數據湖、資料倉庫、即時分析、機器學習特徵存取、跨語言資料交換

### 常見誤區

- 大數據查詢仍用 CSV，導致效能瓶頸
- Parquet/ORC 未正確設壓縮，浪費儲存
- Arrow 未善用零複製，資料傳輸低效
- 格式選型未考慮下游工具支援

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Parquet vs CSV | 查詢/壓縮/適用場景？ |
| Columnar 優勢 | 為何適合 OLAP？ |
| RLE/ZSTD     | 原理與適用欄位？ |
| Arrow        | 零複製如何實現？ |
| 格式選型     | 如何根據場景選擇？ |

---

## 使用注意事項

* 大數據分析建議優先選用欄式格式（Parquet/ORC）
* Arrow 適合高效資料交換與即時查詢
* 壓縮策略需根據資料型態與查詢頻率調整

---

## 延伸閱讀與資源

* [Apache Parquet 官方文件](https://parquet.apache.org/documentation/latest/)
* [Apache Arrow 官方文件](https://arrow.apache.org/docs/)
* [ORC 格式介紹](https://orc.apache.org/docs/)
* [Zstandard 壓縮](https://facebook.github.io/zstd/)
* [Columnar vs Row Storage](https://www.vertica.com/blog/columnar-vs-row-database-storage/)

---

## 經典面試題與解法提示

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

## 結語

資料格式與儲存設計是高效數據平台的基礎。熟悉各種格式、欄式優勢與壓縮技巧，能讓你打造高效能、低成本的數據管線。下一章將進入資料湖、倉庫與湖倉架構，敬請期待！
