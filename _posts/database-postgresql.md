---
title: "PostgreSQL 深入指南：從入門到面試實戰"
date: 2025-05-10 17:30:00 +0800
categories: [Database]
tags: [PostgreSQL, SQL]
---

# PostgreSQL 深入指南：從入門到面試實戰

PostgreSQL 是一套功能完整、穩定且開源的關聯式資料庫管理系統（RDBMS），廣泛應用於各種企業級應用。它不僅支援標準 SQL，還具備強大的擴展性、資料一致性與先進功能，如 JSON 處理、Window Functions、全文搜尋、GIS 等。

本篇文章將從 PostgreSQL 的架構與特性講起，帶你快速掌握實作技巧、資料庫管理指令，並涵蓋面試中常被提問的進階知識。

---

## 🔧 什麼是 PostgreSQL？

> PostgreSQL 是一套 **ACID 保證** 的開源資料庫，支援複雜查詢、事務處理、多版本並發控制（MVCC）與資料一致性機制。

- 開發語言：C
- 開源協議：PostgreSQL License
- 支援平台：Linux、macOS、Windows
- 適合場景：企業應用、金融系統、地理資訊（GIS）、資料科學

---

## 📦 PostgreSQL 的核心特色

| 功能                   | 說明                             |
| ---------------------- | -------------------------------- |
| MVCC                   | 支援高併發不鎖表，避免讀寫阻塞   |
| JSON / JSONB 支援      | 可做為半結構化 NoSQL 使用        |
| CTE / Window Functions | 複雜查詢與分析支援               |
| Extensibility          | 支援自定義資料型別、函式與操作符 |
| Full Text Search       | 內建全文搜尋引擎                 |
| GIS（PostGIS）擴充     | 適合地理座標查詢與空間分析       |

---

## 🚀 安裝 PostgreSQL（以 macOS 為例）

```bash
brew install postgresql
brew services start postgresql
psql postgres
```

或使用 Docker：

```bash
docker run --name pg -e POSTGRES_PASSWORD=pass -p 5432:5432 -d postgres
```

---

## 🧑‍💻 基本語法與實作

### 建立資料庫與使用者

```sql
CREATE DATABASE company;
CREATE USER analyst WITH PASSWORD 'strongpass';
GRANT ALL PRIVILEGES ON DATABASE company TO analyst;
```

### 建立資料表與插入資料

```sql
CREATE TABLE employees (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  department TEXT,
  salary NUMERIC CHECK (salary >= 0)
);

INSERT INTO employees (name, department, salary)
VALUES ('Alice', 'Engineering', 90000);
```

### 查詢資料

```sql
SELECT name, salary FROM employees WHERE salary > 80000;
```

---

## 🧱 JSON 與半結構化資料查詢

```sql
CREATE TABLE logs (
  id SERIAL PRIMARY KEY,
  payload JSONB
);

-- 查詢含有某欄位的紀錄
SELECT * FROM logs WHERE payload ? 'user';

-- 查詢嵌套值
SELECT payload->>'user' AS username FROM logs;
```

---

## 📊 分析查詢與性能優化

### EXPLAIN 與 ANALYZE

```sql
EXPLAIN ANALYZE SELECT * FROM employees WHERE salary > 100000;
```

* `Seq Scan` 表示掃描整張表
* `Index Scan` 表示使用索引
* 可依結果判斷是否需要建 index

---

## ⚙️ Index 與效能實務

```sql
CREATE INDEX idx_salary ON employees(salary);

-- 多欄索引
CREATE INDEX idx_dept_salary ON employees(department, salary);
```

**面試常問：**

* 索引對 `LIKE '%abc'` 無效，為什麼？
* 多欄索引建立順序對查詢是否有影響？

---

## 🧠 PostgreSQL 面試問題精選（附解說）

### 1. PostgreSQL 是如何實作 MVCC 的？

> PostgreSQL 透過 **tuple versioning** 機制（每筆資料有 xmin/xmax 欄位），實現多版本共存。舊版本不會立即被覆蓋，而是由 `VACUUM` 回收。

### 2. 你如何解釋 "PostgreSQL 事務隔離等級"？

* READ COMMITTED（預設）
* REPEATABLE READ
* SERIALIZABLE

> PostgreSQL 的事務隔離等級由 `SET TRANSACTION ISOLATION LEVEL` 控制，支援標準 SQL 四級隔離模型。

### 3. 如何備份與還原資料庫？

```bash
pg_dump company > backup.sql
psql -U postgres -d company < backup.sql
```

### 4. PostgreSQL 如何處理索引膨脹（index bloat）？

> 定期使用 `REINDEX` 或 `VACUUM FULL` 清理碎片。

---

## 🔒 權限與角色管理

```sql
CREATE ROLE readonly;
GRANT CONNECT ON DATABASE company TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
```

---

## 🔍 延伸功能：CTE、Window Functions、全文搜尋

### CTE（Common Table Expression）

```sql
WITH high_salary AS (
  SELECT * FROM employees WHERE salary > 100000
)
SELECT department, COUNT(*) FROM high_salary GROUP BY department;
```

### Window Function

```sql
SELECT name, department, salary,
  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
```

### Full-Text Search

```sql
SELECT * FROM articles
WHERE to_tsvector('english', content) @@ to_tsquery('data & science');
```
---

## 🆚 PostgreSQL vs 其他主流資料庫比較

在選擇資料庫時，了解不同系統的設計理念、功能強項與使用場景至關重要。以下針對 PostgreSQL、MySQL、MongoDB 做出實務面比較：

| 特性 / 系統       | **PostgreSQL**                                   | **MySQL**                                       | **MongoDB**                                     |
| ----------------- | ------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------- |
| 資料模型          | 關聯式（支援 JSON/NoSQL 功能）                   | 關聯式                                          | 文件導向（BSON 格式）                           |
| ACID 支援         | ✅ 完整支援（預設強一致性）                       | ✅ 也支援，但預設隔離等級較低（REPEATABLE READ） | 🚫 預設為最終一致性，可調整為強一致              |
| JSON/半結構化支援 | ✅ 強大（JSONB 有索引與運算支援）                 | ⚠️ 支援 JSON 欄位但查詢能力較弱                  | ✅ 原生設計，為主要資料格式                      |
| 索引能力          | ✅ 全面（B-Tree、GIN、GiST、BRIN、Partial Index） | ✅ 常見索引如 B-Tree、全文搜尋                   | ✅ 原生索引支援（複合欄位、Array Index）         |
| 查詢語言          | 標準 SQL + 擴展語法                              | 標準 SQL                                        | Mongo Query Language（與 SQL 概念不同）         |
| 擴充性 / 自定義   | ✅ 強（支援自定義型別、函式、操作符）             | ⚠️ 有限                                          | ⚠️ 擴展性弱於 PostgreSQL                         |
| 交易處理          | ✅ 強（支援多表與 SAVEPOINT）                     | ✅ 可用，但 XA Transaction 較複雜                | ⚠️ 內建交易支援自 4.0 版起開始提供（單集合交易） |
| 使用場景          | 金融系統、地理資訊、分析查詢                     | 網站後端、CMS、小型應用                         | 快速開發、原型設計、大量非結構化資料            |
| 社群與支援        | 強（開源社群活躍 + 企業採用廣泛）                | 非常大（歷史悠久、網路資源豐富）                | 活躍，但仍以 NoSQL 社群為主                     |


### ✍️ 實務建議

* 若你需要 **關聯資料一致性 + JSON 處理能力**，**PostgreSQL 是最強選擇**。
* 若專案開發人力有限、偏向網站類 CRUD 應用，可考慮使用 **MySQL/MariaDB**。
* 若資料結構變動大、原型開發頻繁，**MongoDB** 提供最彈性文件存儲模式。

---

### 💬 面試常見延伸問題

1. **你為什麼選 PostgreSQL 而不是 MySQL？**

   > PostgreSQL 支援更完整的 SQL 標準，支援 JSONB 與多種索引結構，對於需要混合結構化與半結構化資料的系統更為適合。

2. **MongoDB 與 PostgreSQL 哪個適合做全文搜尋？**

   > MongoDB 內建全文搜尋語法簡單，但 PostgreSQL 的 `tsvector` 加上 `GIN Index` 提供更靈活的搜尋語法與效率微調能力，且可和 SQL 聚合結合使用。

3. **MySQL 的主從同步與 PostgreSQL 有何不同？**

   > PostgreSQL 支援 streaming replication，且自 10 版起支援 logical replication（可做行級複製），在彈性與一致性控制上更進一步。

---

## 📚 推薦資源

* [PostgreSQL 官方文件](https://www.postgresql.org/docs/)
* 書籍：《PostgreSQL: Up and Running》、《Mastering PostgreSQL in Application Development》
* 練習平台：LeetCode SQL、DB Fiddle、pgExercises

---

## 結語

PostgreSQL 是目前最被企業與開源社群信賴的資料庫系統之一。具備高併發、彈性擴展與功能深度，不僅適合開發與部署，更是面試中經常被考察的主題。本文涵蓋的內容將幫助你在實作與面談時都能應對如流。
