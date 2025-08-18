---
title: "安全・隱私・合規全攻略：PII、Tokenization、K-Anonymity、GDPR/CCPA、Row-Level Security"
date: 2025-05-21 23:30:00 +0800
categories: [Data Engineering]
tags: [安全, 隱私, 合規, PII, Tokenization, K-Anonymity, GDPR, CCPA, Row-Level Security, IAM]
---

# 安全・隱私・合規全攻略：PII、Tokenization、K-Anonymity、GDPR/CCPA、Row-Level Security

隨著數據應用於金融、醫療、電商等敏感領域，資料安全、隱私與合規成為數據工程不可忽視的議題。本章將深入 PII 分類、Tokenization、K-Anonymity、GDPR/CCPA 法規、Row-Level Security、IAM 最佳實踐，結合理論、實作、面試熱點與常見誤區，幫助你打造合規且安全的數據平台。

---

## PII 分類、Tokenization、K-Anonymity

### PII（Personally Identifiable Information）分類

- 直接識別：姓名、身分證號、電話、Email
- 間接識別：生日、地區、職稱、裝置 ID
- 分級管理：根據敏感度設置存取權限與加密

### Tokenization

- 將敏感資料以不可逆 token 替換，原始值僅限授權查詢
- 適合支付、醫療、金融等場景
- 支援格式保留（Format Preserving Tokenization）

### K-Anonymity

- 透過泛化、抑制等方法，確保每筆資料至少與 k-1 筆無法區分
- 防止重識別攻擊，常用於資料開放、共享

```python
# 簡易 K-Anonymity 實作
import pandas as pd

df = pd.DataFrame({'zip': ['12345', '12346', '12345', '12347'], 'age': [34, 35, 34, 36]})
df['zip'] = df['zip'].str[:3] + 'XX'  # 泛化郵遞區號
```

---

## GDPR / CCPA 對數據管線影響

### GDPR（歐盟一般資料保護規則）

- 強調資料主體權利、資料最小化、可刪除、可攜性、可解釋性
- 需記錄資料來源、獲取同意、支援刪除請求、審計追蹤

### CCPA（加州消費者隱私法）

- 強調消費者知情權、刪除權、拒絕銷售權
- 需提供資料存取、刪除、拒絕銷售等功能

### 合規實踐

- 設計資料流追蹤、敏感資料標註、刪除/匿名化流程
- 定期審查資料存取權限與合規性

---

## Row-Level Security & IAM 最佳實踐

### Row-Level Security（RLS）

- 根據用戶角色/屬性動態限制資料存取範圍
- 支援 SQL/BI 工具（如 Snowflake、BigQuery、Power BI）

```sql
-- Snowflake RLS Policy 範例
CREATE ROW ACCESS POLICY region_rls AS (region STRING) RETURNS BOOLEAN ->
  CURRENT_ROLE() = 'ADMIN' OR region = CURRENT_REGION();
```

### IAM（Identity and Access Management）

- 精細化權限控管，最小權限原則
- 定期審查、分層授權、審計日誌

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融、醫療、電商、政府、資料共享平台、法規遵循

### 常見誤區

- PII 未分類，敏感資料外洩風險高
- Tokenization/匿名化未覆蓋所有欄位
- GDPR/CCPA 僅形式合規，未落實刪除/審計
- RLS/IAM 權限設計過寬，導致越權存取

---

## 面試熱點與經典問題

| 主題         | 常見問題             |
| ------------ | -------------------- |
| PII 分類     | 如何分級管理？       |
| Tokenization | 原理與應用場景？     |
| K-Anonymity  | 如何實作與評估？     |
| GDPR/CCPA    | 對數據管線有何要求？ |
| RLS/IAM      | 如何設計與落地？     |

---

## 使用注意事項

* 敏感資料需全流程標註與加密
* Tokenization/匿名化建議結合審計追蹤
* RLS/IAM 權限需定期審查與最小化

---

## 延伸閱讀與資源

* [GDPR 官方文件](https://gdpr-info.eu/)
* [CCPA 官方文件](https://oag.ca.gov/privacy/ccpa)
* [K-Anonymity 論文](https://dataprivacylab.org/dataprivacy/projects/kanonymity/)
* [Snowflake Row-Level Security](https://docs.snowflake.com/en/user-guide/security-row-access-policies)
* [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

---

## 經典面試題與解法提示

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

## 結語

安全、隱私與合規是數據工程的底線。熟悉 PII、Tokenization、K-Anonymity、GDPR/CCPA、Row-Level Security 與 IAM，能讓你打造合規且安全的數據平台並在面試中展現專業素養。下一章將進入數據工程挑戰題庫，敬請期待！
