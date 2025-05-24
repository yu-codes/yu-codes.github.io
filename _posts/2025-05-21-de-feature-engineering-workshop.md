---
title: "特徵工程工坊：時序擴充、類別編碼、Feature Store 與自動化實戰"
date: 2025-05-21 20:00:00 +0800
categories: [數據工程]
tags: [特徵工程, 時序擴充, Lag, Rolling, Category Encoding, Feature Store, Feast, Tecton, SageMaker FS]
---

# 特徵工程工坊：時序擴充、類別編碼、Feature Store 與自動化實戰

特徵工程是 ML 成敗的關鍵，數據工程師需設計高效、可重用、可追溯的特徵管線。本章將深入時序資料擴充（Lag, Rolling, Expanding）、類別編碼（Target, Hash, Leave-one-out）、Feature Store（Feast, SageMaker FS, Tecton）設計與自動化，結合理論、實作、面試熱點與常見誤區，幫助你打造高質量特徵平台。

---

## 時間序列擴充：Lag, Rolling, Expanding

### Lag 特徵

- 取前 n 期數值，捕捉時序依賴
- 適合預測、異常偵測、用戶行為分析

### Rolling/Moving Window

- 計算移動平均、最大、最小等統計量
- 平滑波動、捕捉趨勢

### Expanding

- 累積統計（如累積平均），反映長期趨勢

```python
import pandas as pd

df = pd.DataFrame({'ts': pd.date_range('2023-01-01', periods=100), 'val': range(100)})
df['lag_1'] = df['val'].shift(1)
df['rolling_mean_7'] = df['val'].rolling(7).mean()
df['expanding_sum'] = df['val'].expanding().sum()
```

---

## Category Encoding：Target / Hash / Leave-one-out

### Target Encoding

- 用目標變數的平均值取代類別，提升模型表現
- 需防止資料洩漏，建議用交叉驗證計算

### Hash Encoding

- 將類別經 hash 映射到固定維度，適合高基數欄位
- 節省記憶體，防止過擬合

### Leave-one-out Encoding

- 每筆資料用去除自身的目標平均，進一步防止洩漏

```python
import category_encoders as ce

encoder = ce.TargetEncoder()
df['cat_te'] = encoder.fit_transform(df['cat'], df['target'])
encoder = ce.HashingEncoder(n_components=8)
df_hash = encoder.fit_transform(df['cat'])
```

---

## Feature Store：Feast, SageMaker FS、Tecton

### Feature Store 作用

- 集中管理特徵計算、儲存、版本控管、線上/離線一致性
- 支援特徵重用、追溯、即時查詢

### 主流工具

- Feast：開源，支援多雲、線上/離線同步
- SageMaker Feature Store：AWS 雲端原生，整合 SageMaker
- Tecton：商用，支援複雜特徵管線、監控、治理

### Feature Store 實戰設計

1. 定義特徵規格（schema、計算邏輯、更新頻率）
2. 建立特徵管線（ETL、驗證、監控）
3. 管理特徵版本與追溯
4. 線上/離線特徵一致性驗證
5. 提供 API 給模型訓練與推論

---

## 自動化特徵工程與治理

- 結合 Orchestrator（Airflow、Dagster）自動化特徵計算
- 設計資料驗證（Great Expectations、Deequ）確保特徵質量
- 特徵 lineage 追蹤與審計

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融風控、推薦系統、用戶行為分析、IoT 預測、即時特徵查詢

### 常見誤區

- Lag/Rolling 特徵未排序，導致資料穿越
- Target Encoding 未防洩漏，模型過擬合
- Feature Store 未設計版本控管，特徵不可追溯
- 線上/離線特徵不一致，推論失敗

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Lag/Rolling  | 如何設計？有何風險？ |
| Target/Hash/Leave-one-out | 差異與適用場景？ |
| Feature Store| 作用與設計要點？ |
| 特徵自動化   | 如何確保質量與追溯？ |
| 線上/離線一致性 | 如何驗證？ |

---

## 使用注意事項

* 時序特徵需嚴格排序與分割，防止未來資訊洩漏
* Target/Leave-one-out Encoding 建議用交叉驗證
* Feature Store 設計需兼顧效能、治理與安全

---

## 延伸閱讀與資源

* [Feast 官方文件](https://docs.feast.dev/)
* [SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
* [Tecton 官方文件](https://docs.tecton.ai/)
* [Category Encoders](https://contrib.scikit-learn.org/category_encoders/)
* [Feature Store Survey](https://arxiv.org/abs/2209.08350)

---

## 經典面試題與解法提示

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

## 結語

特徵工程工坊是 ML 成敗的關鍵。熟悉時序擴充、類別編碼、Feature Store 與自動化治理，能讓你打造高質量、可追溯的特徵平台。下一章將進入資料品質與治理，敬請期待！
