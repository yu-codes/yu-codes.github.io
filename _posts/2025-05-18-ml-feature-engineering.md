---
title: "特徵工程與選擇全攻略：編碼、標準化、特徵選擇三大法門"
date: 2025-05-18 17:00:00 +0800
categories: [Machine Learning]
tags: [特徵工程, One-Hot, 標準化, 特徵選擇, Encoding, Feature Selection]
---

# 特徵工程與選擇全攻略：編碼、標準化、特徵選擇三大法門

特徵工程是機器學習成敗的關鍵。從資料前處理、編碼、標準化，到特徵選擇，這些步驟直接影響模型表現與泛化能力。本章將深入 One-Hot/Target/Frequency Encoding、標準化/正規化/Whitening、三大特徵選擇法（Filter/Wrapper/Embedded），結合理論、實作、面試熱點與常見誤區，讓你打造高質量特徵集。

---

## 編碼技巧：One-Hot、Target、Frequency Encoding

### One-Hot Encoding

- 將類別特徵轉為 0/1 向量，適合無序類別。
- 缺點：高基數時維度爆炸。

```python
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue']})
onehot = pd.get_dummies(df['color'])
print(onehot)
```

### Target Encoding

- 用目標變數的平均值取代類別，適合有序類別或高基數特徵。
- 需防止資料洩漏，建議用交叉驗證計算。

```python
df['target'] = [1, 0, 1, 0]
mean_map = df.groupby('color')['target'].mean()
df['color_te'] = df['color'].map(mean_map)
print(df[['color', 'color_te']])
```

### Frequency Encoding

- 用類別出現頻率取代原值，適合高基數特徵。
- 不引入目標資訊，無洩漏風險。

```python
freq_map = df['color'].value_counts() / len(df)
df['color_fe'] = df['color'].map(freq_map)
print(df[['color', 'color_fe']])
```

---

## 標準化 vs. 正規化 vs. Whitening

### 標準化（Standardization）

- 轉換為均值 0、標準差 1，適合大多數 ML 演算法。
- 常用於 SVM、Logistic Regression、神經網路。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform([[1, 2], [3, 4], [5, 6]])
print(X_scaled)
```

### 正規化（Normalization）

- 將特徵縮放到固定範圍（如 0~1），適合距離度量敏感的演算法（如 k-NN）。
- 常用 MinMaxScaler。

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_norm = scaler.fit_transform([[1, 2], [3, 4], [5, 6]])
print(X_norm)
```

### Whitening

- 進一步去除特徵間相關性，使協方差矩陣為單位矩陣。
- 常用於 PCA 前處理、深度學習 BatchNorm。

---

## 特徵選擇三大法門

### Filter 方法

- 根據統計量（如相關係數、卡方、互資訊）篩選特徵。
- 優點：計算快、無需模型。
- 缺點：忽略特徵間交互作用。

```python
from sklearn.feature_selection import SelectKBest, f_classif

X = [[1,2,3],[4,5,6],[7,8,9],[1,3,5]]
y = [0,1,0,1]
skb = SelectKBest(f_classif, k=2).fit(X, y)
print("選中特徵索引:", skb.get_support(indices=True))
```

### Wrapper 方法

- 用模型評估特徵子集（如遞迴特徵消除 RFE）。
- 優點：考慮特徵交互，效果好。
- 缺點：計算量大。

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(), n_features_to_select=2)
rfe.fit(X, y)
print("RFE 選中特徵:", rfe.support_)
```

### Embedded 方法

- 特徵選擇與模型訓練同時進行（如 Lasso、樹模型）。
- 優點：效率高，常為預設選擇。

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1).fit(X, y)
print("Lasso 係數:", lasso.coef_)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- One-Hot：低基數類別、無序特徵
- Target/Frequency Encoding：高基數類別、樞紐分析
- 標準化/正規化：距離敏感模型、梯度下降
- 特徵選擇：降維、提升泛化、減少過擬合

### 常見誤區

- One-Hot 用於高基數，導致維度爆炸。
- Target Encoding 未用交叉驗證，導致資料洩漏。
- 標準化與正規化混用，影響模型收斂。
- Wrapper 方法過度耗時，未做特徵預篩。

---

## 面試熱點與經典問題

| 主題                    | 常見問題             |
| ----------------------- | -------------------- |
| One-Hot                 | 何時不用？有何缺點？ |
| Target Encoding         | 如何防止資料洩漏？   |
| 標準化/正規化           | 差異與適用場景？     |
| Filter/Wrapper/Embedded | 各自優缺點？         |
| Lasso                   | 為何能做特徵選擇？   |

---

## 使用注意事項

* 特徵工程需結合業務知識與資料探索。
* 編碼與標準化順序需一致，避免資料洩漏。
* 特徵選擇建議多法並用，交叉驗證效果。

---

## 延伸閱讀與資源

* [StatQuest: Feature Engineering](https://www.youtube.com/c/joshstarmer)
* [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
* [Kaggle: Feature Engineering 教程](https://www.kaggle.com/learn/feature-engineering)

---

## 經典面試題與解法提示

1. One-Hot Encoding 有哪些缺點？如何解決？
2. Target Encoding 如何防止資料洩漏？
3. 標準化與正規化差異？
4. Filter/Wrapper/Embedded 方法比較？
5. Lasso 為何能做特徵選擇？
6. 特徵選擇對模型有何影響？
7. 如何用 Python 實作特徵選擇？
8. 特徵工程常見陷阱有哪些？
9. 如何評估特徵工程效果？
10. 實務上如何設計特徵工程流程？

---

## 結語

特徵工程與選擇是 ML 成敗關鍵。熟悉各種編碼、標準化與特徵選擇方法，能讓你打造高效能模型並在面試中脫穎而出。下一章將進入模型評估與驗證，敬請期待！
