---
title: "經典迴歸模型全攻略：線性、Ridge、Lasso、Logistic 與資料處理"
date: 2025-05-18 13:00:00 +0800
categories: [機器學習理論]
tags: [線性迴歸, Ridge, Lasso, Logistic Regression, Softmax, 偏態資料, 重抽樣]
---

# 經典迴歸模型全攻略：線性、Ridge、Lasso、Logistic 與資料處理

迴歸模型是機器學習的入門與核心。從最基礎的線性迴歸，到正則化的 Ridge/Lasso、再到分類用的 Logistic Regression，這些模型不僅是面試常客，也是實務專案的基石。本章將深入數學推導、直覺圖解、Python 實作、資料處理技巧與面試熱點，幫助你全面掌握迴歸模型。

---

## 線性迴歸與多項式迴歸

### 線性迴歸（Linear Regression）

- 假設 $y = X\beta + \epsilon$，最小化殘差平方和。
- 封閉解：$\hat{\beta} = (X^TX)^{-1}X^Ty$
- 適用於連續型目標預測。

### 多項式迴歸（Polynomial Regression）

- 將特徵升維，擬合非線性關係。
- 容易過擬合，需正則化或交叉驗證。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.linspace(0, 1, 100)[:, None]
y = 2 * X.ravel() + 0.5 + np.random.randn(100) * 0.1

# 線性迴歸
lr = LinearRegression().fit(X, y)
plt.plot(X, lr.predict(X), label="Linear")

# 多項式迴歸
poly = PolynomialFeatures(4)
X_poly = poly.fit_transform(X)
lr_poly = LinearRegression().fit(X_poly, y)
plt.plot(X, lr_poly.predict(X_poly), label="Poly deg=4")
plt.scatter(X, y, s=10, color='black')
plt.legend(); plt.title("Linear vs Polynomial Regression"); plt.show()
```

---

## Ridge、Lasso、Elastic Net 正則化

### Ridge Regression（L2 正則化）

- 加入懲罰項 $\lambda \|\beta\|^2$，抑制參數過大。
- 適合多重共線性、特徵多的情境。

### Lasso Regression（L1 正則化）

- 懲罰項 $\lambda \|\beta\|_1$，可做特徵選擇（產生稀疏解）。
- 適合特徵選擇需求。

### Elastic Net

- 結合 L1 與 L2，兼顧特徵選擇與穩定性。

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
print("Ridge 係數:", ridge.coef_)
print("Lasso 係數:", lasso.coef_)
print("ElasticNet 係數:", enet.coef_)
```

---

## Logistic & Softmax Regression

### Logistic Regression（二元分類）

- 預測機率 $P(y=1|x) = \sigma(x^T\beta)$，$\sigma$ 為 sigmoid。
- 損失函數為交叉熵，常用於二元分類。

### Softmax Regression（多元分類）

- 將 sigmoid 擴展為 softmax，適用於多類別。
- 輸出每類別的機率，損失同為交叉熵。

```python
from sklearn.linear_model import LogisticRegression

X_cls = np.random.randn(200, 2)
y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
clf = LogisticRegression().fit(X_cls, y_cls)
print("預測機率:", clf.predict_proba(X_cls[:5]))
print("預測標籤:", clf.predict(X_cls[:5]))
```

---

## 偏態資料的對數轉換與重抽樣

### 偏態資料（Skewed Data）

- 常見於收入、房價等資料，右偏或左偏。
- 直接建模易受極端值影響。

### 對數轉換（Log Transform）

- 將偏態資料轉為近似常態，提升模型穩定性。
- 注意：資料需大於 0。

### 重抽樣（Resampling）

- 欠抽樣（Undersampling）：減少多數類樣本。
- 過抽樣（Oversampling）：增加少數類樣本（如 SMOTE）。

```python
import pandas as pd

data = pd.DataFrame({'income': np.random.exponential(50000, 1000)})
data['log_income'] = np.log1p(data['income'])
data['income'].hist(alpha=0.5, label='Original')
data['log_income'].hist(alpha=0.5, label='Log-Transformed')
plt.legend(); plt.title("Skewed Data Log Transform"); plt.show()
```

---

## 面試熱點與常見誤區

### 面試熱點

| 主題                | 常見問題 |
|---------------------|----------|
| Ridge/Lasso         | 何時選用？數學差異？ |
| Logistic Regression | 為何用交叉熵？如何推導？ |
| 多項式迴歸          | 如何避免過擬合？ |
| 偏態資料處理        | 何時用對數轉換？有何風險？ |
| 重抽樣              | 何時用 SMOTE？有何缺點？ |

### 常見誤區

- 忽略特徵標準化對正則化模型的影響。
- Lasso 係數全為 0 可能是 alpha 太大。
- Logistic Regression 不適合極端不平衡資料。
- 對數轉換後忘記逆轉換預測結果。

---

## 使用注意事項

* 正則化模型需標準化特徵，避免懲罰不公平。
* 多項式迴歸易過擬合，建議搭配交叉驗證。
* 處理偏態資料時，注意資料分布與業務意義。

---

## 延伸閱讀與資源

* [StatQuest: Ridge, Lasso, Elastic Net](https://www.youtube.com/watch?v=NGf0voTMlcs)
* [Scikit-learn Regression Models](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
* [Logistic Regression 推導](https://www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf)
* [Imbalanced-learn 官方文件](https://imbalanced-learn.org/stable/)

---

## 結語

經典迴歸模型是機器學習的基礎。熟悉線性、Ridge、Lasso、Logistic Regression 與資料處理技巧，能讓你在面試與實務中游刃有餘。下一章將進入分類演算法百寶箱，敬請期待！
