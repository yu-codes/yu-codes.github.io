---
title: "分類演算法百寶箱：k-NN、SVM、Naïve Bayes、決策樹全解析"
date: 2025-05-18 14:00:00 +0800
categories: [機器學習理論]
tags: [分類, k-NN, SVM, Naive Bayes, 決策樹, Kernel Trick]
---

# 分類演算法百寶箱：k-NN、SVM、Naïve Bayes、決策樹全解析

分類演算法是機器學習面試與實務的重點。從無參數的 k-NN，到強大的 SVM、直觀的 Naïve Bayes、靈活的決策樹，每種方法都有其數學基礎、優缺點與適用場景。本章將深入數學推導、直覺圖解、Python 實作、面試熱點與常見誤區，幫助你全面掌握分類演算法。

---

## k-NN（K-Nearest Neighbors）

### 原理與數學基礎

- 無需訓練，直接根據距離尋找最近的 k 個鄰居，投票決定類別。
- 距離度量常用歐氏距離、曼哈頓距離等。

### 優缺點

- 優點：簡單、無需訓練、可處理多分類。
- 缺點：資料量大時預測慢、對特徵縮放敏感、維度災難。

### Python 實作

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print("預測:", knn.predict(X[:5]))
```

---

## SVM（Support Vector Machine）

### 原理與數學推導

- 尋找最大化類別間隔的超平面。
- 支援硬邊界（無誤差）與軟邊界（允許部分誤差）。
- Kernel Trick 可將資料映射到高維空間，處理非線性分類。

### 常見 Kernel

- 線性、RBF（高斯）、多項式、Sigmoid

### 優缺點

- 優點：泛化能力強、可處理高維資料。
- 缺點：對參數敏感、資料量大時訓練慢、不易解釋。

### Python 實作

```python
from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1.0)
svc.fit(X, y)
print("SVM 預測:", svc.predict(X[:5]))
```

---

## Naïve Bayes 家族

### 原理

- 假設特徵間條件獨立，根據貝氏定理計算後驗機率。
- 常見：高斯、伯努利、多項式 Naïve Bayes。

### 優缺點

- 優點：訓練快、對高維資料友好、可處理缺失值。
- 缺點：特徵獨立假設常不成立、預測機率不精確。

### Python 實作

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X, y)
print("Naive Bayes 預測:", nb.predict(X[:5]))
```

---

## 決策樹（C4.5 / CART）

### 原理

- 依據資訊增益（C4.5）或基尼指數（CART）分裂特徵，構建樹狀結構。
- 可處理數值與類別特徵，支援多分類。

### 優缺點

- 優點：易解釋、可視化、處理異質特徵。
- 缺點：易過擬合、對資料微小變動敏感。

### Python 實作

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini')
dt.fit(X, y)
print("決策樹預測:", dt.predict(X[:5]))
```

---

## 常見面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| k-NN         | 為何不用訓練？如何選 k？ |
| SVM          | Kernel Trick 原理？硬/軟邊界差異？ |
| Naive Bayes  | 何時適用？獨立假設有何影響？ |
| 決策樹       | 如何避免過擬合？資訊增益與基尼指數差異？ |

---

## 實務應用與常見誤區

### 實務應用

- k-NN 適合小型資料集、推薦系統、異常偵測。
- SVM 適合高維資料、文本分類、影像辨識。
- Naive Bayes 適合垃圾郵件分類、醫療診斷。
- 決策樹適合特徵異質、需解釋性的任務。

### 常見誤區

- k-NN 忽略特徵標準化，導致距離失真。
- SVM Kernel 參數未調整，效果不佳。
- Naive Bayes 忽略特徵相關性，預測失準。
- 決策樹未剪枝，嚴重過擬合。

---

## 使用注意事項

* 分類演算法需根據資料特性選擇，並搭配特徵工程與正則化。
* k-NN、SVM 對特徵縮放敏感，建議標準化。
* 決策樹建議搭配剪枝與集成方法提升泛化能力。

---

## 延伸閱讀與資源

* [StatQuest: SVM, k-NN, Naive Bayes](https://www.youtube.com/c/joshstarmer)
* [Scikit-learn 分類演算法](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
* [Kernel Trick 直覺動畫](https://www.youtube.com/watch?v=3liCbRZPrZA)
* [決策樹與集成方法](https://scikit-learn.org/stable/modules/tree.html)

---

## 結語

分類演算法是機器學習的基礎。熟悉 k-NN、SVM、Naive Bayes、決策樹的原理、優缺點與實作細節，能讓你在面試與專案中靈活應用。下一章將進入集成學習，敬請期待！
