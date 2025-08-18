---
title: "機器學習核心概念暖身：監督/非監督、Bias-Variance、泛化誤差全解析"
date: 2025-05-18 12:00:00 +0800
categories: [Machine Learning]
tags: [監督學習, 非監督學習, 泛化誤差, Bias-Variance, Overfitting, Underfitting]
---

# 機器學習核心概念暖身：監督/非監督、Bias-Variance、泛化誤差全解析

機器學習的世界博大精深，但所有進階理論與實作都建立在核心概念之上。本章將帶你從監督/非監督/半監督/強化式學習的本質出發，深入理解 Bias-Variance 葛藤、泛化誤差分解，以及過擬合/欠擬合的診斷與調整流程。內容涵蓋理論、圖解、Python 實作、面試熱點與常見誤區，為後續章節打下堅實基礎。

---

## 監督、非監督、半監督、強化式學習差異

### 監督學習（Supervised Learning）

- 有標註資料（輸入+目標），學習輸入到目標的映射。
- 常見任務：分類、迴歸。
- 例：房價預測、圖片分類。

### 非監督學習（Unsupervised Learning）

- 無標註資料，探索資料內部結構。
- 常見任務：聚類、降維、密度估計。
- 例：客戶分群、主成分分析（PCA）。

### 半監督學習（Semi-supervised Learning）

- 標註資料稀少，結合大量未標註資料提升表現。
- 例：少量標註圖片+大量未標註圖片訓練分類器。

### 強化式學習（Reinforcement Learning）

- 透過與環境互動獲取回饋（獎勵/懲罰），學習最佳策略。
- 例：AlphaGo、機器人控制。

| 類型       | 標註需求 | 典型任務  | 代表演算法             |
| ---------- | -------- | --------- | ---------------------- |
| 監督學習   | 高       | 分類/迴歸 | SVM, LR, RF, NN        |
| 非監督學習 | 無       | 聚類/降維 | K-means, PCA, GMM      |
| 半監督學習 | 低       | 分類/迴歸 | Pseudo-label, MixMatch |
| 強化學習   | 無       | 控制/決策 | Q-Learning, DQN, PG    |

---

## Bias-Variance 葛藤與泛化誤差分解

### 泛化誤差（Generalization Error）

- 模型在未見過資料上的預測誤差。
- 分解為三部分：偏差（Bias）、變異（Variance）、噪聲（Noise）。

$$
\text{Generalization Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

### 偏差（Bias）

- 模型假設與真實分布的差距。
- 偏差高 → 欠擬合（Underfitting）。

### 變異（Variance）

- 模型對訓練資料的敏感度。
- 變異高 → 過擬合（Overfitting）。

### 欠擬合與過擬合診斷

- 欠擬合：訓練/驗證誤差都高，模型太簡單。
- 過擬合：訓練誤差低、驗證誤差高，模型太複雜。

#### 圖解

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.linspace(0, 1, 100)[:, None]
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(100) * 0.1

for degree in [1, 4, 15]:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    plt.plot(X, y_pred, label=f"degree={degree}")
plt.scatter(X, y, s=10, color='black')
plt.legend(); plt.title("Bias-Variance Trade-off"); plt.show()
```

---

## Underfitting/Overfitting 診斷與調整流程

### 診斷流程

1. **觀察訓練/驗證誤差**：兩者都高→欠擬合；訓練低驗證高→過擬合。
2. **學習曲線**：隨資料量變化誤差趨勢。
3. **模型複雜度調整**：增加/減少參數、特徵、正則化。

### 調整技巧

- 欠擬合：增加模型複雜度、特徵工程、減少正則化。
- 過擬合：加強正則化、減少模型複雜度、資料增強、早停（Early Stopping）。

#### Python 實作：學習曲線

```python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeRegressor(), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
plt.xlabel("Training Size"); plt.ylabel("Score"); plt.legend()
plt.title("Learning Curve"); plt.show()
```

---

## 實務應用與常見誤區

### 實務應用

- 模型選擇與調參時，必須同時考慮 Bias-Variance。
- 監督/非監督學習的選擇取決於資料標註情況與任務目標。
- 強化學習適合決策與控制問題，需設計合適的獎勵機制。

### 常見誤區

- 只看訓練誤差，不檢查泛化能力。
- 誤以為模型越複雜越好，忽略過擬合風險。
- 混淆監督與非監督學習的適用場景。

---

## 常見面試熱點整理

| 熱點主題        | 面試常問問題                 |
| --------------- | ---------------------------- |
| 監督/非監督差異 | 何時用哪種？有何代表演算法？ |
| Bias-Variance   | 如何可視化？如何調整？       |
| 泛化誤差        | 如何分解？如何降低？         |
| Overfitting     | 有哪些診斷與解法？           |

---

## 使用注意事項

* 訓練/驗證/測試集需嚴格分離，避免資料洩漏。
* 學習曲線是診斷模型表現的重要工具。
* 調參時建議結合交叉驗證與多指標評估。

---

## 延伸閱讀與資源

* [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
* [Deep Learning Book: Generalization](https://www.deeplearningbook.org/contents/gener.html)
* [Scikit-learn: Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

---

## 結語

核心概念是機器學習進階學習的基石。熟悉監督/非監督/半監督/強化學習、Bias-Variance 葛藤與泛化誤差分解，能讓你在模型設計、調參與面試中展現專業素養。下一章將進入經典迴歸模型，敬請期待！
