---
title: "線性代數快攻：AI 必備的向量與矩陣基礎"
date: 2025-05-17 12:00:00 +0800
categories: [Machine Learning]
tags: [線性代數, 向量, 矩陣, 特徵值, SVD, PCA]
---

# 線性代數快攻：AI 必備的向量與矩陣基礎

在 AI 與機器學習領域，線性代數是不可或缺的基石。無論是神經網路的權重運算、資料降維，還是推薦系統的矩陣分解，背後都離不開向量與矩陣的操作。本篇將帶你快速掌握 AI 常用的線性代數觀念，並以直覺、圖解與 Python 範例說明。

---

## 向量與矩陣運算

### 什麼是向量與矩陣？

- **向量（Vector）**：一組有方向與大小的數值序列，常用於描述資料點、特徵等。
- **矩陣（Matrix）**：由多個向量組成的二維陣列，常見於資料集、權重參數等。

| 名稱 | 例子                                                    | 應用場景           |
| ---- | ------------------------------------------------------- | ------------------ |
| 向量 | $\mathbf{v} = [2, 3, 5]$                                | 特徵向量、嵌入表示 |
| 矩陣 | $\mathbf{A} = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}$ | 影像、權重、資料集 |

### 基本運算

- **加法/減法**：同型向量或矩陣逐元素相加減。
- **內積（Dot Product）**：$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$，常用於相似度計算。
- **矩陣乘法**：$\mathbf{A} \mathbf{B}$，資料轉換、神經網路前向傳播核心。

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)  # 內積

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matmul = np.matmul(A, B)  # 矩陣乘法

print("向量內積：", dot)
print("矩陣乘法：\n", matmul)
```

---

## Rank、Span、Basis：資料的維度與本質

- **Rank（秩）**：矩陣中獨立行（或列）的最大數量，反映資料的「本質維度」。
- **Span（張成空間）**：一組向量可線性組合出所有可能的空間。
- **Basis（基底）**：能張成空間且彼此線性獨立的最小向量組。

| 概念  | 直覺說明       | 應用               |
| ----- | -------------- | ------------------ |
| Rank  | 有幾個獨立方向 | 資料降維、PCA      |
| Span  | 能到達哪些點   | 特徵空間理解       |
| Basis | 最小生成組合   | 向量壓縮、特徵選擇 |

---

## 特徵值、特徵向量、SVD 與 PCA

### 特徵值與特徵向量

- **特徵向量（Eigenvector）**：經過矩陣變換後，方向不變的向量。
- **特徵值（Eigenvalue）**：該向量被拉伸或縮放的倍數。

> 在資料降維、PCA、圖神經網路等場景都會用到。

### SVD（奇異值分解）

- 將任意矩陣拆解為三個矩陣的乘積：$\mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^T$
- 用於資料壓縮、推薦系統、PCA 等。

### PCA（主成分分析）

- 一種常用的降維方法，找出資料中最重要的方向（主成分）。
- 本質上就是對協方差矩陣做特徵分解或 SVD。

```python
from sklearn.decomposition import PCA

X = np.random.rand(100, 5)  # 100 筆 5 維資料
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("降維後資料 shape：", X_reduced.shape)
print("主成分解釋變異量：", pca.explained_variance_ratio_)
```

---

## Kronecker／Hadamard 乘積 & 廣播機制

- **Kronecker 乘積**：兩矩陣的「擴展型」乘法，常用於張量運算。
- **Hadamard 乘積**：對應元素相乘，常見於神經網路的門控機制。
- **廣播（Broadcasting）**：Numpy/PyTorch 等自動擴展維度，簡化運算。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

# Hadamard 乘積
hadamard = A * B

# Kronecker 乘積
kronecker = np.kron(A, B)

print("Hadamard 乘積：\n", hadamard)
print("Kronecker 乘積：\n", kronecker)
```

---

## 常見面試熱點整理

| 熱點主題        | 面試常問問題                  |
| --------------- | ----------------------------- |
| Rank/Basis      | 如何判斷資料可否降維？        |
| 特徵值/特徵向量 | 為何 PCA 要用特徵分解？       |
| SVD             | 與 Eigen Decomposition 差異？ |
| Hadamard 乘積   | 在神經網路哪裡會用到？        |
| 廣播機制        | Numpy 如何自動對齊維度？      |

---

## 使用注意事項

* 向量與矩陣運算常見於資料前處理、特徵工程與模型訓練。
* 高維資料常需降維（如 PCA），以提升效率與可視化。
* 熟悉 Numpy、Pandas、Scikit-learn 等工具可大幅簡化實作。

---

## 延伸閱讀與資源

* [線性代數（台大李宏毅課程）](https://www.youtube.com/watch?v=QK_Hv6pG4nE)
* [3Blue1Brown：線性代數動畫](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
* [Scikit-learn PCA 文件](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---

## 結語

線性代數是 AI 與資料科學的語言。掌握向量、矩陣、特徵值與降維技巧，不僅能幫助你理解模型背後的數學邏輯，也能在面試與實務專案中脫穎而出。未來章節將深入探討微積分、機率統計等 AI 必備數學，敬請期待！