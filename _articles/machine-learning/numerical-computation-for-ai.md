---
title: "數值計算與穩定性：AI 實作必備的數值技巧與陷阱"
date: 2025-05-17 18:00:00 +0800
categories: [AI Math Foundation]
tags: [數值計算, 浮點誤差, Log-Sum-Exp, Gradient Clipping, 稀疏矩陣]
---

# 數值計算與穩定性：AI 實作必備的數值技巧與陷阱

數值計算是連接理論與實作的橋樑。再完美的數學模型，若忽略數值誤差與計算穩定性，實際運作時都可能出現爆炸或崩潰。本篇將深入探討 AI 常見的數值問題、穩定化技巧、稀疏運算加速，並結合理論、實作與面試重點，讓你在實戰中少踩坑。

---

## 浮點誤差、Underflow/Overflow

### 浮點數的本質與限制

- 電腦用有限位元表示實數，導致精度有限。
- 常見問題：加減極小數、極大數時精度損失。

### Underflow/Overflow

- **Underflow**：數值太小被當成 0。
- **Overflow**：數值太大超出表示範圍，變成無窮大或 NaN。

#### 實例：浮點誤差

```python
a = 1e16
b = 1.0
print("a + b - a =", (a + b) - a)  # 理論上應為 1.0，實際上可能為 0.0
```

### 常見陷阱

- 連加小數、極端指數運算、極小機率相乘（如序列模型）

---

## Log-Sum-Exp Trick：避免數值爆炸的黃金法則

### 問題來源

- Softmax、交叉熵等常用到 $\exp(x)$，大 x 易爆炸，小 x 易 underflow。

### 解法：Log-Sum-Exp

$$
\log \sum_i e^{x_i} = a + \log \sum_i e^{x_i - a}
$$

其中 $a = \max(x_i)$，可顯著提升數值穩定性。

```python
import numpy as np

def log_sum_exp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

x = np.array([1000, 1001, 1002])
print("不穩定計算:", np.log(np.sum(np.exp(x))))  # 會溢位
print("穩定計算:", log_sum_exp(x))
```

---

## Gradient Clipping：防止梯度爆炸

- 深度網路訓練時，梯度可能因連乘而爆炸，導致參數更新異常。
- **Gradient Clipping**：將梯度限制在某範圍內，常見於 RNN、深層網路。

```python
import torch

for param in model.parameters():
    torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
```

---

## 稀疏矩陣存儲與運算加速

### 稀疏矩陣的意義

- 多數元素為 0 的矩陣，常見於 NLP、推薦系統、圖神經網路。
- 若用一般矩陣存儲，浪費大量記憶體與運算資源。

### 稀疏矩陣格式

- CSR（Compressed Sparse Row）、CSC、COO 等格式。
- 支援高效的矩陣乘法、切片、轉置等操作。

```python
from scipy.sparse import csr_matrix

dense = np.zeros((1000, 1000))
dense[0, 1] = 3
sparse = csr_matrix(dense)
print("稀疏矩陣非零元素數量:", sparse.nnz)
```

### 稀疏運算加速

- PyTorch、TensorFlow 皆支援稀疏張量，適合大規模圖資料、推薦系統。

---

## 數值穩定性在 AI 實務的關鍵應用

- Softmax 輸出層、交叉熵損失、注意力機制（Transformer）、RNN 訓練
- 生成模型（如 VAE）中的對數機率計算
- 大型稀疏圖的鄰接矩陣運算

---

## 常見面試熱點整理

| 熱點主題          | 面試常問問題          |
| ----------------- | --------------------- |
| 浮點誤差          | 為何 a+b-b 不等於 a？ |
| Log-Sum-Exp       | 何時用？數學推導？    |
| Gradient Clipping | 什麼時候需要？        |
| 稀疏矩陣          | 如何存儲與運算？      |

---

## 使用注意事項

* 計算機精度有限，需主動檢查數值穩定性。
* 訓練深層網路時，建議預設啟用 Gradient Clipping。
* 稀疏矩陣適合高維低密度資料，能大幅節省資源。

---

## 延伸閱讀與資源

* [Scipy Sparse Matrix 官方文件](https://docs.scipy.org/doc/scipy/reference/sparse.html)
* [PyTorch 稀疏張量](https://pytorch.org/docs/stable/sparse.html)
* [Deep Learning Book: Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)

---

## 結語

數值計算與穩定性是 AI 實作不可忽視的細節。掌握浮點誤差、Log-Sum-Exp、Gradient Clipping 與稀疏矩陣運算，能讓你的模型更穩定、更高效，也能在面試與專案中展現專業素養。下一章將進入統計學在 ML 實務的應用，敬請期待！
