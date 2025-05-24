---
title: "深度學習前菜：感知機、MLP 與激活函數演進全解析"
date: 2025-05-19 12:00:00 +0800
categories: [深度學習]
tags: [感知機, MLP, 激活函數, ReLU, GELU, 表達能力]
---

# 深度學習前菜：感知機、MLP 與激活函數演進全解析

深度學習的基礎始於感知機與多層感知機（MLP）。從單層感知機的線性可分，到多層神經網路的強大表達能力，再到激活函數的演進（Sigmoid、Tanh、ReLU、GELU），這些理論與實作是理解現代深度學習的起點。本章將深入數學推導、直覺圖解、Python 實作、面試熱點與常見誤區，幫助你全面掌握深度學習基礎。

---

## 感知機（Perceptron）與線性可分

### 感知機原理

- 最早的神經元模型，單層線性分類器。
- 輸出 $y = \text{sign}(w^T x + b)$，僅能解決線性可分問題。

### 線性可分與限制

- 僅能分割線性可分資料（如 AND），無法處理 XOR 問題。
- 多層感知機（MLP）可突破此限制。

```python
import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, n_iter=10):
        self.lr = lr
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                update = self.lr * (yi - self.predict(xi))
                self.w += update * xi
                self.b += update
    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([-1, -1, -1, 1])  # AND
model = Perceptron()
model.fit(X, y)
print("預測:", model.predict(X))
```

---

## 激活函數演進：Sigmoid / Tanh → ReLU / GELU

### Sigmoid 與 Tanh

- Sigmoid: $f(x) = \frac{1}{1 + e^{-x}}$，輸出範圍 (0,1)
- Tanh: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$，輸出範圍 (-1,1)
- 缺點：梯度消失，收斂慢

### ReLU（Rectified Linear Unit）

- $f(x) = \max(0, x)$，簡單高效，解決梯度消失
- 缺點：死神經元（Dead Neuron）

### GELU（Gaussian Error Linear Unit）

- $f(x) = x \cdot \Phi(x)$，$\Phi$ 為標準常態 CDF
- 近年 Transformer 等模型常用，平滑且表現佳

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 10)
print("Sigmoid:", torch.sigmoid(x))
print("Tanh:", torch.tanh(x))
print("ReLU:", F.relu(x))
print("GELU:", F.gelu(x))
```

---

## 參數量、層數與表達能力

### MLP（多層感知機）結構

- 多層線性層 + 非線性激活函數
- 理論上單隱藏層即可逼近任意連續函數（Universal Approximation Theorem）

### 參數量與層數

- 參數量 = 每層輸入數 × 輸出數 + 偏置
- 層數增加 → 表達能力提升，但訓練更難，易過擬合

### Python 實作：簡單 MLP

```python
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
print(mlp)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 感知機：早期二分類、線性問題
- MLP：表格資料、特徵工程後的分類/迴歸
- 激活函數：ReLU/GELU 幾乎為深度學習標配

### 常見誤區

- 誤以為 MLP 必能解決所有問題（資料需可分）
- 忽略激活函數選擇對收斂與表現的影響
- 層數過多未正則化，導致過擬合

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| 感知機       | 為何只能解線性可分？ |
| 激活函數     | ReLU 為何優於 Sigmoid？GELU 有何優勢？ |
| MLP          | 為何能逼近任意函數？ |
| 參數量       | 如何計算？層數與表達能力關係？ |

---

## 使用注意事項

* 激活函數選擇會影響梯度傳遞與收斂速度
* MLP 易過擬合，建議搭配正則化與早停
* 感知機僅適合線性問題，複雜資料需多層網路

---

## 延伸閱讀與資源

* [Deep Learning Book: Perceptron & MLP](https://www.deeplearningbook.org/contents/mlp.html)
* [PyTorch Activation Functions](https://pytorch.org/docs/stable/nn.functional.html)
* [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

## 經典面試題與解法提示

1. 感知機的學習規則與收斂條件？
2. 為何感知機無法解 XOR 問題？
3. Sigmoid、Tanh、ReLU、GELU 優缺點比較？
4. MLP 為何能逼近任意連續函數？
5. 如何計算 MLP 參數量？
6. 激活函數選擇對訓練有何影響？
7. 如何用 Python 實作簡單感知機/MLP？
8. MLP 過擬合時有哪些解法？
9. ReLU 死神經元問題如何緩解？
10. GELU 為何在 Transformer 中表現佳？

---

## 結語

感知機與 MLP 是深度學習的起點。熟悉激活函數演進、參數量與表達能力，能讓你在後續 CNN、RNN、Transformer 等進階主題中打下堅實基礎。下一章將進入卷積網路精要，敬請期待！
