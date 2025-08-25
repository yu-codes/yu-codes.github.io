---
title: "信息理論 & 損失函數：AI 必備的熵、交叉熵與機率距離全解析"
date: 2025-05-17 17:00:00 +0800
categories: [Machine Learning]
tags: [信息理論, 熵, 交叉熵, KL Divergence, JS Divergence, 損失函數]
---

# 信息理論 & 損失函數：AI 必備的熵、交叉熵與機率距離全解析

信息理論（Information Theory）是現代機器學習、深度學習與資料壓縮的理論基礎。從分類模型的損失函數，到生成模型的機率距離，熵、交叉熵、KL 散度等概念無處不在。本篇將深入剖析這些核心數學工具，並結合直覺、推導、應用與 Python 實作，讓你徹底掌握 AI 必備的信息理論基礎。

---

## 熵（Entropy）：不確定性的度量

### 熵的定義與直覺

- **熵（Entropy）** 衡量隨機變數不確定性的指標，單位為 bit（以 2 為底）或 nat（以 e 為底）。
- 熵越大，表示資訊越分散、越難預測；熵越小，表示資訊越集中、越容易預測。

$$
H(X) = -\sum_{i} P(x_i) \log P(x_i)
$$

### 熵的應用場景

- 決策樹分裂（資訊增益）
- 資料壓縮（霍夫曼編碼）
- 模型不確定性評估

```python
import numpy as np

def entropy(p):
    p = np.array(p)
    p = p[p > 0]  # 避免 log(0)
    return -np.sum(p * np.log2(p))

print("均勻分布熵:", entropy([0.5, 0.5]))
print("偏態分布熵:", entropy([0.9, 0.1]))
```

---

## 互資訊（Mutual Information）：變數間的資訊共享

- **互資訊（MI）** 衡量兩個隨機變數間共享的資訊量。
- MI 越大，表示兩變數關聯越強。

$$
I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}
$$

### 應用場景

- 特徵選擇（選出與標籤最相關的特徵）
- 表徵學習（資訊瓶頸理論）

---

## 交叉熵（Cross-Entropy）：機率分布間的距離

### 交叉熵的定義

- 衡量「真實分布」與「預測分布」間的距離，是分類任務最常用的損失函數。

$$
H(P, Q) = -\sum_{i} P(x_i) \log Q(x_i)
$$

- $P$ 為真實分布，$Q$ 為模型預測分布。

### 交叉熵的直覺

- 預測越接近真實分布，交叉熵越小。
- 若模型預測與真實分布完全一致，交叉熵等於熵。

### Python 實作

```python
def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    q = np.clip(q, 1e-12, 1.0)  # 避免 log(0)
    return -np.sum(p * np.log(q))

p = [1, 0, 0]
q = [0.7, 0.2, 0.1]
print("交叉熵:", cross_entropy(p, q))
```

---

## KL 散度（Kullback-Leibler Divergence）：分布間的非對稱距離

### KL 散度的定義

- 衡量一個分布 Q 假裝成分布 P 時，平均多花多少「資訊量」。
- 非對稱：$KL(P||Q) \neq KL(Q||P)$

$$
KL(P||Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
$$

### KL 散度的應用

- 生成模型（如 VAE）
- 損失函數（如知識蒸餾）
- 機率分布近似

### Python 實作

```python
def kl_divergence(p, q):
    p, q = np.array(p), np.array(q)
    p, q = p[p > 0], q[p > 0]
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)))

print("KL 散度:", kl_divergence([0.8, 0.2], [0.6, 0.4]))
```

---

## JS 散度（Jensen-Shannon Divergence）：對稱的機率距離

- JS 散度是 KL 散度的對稱化版本，常用於生成模型評估（如 GAN）。
- $$
JS(P||Q) = \frac{1}{2} KL(P||M) + \frac{1}{2} KL(Q||M), \quad M = \frac{P+Q}{2}
$$

---

## Softmax + Cross-Entropy：為何好用？

### Softmax 函數

- 將任意實數向量轉換為機率分布，常用於多分類模型的輸出層。

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 結合交叉熵的優點

- 數學上導數簡潔，便於反向傳播。
- 能有效懲罰錯誤預測，提升模型收斂速度。

```python
def softmax(z):
    z = np.array(z)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
print("Softmax 機率:", probs)
print("交叉熵損失:", cross_entropy([1, 0, 0], probs))
```

---

## 理論推導與直覺圖解

- 熵、交叉熵、KL 散度皆可視為「平均資訊量」的不同度量。
- 交叉熵 = 熵 + KL 散度，反映模型預測與真實分布的差異。
- JS 散度則能衡量兩分布的「對稱距離」，適合生成模型評估。

---

## 應用場景與常見誤區

### 應用場景

- 分類模型訓練（交叉熵損失）
- 生成模型（KL/JS 散度）
- 決策樹（資訊增益）
- 知識蒸餾（KL 散度）

### 常見誤區

- KL 散度非對稱，不能當作一般距離使用。
- Softmax 輸出過於極端時，易導致梯度消失或數值不穩。
- 交叉熵損失需搭配 one-hot 標籤或機率分布。

---

## 常見面試熱點整理

| 熱點主題         | 面試常問問題           |
| ---------------- | ---------------------- |
| 熵/交叉熵        | 兩者差異與應用？       |
| KL/JS Divergence | 何時用？有何數學性質？ |
| Softmax + CE     | 為何組合效果好？       |
| 機率距離         | 如何選擇適合的損失？   |

---

## 使用注意事項

* 熵與交叉熵常用於分類與資訊理論相關任務。
* KL/JS 散度適合評估分布間差異，但需注意數值穩定性。
* Softmax + Cross-Entropy 是多分類的黃金組合，但需正確處理標籤格式。

---

## 延伸閱讀與資源

* [Deep Learning Book: Information Theory](https://www.deeplearningbook.org/contents/information-theory.html)
* [StatQuest: Cross Entropy, KL Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
* [Scipy.stats 熵與距離文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)

---

## 結語

信息理論為 AI 提供了衡量不確定性與分布差異的數學工具。掌握熵、交叉熵、KL/JS 散度與 Softmax，不僅能讓你設計更有效的損失函數，也能在生成模型、分類任務與面試中展現深厚的理論素養。下一章將進入數值計算與穩定性，敬請期待！
