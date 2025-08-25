---
title: "損失函數百寶箱：回歸、分類、對比學習與自訂 Loss 全解析"
date: 2025-05-20 12:00:00 +0800
categories: [Machine Learning]
tags: [損失函數, MSE, MAE, Huber, Cross-Entropy, Focal Loss, Triplet, Contrastive, InfoNCE, 可導性, 穩定性]
---

# 損失函數百寶箱：回歸、分類、對比學習與自訂 Loss 全解析

損失函數（Loss Function）是模型訓練的核心，直接影響收斂速度、泛化能力與最終表現。本章將深入回歸、分類、對比學習常用損失，並介紹自訂 Loss 的可導性與穩定性設計，結合理論、實作、面試熱點與常見誤區，幫助你全面掌握損失函數設計。

---

## Regression：MSE／MAE／Huber

### 均方誤差（MSE, Mean Squared Error）

- $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
- 對離群值敏感，常用於回歸任務

### 平均絕對誤差（MAE, Mean Absolute Error）

- $MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$
- 對離群值不敏感，梯度不連續

### Huber Loss

- 結合 MSE 與 MAE，對小誤差用 MSE，大誤差用 MAE
- 更穩健於離群值

```python
import torch
import torch.nn as nn

mse = nn.MSELoss()
mae = nn.L1Loss()
huber = nn.HuberLoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.2, 1.9, 2.7])
print("MSE:", mse(y_pred, y_true).item())
print("MAE:", mae(y_pred, y_true).item())
print("Huber:", huber(y_pred, y_true).item())
```

---

## Classification：Cross-Entropy, Focal Loss, Label Smoothing

### Cross-Entropy Loss

- 多分類標配，衡量預測分布與真實分布的距離
- 適合 one-hot 或機率標籤

### Focal Loss

- 解決類別不平衡，對難分樣本加大懲罰
- 常用於目標檢測、醫療影像

### Label Smoothing

- 將 one-hot 標籤平滑，降低模型過度自信

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, 0.1]])
labels = torch.tensor([0])
ce = F.cross_entropy(logits, labels)
print("Cross-Entropy:", ce.item())
# Focal Loss 需自訂實作
```

---

## Triplet / Contrastive / InfoNCE

### Triplet Loss

- 使 anchor 與 positive 距離小於 anchor 與 negative
- 常用於人臉辨識、度量學習

### Contrastive Loss

- 拉近正對、推遠負對，適合自監督學習

### InfoNCE

- 自監督對比學習標配，最大化正對 mutual information

```python
# Triplet Loss (PyTorch)
triplet = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(5, 10)
positive = torch.randn(5, 10)
negative = torch.randn(5, 10)
print("Triplet Loss:", triplet(anchor, positive, negative).item())
```

---

## 自訂 Loss：梯度可導性與穩定性

- 自訂 Loss 需確保對參數可導，否則無法反向傳播
- 避免使用不可微函數（如 hard threshold）
- 建議用平滑近似（如 softmax, sigmoid）提升穩定性
- 注意數值穩定（如 log, exp 下溢/溢出）

```python
# 自訂 Loss 範例
def custom_loss(y_pred, y_true):
    diff = y_pred - y_true
    return torch.mean(torch.sqrt(diff ** 2 + 1e-6))
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- MSE/MAE/Huber：回歸、異常偵測
- Cross-Entropy/Focal：分類、目標檢測
- Triplet/Contrastive/InfoNCE：度量學習、自監督、檢索

### 常見誤區

- 分類任務誤用 MSE，導致收斂慢
- 自訂 Loss 未考慮可導性，反向傳播失敗
- Focal Loss 參數設置不當，模型難以收斂

---

## 面試熱點與經典問題

| 主題                | 常見問題                   |
| ------------------- | -------------------------- |
| MSE vs MAE          | 何時選用？對離群值敏感度？ |
| Huber Loss          | 為何更穩健？               |
| Cross-Entropy       | 數學推導與應用？           |
| Focal Loss          | 如何解決類別不平衡？       |
| Triplet/Contrastive | 適用場景與數學原理？       |
| 自訂 Loss           | 如何設計可導且穩定？       |

---

## 使用注意事項

* 損失函數選擇需根據任務與資料特性
* 自訂 Loss 建議先理論推導再實作
* 注意數值穩定與梯度可導性

---

## 延伸閱讀與資源

* [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
* [Focal Loss 論文](https://arxiv.org/abs/1708.02002)
* [InfoNCE 論文](https://arxiv.org/abs/1807.03748)
* [Triplet Loss 論文](https://arxiv.org/abs/1503.03832)

---

## 經典面試題與解法提示

1. MSE、MAE、Huber Loss 差異與適用場景？
2. Cross-Entropy Loss 數學推導？
3. Focal Loss 如何設計與調參？
4. Triplet/Contrastive/InfoNCE 適用場景？
5. 自訂 Loss 如何確保可導與穩定？
6. 分類任務誤用 MSE 有何後果？
7. 如何用 Python 實作自訂 Loss？
8. Focal Loss 參數設置原則？
9. InfoNCE 在自監督學習的作用？
10. Loss 數值不穩定時如何 debug？

---

## 結語

損失函數設計是模型訓練的基石。熟悉回歸、分類、對比學習與自訂 Loss，能讓你在各類任務與面試中展現專業素養。下一章將進入梯度下降家譜，敬請期待！
