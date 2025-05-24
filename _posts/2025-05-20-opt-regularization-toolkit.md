---
title: "正則化武器庫：L1/L2、Dropout、Early Stopping、Label Smoothing 全解析"
date: 2025-05-20 15:00:00 +0800
categories: [模型訓練與優化]
tags: [正則化, L1, L2, Elastic Net, Dropout, Early Stopping, Label Smoothing, Confidence Penalty]
---

# 正則化武器庫：L1/L2、Dropout、Early Stopping、Label Smoothing 全解析

正則化（Regularization）是防止模型過擬合、提升泛化能力的關鍵武器。從 L1/L2/Elastic Net，到 Dropout、DropPath、Stochastic Depth、Early Stopping、Label Smoothing、Confidence Penalty，本章將深入原理、實作、面試熱點與常見誤區，幫助你打造更穩健的模型。

---

## L1 / L2 / Elastic Net

### L1 正則化（Lasso）

- 懲罰參數絕對值和，促使部分權重為 0，具備特徵選擇效果

### L2 正則化（Ridge）

- 懲罰參數平方和，抑制權重過大，提升模型穩定性

### Elastic Net

- 結合 L1 與 L2，兼顧特徵選擇與穩定性

```python
import torch
import torch.nn as nn

l1_lambda = 0.01
l2_lambda = 0.01
l1_norm = sum(p.abs().sum() for p in model.parameters())
l2_norm = sum(p.pow(2).sum() for p in model.parameters())
loss = loss_fn(output, target) + l1_lambda * l1_norm + l2_lambda * l2_norm
```

---

## Dropout / DropPath / Stochastic Depth

### Dropout

- 訓練時隨機丟棄部分神經元，防止 co-adaptation，提升泛化

### DropPath / Stochastic Depth

- 隨機丟棄整個層或路徑，常用於深層網路（如 ResNet、ViT）

```python
drop = nn.Dropout(p=0.5)
x = torch.randn(10, 20)
print("Dropout 輸出:", drop(x))
```

---

## Early Stopping 判斷點

- 監控驗證集 loss，若多輪未提升則提前停止訓練
- 防止過擬合，節省資源

```python
best_loss = float('inf')
patience, counter = 5, 0
for epoch in range(epochs):
    # ...existing code...
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
```

---

## Label Smoothing & Confidence Penalty

### Label Smoothing

- 將 one-hot 標籤平滑，降低模型過度自信，提升泛化

### Confidence Penalty

- 在損失中加入預測分布熵，懲罰過於集中的預測

```python
import torch.nn.functional as F

labels = torch.tensor([0, 1, 2])
n_classes = 3
smooth = 0.1
one_hot = F.one_hot(labels, n_classes).float()
smoothed = one_hot * (1 - smooth) + smooth / n_classes
print("Label Smoothing:", smoothed)

# Confidence Penalty
logits = torch.randn(4, 10)
probs = F.softmax(logits, dim=1)
entropy = - (probs * probs.log()).sum(dim=1).mean()
loss = loss_fn(logits, targets) - 0.1 * entropy
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- L1/L2/Elastic Net：回歸、分類、特徵選擇
- Dropout/DropPath：深度網路、過擬合防治
- Early Stopping：所有需泛化的任務
- Label Smoothing/Confidence Penalty：分類、生成模型

### 常見誤區

- L1/L2 未正確設置權重，導致欠擬合
- Dropout 只在訓練時啟用，推論時需關閉
- Early Stopping 監控訓練集而非驗證集
- Label Smoothing 過度平滑，降低辨識力

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| L1 vs L2     | 差異與適用場景？ |
| Dropout      | 原理與推論差異？ |
| Early Stopping | 如何設計判斷點？ |
| Label Smoothing | 何時用？有何效果？ |
| Confidence Penalty | 如何提升泛化？ |

---

## 使用注意事項

* 正則化強度需根據資料與模型調整
* Dropout/DropPath 建議搭配正規化層
* Early Stopping 需監控驗證集 loss
* Label Smoothing/Confidence Penalty 適度使用

---

## 延伸閱讀與資源

* [Dropout 論文](https://arxiv.org/abs/1207.0580)
* [Early Stopping 論文](https://www.jmlr.org/papers/volume15/prechelt14a/prechelt14a.pdf)
* [Label Smoothing 論文](https://arxiv.org/abs/1512.00567)
* [Elastic Net 論文](https://www.jmlr.org/papers/volume5/zhang04a/zhang04a.pdf)

---

## 經典面試題與解法提示

1. L1/L2/Elastic Net 數學推導與適用場景？
2. Dropout/DropPath 原理與實作？
3. Early Stopping 如何設計與調參？
4. Label Smoothing/Confidence Penalty 差異？
5. 正則化過強/過弱會有什麼後果？
6. 如何用 Python 實作 Early Stopping？
7. Dropout 推論時如何處理？
8. Elastic Net 何時優於單一正則化？
9. Label Smoothing 對模型有何影響？
10. Confidence Penalty 如何提升泛化？

---

## 結語

正則化是模型泛化的基石。熟悉 L1/L2、Dropout、Early Stopping、Label Smoothing 等技巧，能讓你打造更穩健的模型並在面試中脫穎而出。下一章將進入參數初始化與正規化層，敬請期待！