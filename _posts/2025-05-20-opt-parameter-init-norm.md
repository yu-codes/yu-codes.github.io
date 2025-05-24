---
title: "參數初始化與正規化層：Xavier, He, BatchNorm, LayerNorm, ScaleNorm 全解析"
date: 2025-05-20 16:00:00 +0800
categories: [模型訓練與優化]
tags: [參數初始化, Xavier, He, LeCun, BatchNorm, LayerNorm, GroupNorm, RMSNorm, Weight Standardization, ScaleNorm]
---

# 參數初始化與正規化層：Xavier, He, BatchNorm, LayerNorm, ScaleNorm 全解析

參數初始化與正規化層是深度學習模型穩定訓練與高效收斂的基礎。從 Xavier、He、LeCun 初始化，到 BatchNorm、LayerNorm、GroupNorm、RMSNorm、Weight Standardization、ScaleNorm，本章將深入原理、實作、應用場景、面試熱點與常見誤區，幫助你打造穩健的訓練流程。

---

## Xavier, He, LeCun Init 直覺

### Xavier/Glorot Initialization

- 適合 Sigmoid/Tanh，保持前後層方差一致
- $W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$

### He Initialization

- 適合 ReLU，方差更大，避免梯度消失
- $W \sim N(0, \frac{2}{n_{in}})$

### LeCun Initialization

- 適合 SELU，$W \sim N(0, \frac{1}{n_{in}})$

```python
import torch.nn as nn

layer = nn.Linear(128, 64)
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

## Batch / Layer / Group / RMSNorm 比較

### BatchNorm

- 對 mini-batch 特徵做標準化，提升收斂與穩定性
- 常用於 CNN、MLP

### LayerNorm

- 對每個樣本所有特徵標準化，適合 RNN、Transformer

### GroupNorm

- 將特徵分組，各組分別標準化，適合小 batch size

### RMSNorm

- 只用均方根（RMS）做標準化，計算更簡單

```python
bn = nn.BatchNorm2d(32)
ln = nn.LayerNorm([64])
gn = nn.GroupNorm(8, 32)
# RMSNorm 可用第三方實現
```

---

## Weight Standardization & ScaleNorm

### Weight Standardization

- 對卷積/線性層權重做標準化，提升訓練穩定性
- 常與 GroupNorm 搭配

### ScaleNorm

- 用單一縮放參數標準化特徵，計算簡單，適合 Transformer

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- Xavier/He/LeCun：根據激活函數選擇初始化
- BatchNorm/LayerNorm：提升深層網路收斂與穩定性
- GroupNorm/RMSNorm/ScaleNorm：小 batch、Transformer、Vision

### 常見誤區

- 初始化未根據激活函數選擇，導致梯度消失/爆炸
- BatchNorm 用於 RNN 效果不佳，應用 LayerNorm
- GroupNorm 分組數設置不當，效果反而變差
- Weight Standardization/ScaleNorm 未正確實作

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Xavier vs He | 差異與適用場景？ |
| BatchNorm    | 原理與優缺點？ |
| LayerNorm    | 與 BatchNorm 差異？ |
| GroupNorm    | 適用場景與設置？ |
| Weight Standardization | 如何提升穩定性？ |

---

## 使用注意事項

* 初始化方法需根據激活函數與網路結構選擇
* 正規化層建議搭配深層網路與小 batch 訓練
* Weight Standardization/ScaleNorm 需測試兼容性

---

## 延伸閱讀與資源

* [Xavier/He Initialization 論文](https://proceedings.mlr.press/v9/glorot10a.html)
* [He Initialization 論文](https://arxiv.org/abs/1502.01852)
* [BatchNorm 論文](https://arxiv.org/abs/1502.03167)
* [LayerNorm 論文](https://arxiv.org/abs/1607.06450)
* [GroupNorm 論文](https://arxiv.org/abs/1803.08494)
* [Weight Standardization 論文](https://arxiv.org/abs/1903.10520)
* [ScaleNorm 論文](https://arxiv.org/abs/1910.05895)

---

## 經典面試題與解法提示

1. Xavier/He/LeCun 初始化數學推導？
2. BatchNorm/LayerNorm/GroupNorm/RMSNorm 差異？
3. Weight Standardization 原理與應用？
4. ScaleNorm 適用場景？
5. 初始化錯誤對訓練有何影響？
6. 如何用 Python 實作多種初始化？
7. GroupNorm 分組數如何選擇？
8. BatchNorm 在推論時如何運作？
9. LayerNorm 適合哪些模型？
10. ScaleNorm/Weight Standardization 有何優勢？

---

## 結語

參數初始化與正規化層是穩健訓練的基石。熟悉 Xavier、He、BatchNorm、LayerNorm、ScaleNorm 等技巧，能讓你打造高效穩定的深度學習模型。下一章將進入數值穩定技巧，敬請期待！
