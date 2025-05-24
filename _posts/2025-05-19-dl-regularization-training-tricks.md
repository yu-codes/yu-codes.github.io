---
title: "正規化與訓練技巧全攻略：BatchNorm、Dropout、Label Smoothing、MixUp、CutMix"
date: 2025-05-19 20:00:00 +0800
categories: [深度學習]
tags: [正規化, BatchNorm, LayerNorm, Dropout, Stochastic Depth, Label Smoothing, MixUp, CutMix]
---

# 正規化與訓練技巧全攻略：BatchNorm、Dropout、Label Smoothing、MixUp、CutMix

深度學習模型的訓練穩定性與泛化能力，離不開正規化與各種訓練技巧。從 BatchNorm、LayerNorm、GroupNorm、RMSNorm，到 Dropout、Stochastic Depth、Label Smoothing、MixUp、CutMix，這些方法是現代神經網路不可或缺的組件。本章將深入原理、實作、應用場景、面試熱點與常見誤區，幫助你打造更穩健的深度模型。

---

## BatchNorm / LayerNorm / GroupNorm / RMSNorm 場景

### Batch Normalization

- 對每個 mini-batch 的特徵做標準化，提升收斂速度與穩定性
- 常用於 CNN、MLP

### Layer Normalization

- 對每個樣本的所有特徵做標準化，適合 RNN、Transformer

### Group Normalization

- 將特徵分組，各組分別標準化，適合小 batch size

### RMSNorm

- 只用均方根（RMS）做標準化，計算更簡單

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(8)
ln = nn.LayerNorm([16, 16])
gn = nn.GroupNorm(4, 8)
x = torch.randn(4, 8, 16, 16)
print("BatchNorm:", bn(x).shape)
print("LayerNorm:", ln(x).shape)
print("GroupNorm:", gn(x).shape)
```

---

## Residual Connection 優點：梯度路徑、資訊再利用

- 殘差連接（Residual Connection）讓梯度可直接傳遞，緩解梯度消失
- 幫助資訊再利用，提升深層網路訓練穩定性
- ResNet、Transformer 等架構標配

---

## Dropout / Stochastic Depth / DropPath

### Dropout

- 訓練時隨機丟棄部分神經元，防止 co-adaptation，提升泛化

### Stochastic Depth / DropPath

- 隨機丟棄整個層或路徑，常用於深層網路（如 ResNet、Vision Transformer）

```python
drop = nn.Dropout(p=0.5)
x = torch.randn(10, 20)
print("Dropout 輸出:", drop(x))
```

---

## Label Smoothing、MixUp、CutMix

### Label Smoothing

- 將 one-hot 標籤平滑處理，降低模型過度自信，提升泛化

### MixUp

- 隨機線性混合兩筆資料與標籤，提升魯棒性與泛化

### CutMix

- 隨機將一張圖像的區塊貼到另一張，標籤按比例混合

```python
import torch.nn.functional as F

# Label Smoothing
labels = torch.tensor([0, 1, 2])
n_classes = 3
smooth = 0.1
one_hot = F.one_hot(labels, n_classes).float()
smoothed = one_hot * (1 - smooth) + smooth / n_classes
print("Label Smoothing:", smoothed)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- BatchNorm：CNN、MLP
- LayerNorm：RNN、Transformer
- Dropout：全連接層、深度網路
- MixUp/CutMix：影像分類、資料增強

### 常見誤區

- BatchNorm 用於 RNN 效果不佳，應用 LayerNorm
- Dropout 只在訓練時啟用，推論時需關閉
- MixUp/CutMix 過度使用會損失語意

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| BatchNorm    | 原理與優缺點？ |
| Residual     | 如何幫助梯度傳遞？ |
| Dropout      | 原理與推論差異？ |
| MixUp/CutMix | 有何優勢與限制？ |
| Label Smoothing | 何時用？有何效果？ |

---

## 使用注意事項

* 正規化方法需根據模型與資料選擇
* Dropout、MixUp、CutMix 需適度使用，避免過度擾動
* 殘差連接建議搭配正規化提升穩定性

---

## 延伸閱讀與資源

* [BatchNorm 論文](https://arxiv.org/abs/1502.03167)
* [LayerNorm 論文](https://arxiv.org/abs/1607.06450)
* [Stochastic Depth 論文](https://arxiv.org/abs/1603.09382)
* [MixUp 論文](https://arxiv.org/abs/1710.09412)
* [CutMix 論文](https://arxiv.org/abs/1905.04899)

---

## 經典面試題與解法提示

1. BatchNorm、LayerNorm、GroupNorm、RMSNorm 差異？
2. 殘差連接如何幫助深層網路訓練？
3. Dropout 的數學原理與推論差異？
4. Stochastic Depth/DropPath 適用場景？
5. Label Smoothing 有何優缺點？
6. MixUp/CutMix 如何提升泛化？
7. 如何用 Python 實作正規化與資料增強？
8. BatchNorm 在推論時如何運作？
9. Dropout/MixUp 過度使用會有什麼問題？
10. 正規化與訓練技巧如何組合應用？

---

## 結語

正規化與訓練技巧是深度學習模型穩定與泛化的關鍵。熟悉 BatchNorm、Dropout、Label Smoothing、MixUp、CutMix 等方法，能讓你打造更強大、更可靠的深度模型。下一章將進入加速與壓縮實戰，敬請期待！
