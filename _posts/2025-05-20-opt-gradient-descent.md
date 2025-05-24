---
title: "梯度下降家譜：SGD、Momentum、Adam、Adaptive 優化器全解析"
date: 2025-05-20 13:00:00 +0800
categories: [模型訓練與優化]
tags: [梯度下降, SGD, Momentum, Adam, RMSProp, Nesterov, AdaGrad, AdamW, 收斂, 泛化]
---

# 梯度下降家譜：SGD、Momentum、Adam、Adaptive 優化器全解析

梯度下降法是深度學習訓練的核心。從 Batch/Mini-Batch/SGD，到 Momentum、Nesterov、AdaGrad、RMSProp、Adam、AdamW、NAdam、AdamP，這些優化器直接影響收斂速度與泛化能力。本章將深入數學原理、直覺圖解、Python 實作、面試熱點與常見誤區，幫助你全面掌握梯度下降與優化器選擇。

---

## Batch／Mini-Batch／Stochastic GD

- **Batch GD**：每次用全部資料計算梯度，收斂平滑但慢，需大記憶體
- **Mini-Batch GD**：每次用部分資料，兼顧效率與穩定，主流選擇
- **Stochastic GD (SGD)**：每次只用一筆資料，更新頻繁但波動大

```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for x, y in dataloader:  # Mini-Batch
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## Momentum, Nesterov, AdaGrad, RMSProp

### Momentum

- 累積過去梯度，幫助跳出局部極小
- 更新規則：$v_{t+1} = \beta v_t + (1-\beta)\nabla L$，$w_{t+1} = w_t - \eta v_{t+1}$

### Nesterov Momentum

- 先預測一步再計算梯度，收斂更快

### AdaGrad

- 為每個參數自適應調整學習率，適合稀疏資料

### RMSProp

- 解決 AdaGrad 學習率過快衰減，適合非平穩目標

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.01)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
```

---

## Adam / AdamW／NAdam／AdamP

### Adam

- 結合 Momentum 與 RMSProp，適應性強，主流選擇
- $m_t$（一階動量）、$v_t$（二階動量），帶偏差校正

### AdamW

- 將權重衰減與梯度分離，泛化更佳

### NAdam / AdamP

- NAdam：結合 Nesterov 動量
- AdamP：針對 CNN 設計，提升泛化

```python
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001)
```

---

## SGD vs. Adaptive：收斂速度與泛化

- Adaptive（如 Adam）收斂快，適合複雜/稀疏資料
- SGD + Momentum 泛化能力常優於 Adam，適合大規模訓練
- 實務常先用 Adam，收斂後切換 SGD 微調

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- SGD：大規模影像、語音、NLP
- Adam/AdamW：NLP、稀疏特徵、預訓練模型
- AdaGrad/RMSProp：稀疏資料、非平穩目標

### 常見誤區

- Adam 泛化不佳，未調整學習率 schedule
- Nesterov 需正確設置 momentum
- Adaptive 優化器未設 weight decay，易過擬合

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| SGD vs Adam  | 收斂速度與泛化差異？ |
| Momentum     | 如何幫助跳出局部極小？ |
| AdamW        | 為何較 Adam 泛化好？ |
| AdaGrad/RMSProp | 適用場景與數學原理？ |
| Nesterov     | 與 Momentum 差異？ |

---

## 使用注意事項

* 優化器選擇需根據任務與資料特性
* Adaptive 優化器建議搭配學習率衰減與 weight decay
* 訓練後期可考慮切換 SGD 微調

---

## 延伸閱讀與資源

* [PyTorch Optimizer 官方文件](https://pytorch.org/docs/stable/optim.html)
* [Adam 論文](https://arxiv.org/abs/1412.6980)
* [AdamW 論文](https://arxiv.org/abs/1711.05101)
* [RMSProp 論文](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

---

## 經典面試題與解法提示

1. SGD、Momentum、Nesterov、Adam、AdamW 更新規則？
2. Adam 為何收斂快但泛化差？
3. AdamW 與 Adam 的數學差異？
4. AdaGrad/RMSProp 適用場景？
5. Adaptive 優化器 weight decay 設置？
6. SGD 何時優於 Adam？
7. 如何用 Python 切換優化器？
8. Nesterov Momentum 的數學推導？
9. AdamP 有何創新？
10. 優化器選擇對訓練有何影響？

---

## 結語

梯度下降與優化器是模型訓練的核心。熟悉 SGD、Momentum、Adam、AdamW 等優化器原理與實作，能讓你在訓練與面試中展現專業素養。下一章將進入學習率策略，敬請期待！
