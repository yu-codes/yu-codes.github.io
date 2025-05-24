---
title: "最適化基石：AI 訓練的數學與演算法核心"
date: 2025-05-17 14:00:00 +0800
categories: [AI 數學基礎]
tags: [最適化, 凸集, Lagrange, SGD, Adam, Learning Rate]
---

# 最適化基石：AI 訓練的數學與演算法核心

最適化是機器學習與深度學習模型訓練的靈魂。從損失函數的設計，到參數的更新策略，背後都依賴數學上的最適化理論與演算法。本篇將帶你掌握 AI 常用的最適化觀念，並以直覺、圖解與 Python 範例說明。

---

## 凸集、凸函數與 Lagrange 乘子

### 凸集（Convex Set）

- 若集合內任兩點的連線也都在集合內，則為凸集。
- 在最適化中，凸集保證只有一個全域最小值，避免陷入局部極小。

### 凸函數（Convex Function）

- 任意兩點連線上的函數值不高於端點連線。
- 損失函數若為凸函數，訓練更穩定、易於收斂。

### Lagrange 乘子法

- 處理有約束條件的最適化問題。
- 常見於 SVM、正則化等模型。

```python
import numpy as np
from scipy.optimize import minimize

# 無約束最小化
def f(x):
    return (x - 2) ** 2 + 1

res = minimize(f, x0=0)
print("最小值 x =", res.x, "f(x) =", res.fun)
```

---

## Batch / Mini-Batch / SGD 家族

- **Batch Gradient Descent**：每次用全部資料計算梯度，收斂穩定但慢。
- **Mini-Batch Gradient Descent**：每次用部分資料，兼顧效率與穩定。
- **Stochastic Gradient Descent (SGD)**：每次只用一筆資料，更新頻繁但波動大。

| 方法         | 優點           | 缺點           | 適用場景         |
|--------------|----------------|----------------|------------------|
| Batch        | 收斂平滑       | 記憶體需求高   | 小型資料集       |
| Mini-Batch   | 效率與穩定兼顧 | 需調 batch size| 主流深度學習訓練 |
| SGD          | 即時更新快     | 收斂不穩定     | 線上學習         |

---

## Learning Rate Schedules & Warm-up

- **Learning Rate（學習率）**：控制每次參數更新的步伐，過大易震盪，過小則收斂慢。
- **Learning Rate Schedule**：隨訓練進度調整學習率（如 Step Decay、Cosine Annealing）。
- **Warm-up**：訓練初期用較小學習率，避免一開始震盪過大。

```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(20):
    # ...訓練步驟...
    scheduler.step()
    print("Epoch", epoch, "Learning Rate:", scheduler.get_last_lr())
```

---

## Momentum, Adam, RMSProp：何時選誰？

- **Momentum**：累積過去梯度，幫助跳出局部極小。
- **RMSProp**：自動調整每個參數的學習率，適合非平穩目標。
- **Adam**：結合 Momentum 與 RMSProp，現今最常用的優化器之一。

| Optimizer | 特性             | 適用情境         |
|-----------|------------------|------------------|
| SGD       | 基本款，易理解   | 小型/凸問題      |
| Momentum  | 跳脫局部極小     | 深層網路         |
| RMSProp   | 適應性學習率     | 非平穩目標       |
| Adam      | 綜合型，穩定收斂 | 主流深度學習訓練 |

---

## 常見面試熱點整理

| 熱點主題         | 面試常問問題 |
|------------------|-------------|
| 凸函數           | 為何凸函數好優化？ |
| Lagrange 乘子    | 什麼時候用？怎麼推導？ |
| SGD/Adam         | 兩者差異與選擇時機？ |
| Learning Rate    | 如何調整學習率？ |

---

## 使用注意事項

* 學習率是最重要的超參數之一，建議多做實驗與調整。
* Adam 雖穩定，但有時 SGD + Momentum 反而泛化較佳。
* 了解最適化理論有助於 debug 訓練異常與提升模型表現。

---

## 延伸閱讀與資源

* [Stanford CS231n: Optimization](http://cs231n.stanford.edu/slides/2023/cs231n_2023_lecture6.pdf)
* [Adam 論文](https://arxiv.org/abs/1412.6980)
* [PyTorch Optimizer 官方文件](https://pytorch.org/docs/stable/optim.html)

---

## 結語

最適化理論與演算法是 AI 訓練的核心。熟悉各種優化器、學習率策略與數學基礎，能讓你在模型訓練與調參時更加得心應手。下一章將進入機率論，敬請期待！
