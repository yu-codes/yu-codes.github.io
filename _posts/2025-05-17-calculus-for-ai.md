---
title: "微積分與連鎖法則：AI 必備的微分直覺與應用"
date: 2025-05-17 13:00:00 +0800
categories: [AI 數學基礎]
tags: [微積分, 導數, 偏導, 連鎖法則, Jacobian, 泰勒展開]
---

# 微積分與連鎖法則：AI 必備的微分直覺與應用

微積分是機器學習與深度學習的數學核心。從模型訓練的梯度下降，到神經網路的反向傳播，背後都離不開導數、偏導與連鎖法則。本篇將帶你掌握 AI 常用的微積分觀念，並以直覺、圖解與 Python 範例說明。

---

## 極限、導數、偏導與梯度

### 極限（Limit）

- 描述函數在某點附近的趨勢，是導數與連續性的基礎。
- 訓練過程中，損失函數收斂本質上就是極限的應用。

### 導數（Derivative）

- 表示函數變化率，記作 $f'(x)$ 或 $\frac{df}{dx}$。
- 在機器學習中，導數用於描述損失函數對參數的敏感度。

### 偏導數（Partial Derivative）

- 多變數函數對單一變數的導數，記作 $\frac{\partial f}{\partial x}$。
- 神經網路每個權重的梯度即為偏導數。

### 梯度（Gradient）

- 所有偏導數組成的向量，指向函數上升最快的方向。
- 在優化中，梯度用於指引參數更新方向。

```python
import numpy as np

# 單變數導數
def f(x):
    return x**2 + 3*x + 2

x = 1.0
h = 1e-5
df = (f(x + h) - f(x)) / h
print("f'(1) ≈", df)

# 多變數偏導與梯度
def g(x, y):
    return x**2 + y**2

x, y = 1.0, 2.0
df_dx = (g(x + h, y) - g(x, y)) / h
df_dy = (g(x, y + h) - g(x, y)) / h
print("∂g/∂x ≈", df_dx)
print("∂g/∂y ≈", df_dy)
```

---

## Jacobian 與高維導數

- **Jacobian 矩陣**：多輸入多輸出函數的偏導數矩陣，常見於神經網路的層間傳遞。
- 在自動微分與反向傳播中，Jacobian 是計算梯度的基礎。

| 概念     | 直覺說明           | 應用                |
|----------|--------------------|---------------------|
| 導數     | 單變數變化率       | 線性回歸、損失函數  |
| 偏導     | 多變數單方向變化率 | 神經網路權重更新    |
| 梯度     | 所有偏導組成向量   | 最佳化、梯度下降    |
| Jacobian | 多輸入多輸出偏導   | 反向傳播、複合模型  |

---

## 連鎖法則在反向傳播的角色

- **連鎖法則（Chain Rule）**：複合函數的導數計算法則，核心公式：
  $$
  \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
  $$
- 在神經網路反向傳播（Backpropagation）中，連鎖法則用於將損失對輸出層的梯度逐層傳回輸入層。

```python
# 連鎖法則範例：z = f(y), y = g(x)
def g(x):
    return 2 * x

def f(y):
    return y ** 3

x = 1.5
y = g(x)
z = f(y)

# 手動計算
dz_dy = 3 * y ** 2
dy_dx = 2
dz_dx = dz_dy * dy_dx
print("dz/dx =", dz_dx)
```

---

## 泰勒展開與損失曲面直覺

- **泰勒展開（Taylor Expansion）**：用多項式近似函數，理解損失曲面形狀與優化步伐。
- 在優化中，泰勒展開幫助我們預測損失變化，設計更有效的學習率與步長。

> 例如，二階泰勒展開可用於牛頓法（Newton's Method）等高級優化演算法。

---

## 常見面試熱點整理

| 熱點主題         | 面試常問問題 |
|------------------|-------------|
| 導數/偏導        | 如何手算梯度？ |
| 連鎖法則         | 反向傳播如何應用連鎖法則？ |
| Jacobian         | 什麼時候需要 Jacobian？ |
| 泰勒展開         | 為何優化時要考慮二階導數？ |

---

## 使用注意事項

* 導數與梯度計算是所有深度學習框架（如 PyTorch、TensorFlow）的底層核心。
* 連鎖法則是理解反向傳播的關鍵，建議多做手算練習。
* 泰勒展開有助於理解學習率、損失曲面與優化策略。

---

## 延伸閱讀與資源

* [MIT OCW：Calculus for Machine Learning](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
* [3Blue1Brown：微積分動畫](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
* [PyTorch Autograd 官方文件](https://pytorch.org/docs/stable/autograd.html)

---

## 結語

微積分讓我們能精確描述模型的變化與學習過程。掌握導數、偏導、連鎖法則與泰勒展開，不僅能幫助你理解 AI 模型訓練的本質，也能在面試與實作中展現數學底蘊。下一章將進入最適化理論，敬請期待！
