---
title: "機率論 Essentials：AI 必備的隨機思維與分布直覺"
date: 2025-05-17 15:00:00 +0800
categories: [Machine Learning]
tags: [機率論, 隨機變數, 機率分布, 貝氏定理, 條件機率]
---

# 機率論 Essentials：AI 必備的隨機思維與分布直覺

機率論是資料科學與 AI 的基礎語言。從模型預測的不確定性，到生成模型的機率分布，背後都依賴機率論的核心概念。本篇將帶你掌握 AI 常用的機率觀念，並以直覺、圖解與 Python 範例說明。

---

## 隨機變數、CDF 與 PDF

### 隨機變數（Random Variable）

- 將隨機實驗的結果對應到數值的函數。
- 分為離散型（Discrete）與連續型（Continuous）。

### 機率質量函數（PMF）、機率密度函數（PDF）、累積分布函數（CDF）

- **PMF**：離散型隨機變數的機率分布（如擲骰子）。
- **PDF**：連續型隨機變數的機率密度（如身高分布）。
- **CDF**：隨機變數小於等於某值的累積機率。

```python
import numpy as np
from scipy.stats import norm, bernoulli
import matplotlib.pyplot as plt

# 連續型：標準常態分布 PDF/CDF
x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x)
cdf = norm.cdf(x)
plt.plot(x, pdf, label="PDF")
plt.plot(x, cdf, label="CDF")
plt.legend(); plt.title("Normal Distribution"); plt.show()

# 離散型：伯努利分布 PMF
p = 0.7
x = [0, 1]
pmf = bernoulli.pmf(x, p)
print("Bernoulli PMF:", pmf)
```

---

## 常見分布一覽

| 分布名稱   | 參數            | 應用場景           |
| ---------- | --------------- | ------------------ |
| 常態分布   | $\mu, \sigma$   | 誤差、噪聲建模     |
| 伯努利分布 | $p$             | 二元分類           |
| 卡方分布   | $k$             | 假設檢定           |
| Beta 分布  | $\alpha, \beta$ | 機率建模、貝氏推論 |

---

## 獨立、條件機率、全機率公式

- **獨立事件**：$P(A \cap B) = P(A)P(B)$
- **條件機率**：$P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **全機率公式**：將複雜事件拆解為多個子事件的加總。

```python
# 條件機率範例
P_A = 0.3
P_B = 0.5
P_A_and_B = 0.15
P_A_given_B = P_A_and_B / P_B
print("P(A|B) =", P_A_given_B)
```

---

## 貝氏定理 & 先驗 / 後驗直覺

- **貝氏定理**：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- **先驗（Prior）**：事前對事件的信念。
- **後驗（Posterior）**：觀察到資料後，更新的信念。

> 在 AI 中，貝氏定理用於機率推論、生成模型、貝氏優化等場景。

---

## 常見面試熱點整理

| 熱點主題 | 面試常問問題           |
| -------- | ---------------------- |
| 常態分布 | 為何常態分布如此重要？ |
| 條件機率 | 如何用全機率公式推導？ |
| 貝氏定理 | 先驗/後驗的直覺？      |
| 獨立性   | 什麼時候事件獨立？     |

---

## 使用注意事項

* 機率分布選擇會影響模型假設與推論結果。
* 條件機率與貝氏定理是理解生成模型與推論的基礎。
* 熟悉 SciPy、NumPy 等工具可簡化機率分布運算。

---

## 延伸閱讀與資源

* [StatQuest: Probability Distributions](https://www.youtube.com/watch?v=Vfo5le26IhY)
* [Khan Academy: 機率與統計](https://zh.khanacademy.org/math/statistics-probability)
* [Scipy.stats 官方文件](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## 結語

機率論讓我們能量化不確定性，為 AI 模型提供理論基礎。掌握隨機變數、分布、條件機率與貝氏定理，能幫助你設計更強大、更靈活的機器學習模型。下一章將進入統計推論，敬請期待！
