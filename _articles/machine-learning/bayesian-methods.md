---
title: "貝式方法與機率視角全攻略：生成/判別、貝式迴歸、變分推論與 MC Dropout"
date: 2025-05-18 21:00:00 +0800
categories: [Machine Learning]
tags: [貝式方法, 機率視角, 生成模型, 判別模型, 貝式迴歸, Gaussian Process, Variational Inference, Monte Carlo Dropout]
---

# 貝式方法與機率視角全攻略：生成/判別、貝式迴歸、變分推論與 MC Dropout

貝式方法與機率視角是現代機器學習理解不確定性、提升泛化能力的核心。從生成模型與判別模型的本質，到貝式線性迴歸、Gaussian Process Regression、變分推論與 Monte Carlo Dropout，這些理論與實作是面試與研究的熱門話題。本章將深入數學推導、直覺圖解、Python 實作、應用場景、面試熱點與常見誤區，幫助你全面掌握貝式方法。

---

## Generative vs. Discriminative Models

### 生成模型（Generative Model）

- 學習 $P(X, Y)$ 或 $P(X)$，可生成新資料、進行密度估計。
- 例：Naive Bayes、GMM、GAN、VAE。

### 判別模型（Discriminative Model）

- 學習 $P(Y|X)$，直接預測標籤。
- 例：Logistic Regression、SVM、Random Forest。

| 類型     | 學習目標  | 代表模型                   | 優缺點                       |
| -------- | --------- | -------------------------- | ---------------------------- |
| 生成模型 | $P(X, Y)$ | Naive Bayes, GMM, GAN, VAE | 可生成資料、處理缺失，訓練慢 |
| 判別模型 | $P(Y      | X)$                        | LR, SVM, RF                  | 預測準確、訓練快 |

---

## 貝式線性迴歸（Bayesian Linear Regression）

### 理論基礎

- 將參數視為隨機變數，給定先驗分布，觀察資料後更新為後驗分布。
- 可直接量化預測不確定性。

### 數學推導

- 先驗：$\beta \sim \mathcal{N}(0, \lambda^{-1}I)$
- 似然：$y|X, \beta \sim \mathcal{N}(X\beta, \sigma^2I)$
- 後驗：$\beta|X, y \sim \mathcal{N}(\mu, \Sigma)$，可解析計算。

### Python 實作

```python
import numpy as np

X = np.linspace(0, 1, 20)[:, None]
y = 2 * X.ravel() + np.random.randn(20) * 0.2
lambda_ = 1.0
sigma2 = 0.04
Sigma_inv = lambda_ * np.eye(1) + (X.T @ X) / sigma2
Sigma = np.linalg.inv(Sigma_inv)
mu = Sigma @ (X.T @ y) / sigma2
print("後驗均值:", mu)
```

---

## Gaussian Process Regression（高斯過程回歸）

- 非參數貝式模型，直接對函數分布建模。
- 可量化預測均值與不確定性，適合小資料、函數擬合。

### 理論直覺

- 假設任意有限點的函數值服從多元高斯分布。
- 透過核函數（Kernel）控制平滑度與相關性。

### Python 實作

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))
gp.fit(X, y)
y_pred, y_std = gp.predict(X, return_std=True)
print("預測均值:", y_pred[:5])
print("預測標準差:", y_std[:5])
```

---

## Variational Inference（變分推論）

- 用簡單分布近似複雜後驗分布，最大化 Evidence Lower Bound (ELBO)。
- 常用於 VAE、貝式深度學習。

### 理論推導

- 將後驗 $P(\theta|D)$ 近似為 $q(\theta)$，最小化 $KL(q||P)$。
- 透過梯度下降優化 ELBO。

---

## Monte Carlo Dropout

- 在推論時也啟用 Dropout，多次前向傳播取得預測分布。
- 可近似貝式不確定性，常用於深度學習模型。

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.fc(self.drop(x))

model = MCDropoutModel()
model.train()  # 保持 Dropout 啟用
preds = [model(torch.randn(1, 10)).item() for _ in range(100)]
print("MC Dropout 預測均值:", np.mean(preds), "標準差:", np.std(preds))
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 不確定性量化：醫療、金融、風險控制
- 小樣本學習：Gaussian Process、貝式迴歸
- 生成模型：VAE、GAN
- 深度學習不確定性：MC Dropout

### 常見誤區

- 混淆生成/判別模型的學習目標
- 貝式方法計算量大，實務常需近似推論
- MC Dropout 需在推論時保持訓練模式
- Gaussian Process 不適合大規模資料

---

## 面試熱點與經典問題

| 主題             | 常見問題                  |
| ---------------- | ------------------------- |
| 生成 vs 判別     | 差異與適用場景？          |
| 貝式迴歸         | 先驗/後驗推導？           |
| Gaussian Process | 優缺點與應用？            |
| 變分推論         | 為何需近似？ELBO 是什麼？ |
| MC Dropout       | 如何量化不確定性？        |

---

## 使用注意事項

* 貝式方法適合需量化不確定性的任務，但計算成本高。
* 變分推論與 MC Dropout 需多次前向傳播，注意效能。
* Gaussian Process 適合小資料、函數擬合，需選好核函數。

---

## 延伸閱讀與資源

* [Deep Learning Book: Bayesian Methods](https://www.deeplearningbook.org/contents/probabilistic.html)
* [Pyro: Probabilistic Programming](https://pyro.ai/)
* [Scikit-learn Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html)
* [MC Dropout 論文](https://arxiv.org/abs/1506.02142)

---

## 經典面試題與解法提示

1. 生成模型與判別模型的差異？
2. 貝式線性迴歸的數學推導？
3. Gaussian Process 如何量化不確定性？
4. 變分推論的核心思想與應用？
5. MC Dropout 如何近似貝式不確定性？
6. 生成模型有哪些應用？
7. 何時選用貝式方法？
8. Gaussian Process 的核函數如何選擇？
9. 變分推論與 MCMC 差異？
10. 如何用 Python 實作 MC Dropout？

---

## 結語

貝式方法與機率視角是 ML 理論與實務的高階武器。熟悉生成/判別模型、貝式迴歸、Gaussian Process、變分推論與 MC Dropout，能讓你在不確定性建模、研究與面試中展現專業深度。下一章將進入強化式學習速查，敬請期待！
