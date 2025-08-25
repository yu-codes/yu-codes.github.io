---
title: "統計學在 ML 實務：模型評估、偏差-變異與信賴區間全攻略"
date: 2025-05-17 19:00:00 +0800
categories: [Machine Learning]
tags: [統計學, 偏差-變異, 評估指標, 置信帶, 預測區間]
---

# 統計學在 ML 實務：模型評估、偏差-變異與信賴區間全攻略

統計學不僅是理論，更是機器學習實務不可或缺的工具。從模型評估、調參，到結果解讀，統計思維貫穿整個 ML 流程。本篇將深入探討偏差-變異權衡、各種評估指標、置信帶與預測區間，並結合圖解、Python 實作與面試重點，讓你在專案與面試中都能精準發揮。

---

## Bias-Variance Trade-off 可視化

### 偏差（Bias）與變異（Variance）

- **偏差**：模型預測與真實值的平均差距，反映模型假設的簡化程度。
- **變異**：模型對不同資料集的敏感度，反映模型對資料的擬合程度。

### 權衡與圖解

- 偏差高 → 模型過於簡單（欠擬合）
- 變異高 → 模型過於複雜（過擬合）
- 權衡：找到最佳複雜度，兼顧泛化與準確

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
X = np.linspace(0, 1, 100)[:, None]
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(100) * 0.1

for degree in [1, 4, 15]:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    plt.plot(X, y_pred, label=f"degree={degree}")
plt.scatter(X, y, s=10, color='black')
plt.legend(); plt.title("Bias-Variance Trade-off"); plt.show()
```

---

## 評估指標：MSE, MAE, R², AUC

| 指標 | 公式                                                  | 適用場景      | 直覺說明         |
| ---- | ----------------------------------------------------- | ------------- | ---------------- |
| MSE  | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$                    | 回歸          | 懲罰大誤差       |
| MAE  | $\frac{1}{n}\sum                                      | y_i-\hat{y}_i | $                | 回歸 | 對離群值不敏感 |
| R²   | $1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 回歸          | 解釋變異比例     |
| AUC  | 曲線下方面積                                          | 分類          | 區分正負樣本能力 |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0.1, 0.8, 0.7, 0.2, 0.9]
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
print("AUC:", roc_auc_score(y_true, y_pred))
```

---

## 置信帶 & 預測區間解讀

### 置信帶（Confidence Interval）

- 針對模型參數（如平均數、迴歸係數）給出可信範圍。
- 例如：95% 置信帶表示「多次抽樣有 95% 的機會包含真值」。

### 預測區間（Prediction Interval）

- 針對新觀測值的預測結果給出範圍，通常比置信帶寬。
- 反映模型對未來資料的不確定性。

```python
import statsmodels.api as sm

X = np.random.rand(100)
y = 2 * X + np.random.randn(100) * 0.1
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pred = model.get_prediction(X)
conf_int = pred.conf_int()
pred_int = pred.summary_frame(alpha=0.05)[['obs_ci_lower', 'obs_ci_upper']]
print("置信帶範例:", conf_int[:5])
print("預測區間範例:", pred_int.head())
```

---

## 實務應用與常見誤區

### 實務應用

- 調參時用交叉驗證評估泛化能力
- 報告模型時同時給出點估計與置信帶
- 用 AUC 評估分類模型在不平衡資料下的表現

### 常見誤區

- 置信帶 ≠ 單次預測的可信度
- 高 R² 不代表模型一定好（可能過擬合）
- AUC 高但實際預測效果差，需結合混淆矩陣等指標

---

## 常見面試熱點整理

| 熱點主題        | 面試常問問題           |
| --------------- | ---------------------- |
| Bias-Variance   | 權衡如何實作與可視化？ |
| 評估指標        | 何時選 MSE/MAE/AUC？   |
| 置信帶/預測區間 | 差異與解讀？           |
| R²              | 何時不適用？           |

---

## 使用注意事項

* 評估指標需根據任務選擇，避免單一指標誤導。
* 置信帶與預測區間能提升模型解釋力與可信度。
* 偏差-變異分析有助於模型選擇與調參。

---

## 延伸閱讀與資源

* [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
* [Scikit-learn Metrics 官方文件](https://scikit-learn.org/stable/modules/model_evaluation.html)
* [Statsmodels: Confidence & Prediction Intervals](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html)

---

## 結語

統計學在 ML 實務中扮演關鍵角色。掌握偏差-變異權衡、評估指標、置信帶與預測區間，能讓你更科學地評估與解釋模型表現。下一章將進入機率圖模型，敬請期待！
