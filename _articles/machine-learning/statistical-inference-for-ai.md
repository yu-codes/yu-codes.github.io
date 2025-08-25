---
title: "統計推論 Toolkit：AI 必備的估計與檢定方法"
date: 2025-05-17 16:00:00 +0800
categories: [Machine Learning]
tags: [統計推論, MLE, MAP, 假設檢定, Bootstrap, 交叉驗證]
---

# 統計推論 Toolkit：AI 必備的估計與檢定方法

統計推論讓我們能從有限資料中推測未知真相，是機器學習模型訓練與評估的基礎。本篇將帶你掌握 AI 常用的統計推論觀念，並以直覺、圖解與 Python 範例說明。

---

## 點估計 vs. 區間估計

- **點估計（Point Estimation）**：用單一數值估計參數（如平均數、機率）。
- **區間估計（Interval Estimation）**：給出參數的可信區間，反映不確定性。

```python
import numpy as np
from scipy import stats

data = np.random.randn(100)
mean = np.mean(data)
conf_int = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
print("平均數點估計:", mean)
print("95% 信賴區間:", conf_int)
```

---

## MLE、MAP 與貝氏估計

- **最大概似估計（MLE）**：選擇讓觀察資料機率最大的參數。
- **最大後驗估計（MAP）**：結合先驗知識與資料，選擇最可能的參數。
- **貝氏估計**：產生參數的完整後驗分布，反映不確定性。

| 方法 | 公式                                      | 直覺說明          |
| ---- | ----------------------------------------- | ----------------- |
| MLE  | $\hat{\theta}_{MLE} = \arg\max_\theta P(D | \theta)$          | 只看資料                 |
| MAP  | $\hat{\theta}_{MAP} = \arg\max_\theta P(D | \theta)P(\theta)$ | 加入先驗                 |
| 貝氏 | $P(\theta                                 | D) = \frac{P(D    | \theta)P(\theta)}{P(D)}$ | 得到分布 |

---

## 假設檢定 (t-test, χ², ANOVA)

- **假設檢定**：判斷資料是否支持某個假設（如兩組平均數是否相等）。
- **t-test**：比較兩組平均數。
- **卡方檢定（χ²）**：檢查分類資料分布。
- **ANOVA**：多組平均數比較。

```python
from scipy.stats import ttest_ind

group1 = np.random.randn(30)
group2 = np.random.randn(30) + 0.5
t_stat, p_val = ttest_ind(group1, group2)
print("t 檢定統計量:", t_stat, "p 值:", p_val)
```

---

## Bootstrap & 交叉驗證概念

- **Bootstrap**：重複隨機抽樣產生多組樣本，估計參數分布與信賴區間。
- **交叉驗證（Cross-Validation）**：將資料分成多份，輪流訓練與驗證，評估模型泛化能力。

```python
from sklearn.utils import resample
data = np.random.randn(100)
boot_means = [np.mean(resample(data)) for _ in range(1000)]
print("Bootstrap 樣本均值分布範例:", boot_means[:5])
```

---

## 常見面試熱點整理

| 熱點主題  | 面試常問問題           |
| --------- | ---------------------- |
| MLE/MAP   | 兩者差異與應用？       |
| 假設檢定  | p 值是什麼？如何解讀？ |
| Bootstrap | 何時用？有什麼優缺點？ |
| 交叉驗證  | 為何能提升模型泛化？   |

---

## 使用注意事項

* 點估計雖簡單，但區間估計更能反映不確定性。
* 假設檢定需注意前提（如常態性、獨立性）。
* Bootstrap 適合樣本量小或分布未知時使用。
* 交叉驗證是模型選擇與調參的標配工具。

---

## 延伸閱讀與資源

* [StatQuest: Maximum Likelihood, Bayesian, and MAP](https://www.youtube.com/watch?v=BrK7X_XlGB8)
* [Khan Academy: 假設檢定](https://zh.khanacademy.org/math/statistics-probability/significance-tests-one-sample)
* [Scikit-learn 交叉驗證文件](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## 結語

統計推論讓我們能從有限資料中做出有信心的推斷。掌握點估計、區間估計、MLE、MAP、假設檢定與交叉驗證，能幫助你設計更可靠的 AI 模型與實驗。下一章將進入信息理論與損失函數，敬請期待！
