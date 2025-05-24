---
title: "超參數最佳化全攻略：Grid/Random/Bayesian、Hyperband 與重現性"
date: 2025-05-18 20:00:00 +0800
categories: [機器學習理論]
tags: [超參數, Grid Search, Random Search, Bayesian Optimization, Hyperband, Reproducibility]
---

# 超參數最佳化全攻略：Grid/Random/Bayesian、Hyperband 與重現性

超參數最佳化是機器學習模型性能提升的關鍵。從傳統的 Grid/Random Search，到進階的 Bayesian Optimization、Hyperband、Population Based Training（PBT），再到重現性與勢能坑問題，這些技巧與理論是面試與實務不可或缺的能力。本章將深入原理、實作、調參策略、面試熱點與常見誤區，幫助你全面掌握超參數調優。

---

## Grid Search vs. Random Search

### Grid Search

- 枚舉所有超參數組合，逐一訓練與評估。
- 適合超參數數量少、每個值都需測試的情境。
- 缺點：組合數爆炸，計算成本高。

### Random Search

- 隨機抽樣超參數組合，訓練與評估。
- 適合超參數多、部分參數影響大時。
- 理論證明：隨機搜尋常能更快找到好組合。

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit([[0,0],[1,1]], [0,1])

param_dist = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}
rand = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=4, cv=3)
rand.fit([[0,0],[1,1]], [0,1])
```

---

## Bayesian Optimization

- 用貝式方法建模超參數與分數的關係，根據後驗分布選取下次測試點。
- 常用 Gaussian Process、Tree-structured Parzen Estimator（TPE）。
- 優點：能在有限次數內找到更佳組合，適合高成本訓練。

### Python 實作（Optuna 範例）

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    # ...訓練模型並回傳分數...
    return n_estimators - max_depth  # 範例

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("最佳參數:", study.best_params)
```

---

## Hyperband / BOHB / Population Based Training

### Hyperband

- 結合隨機搜尋與早停，快速淘汰表現差的組合。
- 適合大規模超參數搜尋。

### BOHB（Bayesian Optimization + Hyperband）

- 結合 Bayesian Optimization 與 Hyperband，兼顧探索與效率。

### Population Based Training（PBT）

- 多組模型並行訓練，定期交換與微調超參數。
- 適合深度學習大規模訓練。

---

## 勢能坑 (Local Minima) 與重現性 (Reproducibility)

### 勢能坑（Local Minima）

- 訓練過程易陷入局部最小值，導致模型表現不穩。
- 解法：多次初始化、使用動量、調整學習率、集成多模型。

### 重現性（Reproducibility）

- 設定隨機種子、固定資料分割、記錄環境與參數，確保結果可重現。
- 常見於論文、競賽、產業部署。

```python
import numpy as np
import torch
import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- Grid/Random Search：小型專案、少量超參數
- Bayesian Optimization/Hyperband：深度學習、大型專案
- PBT：分散式訓練、AutoML

### 常見誤區

- 忽略超參數間交互作用，僅單獨調整
- 未設隨機種子，導致結果不穩
- 只用預設參數，未做調參實驗
- 忽略早停與資源分配，浪費計算資源

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Grid vs Random| 何時選用？優缺點？ |
| Bayesian Opt | 原理與優勢？ |
| Hyperband    | 如何加速搜尋？ |
| 勢能坑       | 如何避免？ |
| 重現性       | 如何確保？有哪些步驟？ |

---

## 使用注意事項

* 超參數調優需結合交叉驗證與多指標評估。
* 設定隨機種子與記錄參數，確保實驗可重現。
* 大型搜尋建議用 Hyperband/BOHB/PBT 提升效率。

---

## 延伸閱讀與資源

* [Optuna 官方文件](https://optuna.org/)
* [Ray Tune: Hyperparameter Search](https://docs.ray.io/en/latest/tune/index.html)
* [Scikit-learn Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
* [Deep Learning Book: Optimization](https://www.deeplearningbook.org/contents/optimization.html)

---

## 經典面試題與解法提示

1. Grid Search 與 Random Search 差異與適用場景？
2. Bayesian Optimization 如何選下次測試點？
3. Hyperband 如何加速搜尋？
4. 勢能坑與全域最小值的差異？
5. 如何確保實驗重現性？
6. PBT 的原理與優缺點？
7. 超參數調優常見指標有哪些？
8. 如何用 Python 設定隨機種子？
9. BOHB 與 Hyperband 差異？
10. AutoML 如何自動化超參數搜尋？

---

## 結語

超參數最佳化是 ML 成敗的最後一哩路。熟悉 Grid/Random/Bayesian、Hyperband、PBT 與重現性技巧，能讓你打造更強大、穩定的模型並在面試中展現專業素養。下一章將進入貝式方法與機率視角，敬請期待！
