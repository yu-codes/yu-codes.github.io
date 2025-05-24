---
title: "超參數尋優進階：Grid/Random/Bayesian、Hyperband、PBT 與多目標優化"
date: 2025-05-20 20:00:00 +0800
categories: [模型訓練與優化]
tags: [超參數, HPO, Grid Search, Random Search, Bayesian Optimization, Hyperband, PBT, Optuna, BOHB, 多目標優化]
---

# 超參數尋優進階：Grid/Random/Bayesian、Hyperband、PBT 與多目標優化

超參數尋優（Hyperparameter Optimization, HPO）是提升模型表現與資源利用率的關鍵。從 Grid/Random Search、Bayesian Optimization（BOHB, Optuna）、Hyperband、Population-Based Training（PBT），到多目標優化（精度、延遲、資源），本章將深入原理、實作、面試熱點與常見誤區，幫助你高效調參。

---

## Grid / Random / Bayesian (BOHB, Optuna)

### Grid Search

- 枚舉所有超參數組合，適合少量參數
- 缺點：組合爆炸，計算成本高

### Random Search

- 隨機抽樣超參數組合，適合高維空間
- 理論證明：常能更快找到好組合

### Bayesian Optimization

- 用貝式方法建模超參數與分數關係，根據後驗分布選下次測試點
- 工具：Optuna、BOHB

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # ...訓練模型並回傳分數...
    return lr - batch_size  # 範例

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("最佳參數:", study.best_params)
```

---

## Hyperband／Population-Based Training

### Hyperband

- 結合隨機搜尋與早停，快速淘汰表現差的組合
- 適合大規模搜尋

### PBT（Population-Based Training）

- 多組模型並行訓練，定期交換與微調超參數
- 適合深度學習大規模訓練

---

## 多目標優化：資源、精度、延遲

- 同時考慮多個目標（如精度、延遲、記憶體），尋找 Pareto 最佳解
- 工具：Optuna、Ray Tune 支援多目標搜尋

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 大模型訓練、AutoML、資源受限部署、精度/延遲 trade-off

### 常見誤區

- 只調單一指標，忽略資源/延遲
- 未設隨機種子，導致結果不穩
- 搜尋空間設置過大，浪費資源
- 早停策略設置不當，淘汰潛力組合

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Grid vs Random| 差異與適用場景？ |
| Bayesian Opt | 原理與優勢？ |
| Hyperband    | 如何加速搜尋？ |
| PBT          | 如何提升泛化？ |
| 多目標優化   | 如何設計與評估？ |

---

## 使用注意事項

* HPO 需結合交叉驗證與多指標評估
* 設定隨機種子與記錄參數，確保可重現
* 大型搜尋建議用 Hyperband/PBT 提升效率

---

## 延伸閱讀與資源

* [Optuna 官方文件](https://optuna.org/)
* [Ray Tune 官方文件](https://docs.ray.io/en/latest/tune/index.html)
* [BOHB 論文](https://arxiv.org/abs/1807.01774)
* [Hyperband 論文](https://arxiv.org/abs/1603.06560)
* [Population Based Training 論文](https://arxiv.org/abs/1711.09846)

---

## 經典面試題與解法提示

1. Grid/Random/Bayesian 搜尋原理與差異？
2. Hyperband 如何加速搜尋？
3. PBT 的原理與優缺點？
4. 多目標優化如何設計？
5. HPO 結果如何確保可重現？
6. 搜尋空間設計原則？
7. 如何用 Python 實作 Optuna 搜尋？
8. Hyperband 早停策略設計？
9. 多目標優化的評估方法？
10. HPO 常見陷阱與 debug？

---

## 結語

超參數尋優是模型訓練的最後一哩路。熟悉 Grid/Random/Bayesian、Hyperband、PBT 與多目標優化，能讓你高效調參並在面試中展現專業素養。下一章將進入訓練監控與 Debug，敬請期待！
