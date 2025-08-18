---
title: "模型評估與驗證全攻略：交叉驗證、分類/迴歸指標、資料洩漏與早停"
date: 2025-05-18 18:00:00 +0800
categories: [Machine Learning]
tags: [模型評估, 交叉驗證, Precision, Recall, ROC-AUC, F1, MSE, MAE, Data Leakage, Early Stopping]
---

# 模型評估與驗證全攻略：交叉驗證、分類/迴歸指標、資料洩漏與早停

模型評估與驗證是機器學習流程中不可或缺的一環。正確的評估策略能幫助我們選擇最佳模型、避免過擬合、提升泛化能力。本章將深入交叉驗證策略、分類與迴歸指標、資料洩漏與早停技巧，結合理論、實作、面試熱點與常見誤區，讓你在專案與面試中都能精準評估模型表現。

---

## 交叉驗證策略 (K-Fold, Stratified, TimeSeriesSplit)

### K-Fold Cross-Validation

- 將資料分成 K 份，輪流用一份驗證，其餘訓練，平均結果。
- 適合資料量有限時提升評估穩定性。

### Stratified K-Fold

- 分層抽樣，確保每折類別比例與原資料一致。
- 適合分類任務，避免類別不平衡影響。

### TimeSeriesSplit

- 適合時間序列資料，保留時間順序，避免未來資料洩漏到過去。

```python
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

X = [[i] for i in range(10)]
y = [0, 1]*5

kf = KFold(n_splits=5)
skf = StratifiedKFold(n_splits=2)
tscv = TimeSeriesSplit(n_splits=3)

for train, test in kf.split(X):
    print("KFold:", train, test)
for train, test in skf.split(X, y):
    print("StratifiedKFold:", train, test)
for train, test in tscv.split(X):
    print("TimeSeriesSplit:", train, test)
```

---

## 分類指標：Precision-Recall / ROC-AUC / F1

### Precision, Recall, F1

- **Precision**：預測為正的樣本中有多少是真的正。
- **Recall**：所有正樣本中有多少被正確預測。
- **F1 Score**：Precision 與 Recall 的調和平均。

### ROC-AUC

- ROC 曲線：TPR vs FPR，AUC 為曲線下方面積。
- 適合不平衡資料，反映模型區分正負樣本能力。

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
y_prob = [0.1, 0.8, 0.4, 0.3, 0.9]
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_prob))
```

---

## 迴歸指標：MSE / MAE / R²

| 指標 | 公式                                                  | 適用場景      | 直覺說明     |
| ---- | ----------------------------------------------------- | ------------- | ------------ |
| MSE  | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$                    | 回歸          | 懲罰大誤差   |
| MAE  | $\frac{1}{n}\sum                                      | y_i-\hat{y}_i | $            | 回歸 | 對離群值不敏感 |
| R²   | $1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 回歸          | 解釋變異比例 |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
```

---

## Data Leakage & 早停 (Early Stopping)

### Data Leakage（資料洩漏）

- 訓練時不小心用到未來或驗證資料，導致評估過於樂觀。
- 常見於特徵工程、標準化、目標編碼時未分開處理。

### Early Stopping（早停）

- 監控驗證集表現，若多輪未提升則提前停止訓練，防止過擬合。
- 常用於深度學習、Boosting 類模型。

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, validation_fraction=0.1, n_iter_no_change=5)
gb.fit(X, y)
print("最佳迭代數:", gb.n_estimators_)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 交叉驗證：模型選擇、調參、泛化能力評估
- Precision-Recall：醫療、金融等高風險領域
- ROC-AUC：不平衡分類、模型比較
- Early Stopping：深度學習、Boosting

### 常見誤區

- 交叉驗證未分層，導致類別不平衡。
- 評估指標選錯，誤導模型選擇。
- 特徵工程未分開訓練/驗證集，產生資料洩漏。
- 早停監控訓練集而非驗證集，無法防止過擬合。

---

## 面試熱點與經典問題

| 主題             | 常見問題                                    |
| ---------------- | ------------------------------------------- |
| 交叉驗證         | 為何能提升泛化？K-Fold 與 Stratified 差異？ |
| Precision/Recall | 何時優先考慮？有何 trade-off？              |
| ROC-AUC          | 如何解讀？何時不適用？                      |
| Data Leakage     | 常見來源？如何避免？                        |
| Early Stopping   | 如何設計？有何風險？                        |

---

## 使用注意事項

* 評估指標需根據任務選擇，避免單一指標誤導。
* 交叉驗證與特徵工程順序需正確，防止資料洩漏。
* Early Stopping 須監控驗證集，並設合理 patience。

---

## 延伸閱讀與資源

* [StatQuest: Model Evaluation](https://www.youtube.com/c/joshstarmer)
* [Scikit-learn Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
* [Early Stopping in Deep Learning](https://keras.io/api/callbacks/early_stopping/)

---

## 經典面試題與解法提示

1. K-Fold 與 Stratified K-Fold 差異？
2. Precision 與 Recall 何時優先考慮？
3. ROC-AUC 有哪些限制？
4. Data Leakage 常見來源與防範？
5. Early Stopping 如何設計與調參？
6. 交叉驗證如何提升泛化能力？
7. 何時用 MAE 而非 MSE？
8. R² 何時不適用？
9. 如何用 Python 畫 ROC 曲線？
10. 多指標綜合評估的策略？

---

## 結語

模型評估與驗證是 ML 成敗關鍵。熟悉交叉驗證、分類/迴歸指標、資料洩漏與早停技巧，能讓你打造更可靠的模型並在面試中展現專業素養。下一章將進入正則化與泛化理論，敬請期待！
