---
title: "ML 經典面試題庫：13 章重點題型與解法直覺"
date: 2025-05-18 23:30:00 +0800
categories: [機器學習理論]
tags: [面試題, 機器學習, 解題技巧, 直覺, 經典題庫]
---

# ML 經典面試題庫：13 章重點題型與解法直覺

本章彙整前述 12 章機器學習理論的經典面試題，每章精選 10-15 題，涵蓋計算、推導、直覺解釋與實務應用。每題附上解法提示與常見誤區，幫助你在面試與實戰中脫穎而出。

---

## ML1 核心概念暖身

1. 監督、非監督、半監督、強化學習差異？
2. Bias-Variance 葛藤的數學分解與圖解？
3. 泛化誤差如何分解？如何降低？
4. 欠擬合與過擬合的診斷與調整方法？
5. 學習曲線的意義與應用？
6. 監督學習常見任務與代表演算法？
7. 強化學習適用場景？
8. 泛化能力與資料分割的關係？
9. Underfitting/Overfitting 的實務案例？
10. 如何用 Python 畫學習曲線？

---

## ML2 經典迴歸模型

1. 線性迴歸的數學推導與假設？
2. Ridge、Lasso、Elastic Net 差異與適用場景？
3. Logistic Regression 的損失函數與推導？
4. 多項式迴歸如何避免過擬合？
5. 偏態資料如何處理？對數轉換有何風險？
6. 重抽樣方法有哪些？何時用 SMOTE？
7. 正則化對模型有何影響？
8. 如何用 Python 實作 Lasso？
9. Logistic Regression 適用於哪些任務？
10. Ridge/Lasso 係數全為 0 的原因？

---

## ML3 分類演算法百寶箱

1. k-NN 為何不用訓練？如何選 k？
2. SVM 的 Kernel Trick 原理與應用？
3. Naive Bayes 的條件獨立假設有何影響？
4. 決策樹如何避免過擬合？
5. k-NN 對特徵縮放敏感嗎？為什麼？
6. SVM 硬/軟邊界差異？
7. 決策樹的資訊增益與基尼指數差異？
8. Naive Bayes 適用於哪些資料型態？
9. 如何用 Python 實作 SVM？
10. 決策樹剪枝的原理？

---

## ML4 集成學習 (Ensemble)

1. Bagging 與 Boosting 差異與數學推導？
2. Random Forest 為何不易過擬合？
3. XGBoost 如何處理缺失值？
4. Stacking 如何設計 Level-1 模型？
5. Boosting 為何對異常值敏感？
6. Bagging 適合哪些弱模型？
7. LightGBM 與 XGBoost 差異？
8. 集成方法如何提升泛化能力？
9. 超學習器理論基礎？
10. 如何用 Python 實作 Stacking？

---

## ML5 非監督學習大補帖

1. K-means 為何對初始值敏感？如何改進？
2. DBSCAN 如何自動判斷群數？有何限制？
3. GMM 與 K-means 差異？
4. EM 演算法的數學推導？
5. PCA 如何選主成分數量？
6. t-SNE/UMAP 適合哪些應用？
7. 密度估計有哪些方法？各自優缺點？
8. 聚類評估指標有哪些？
9. 非監督學習如何驗證效果？
10. 如何用 Python 實作多種聚類並比較？

---

## ML6 特徵工程 & 選擇

1. One-Hot Encoding 有哪些缺點？如何解決？
2. Target Encoding 如何防止資料洩漏？
3. 標準化與正規化差異？
4. Filter/Wrapper/Embedded 方法比較？
5. Lasso 為何能做特徵選擇？
6. 特徵選擇對模型有何影響？
7. 如何用 Python 實作特徵選擇？
8. 特徵工程常見陷阱有哪些？
9. 如何評估特徵工程效果？
10. 實務上如何設計特徵工程流程？

---

## ML7 模型評估 & 驗證

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

## ML8 正則化與泛化理論

1. VC Dimension 如何影響泛化能力？
2. Dropout 的數學原理與實作細節？
3. Label Smoothing 有哪些優缺點？
4. Data Augmentation 在 NLP/影像的常見方法？
5. Ensemble 與正則化的異同？
6. 如何評估模型泛化能力？
7. Dropout、Ensemble、L1/L2 正則化如何選擇？
8. 泛化能力與 Bias-Variance 的關係？
9. 實務上如何設計正則化策略？
10. 如何用 Python 實作 Dropout/Label Smoothing？

---

## ML9 超參數最佳化

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

## ML10 貝式方法 & 機率視角

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

## ML11 強化式學習速查

1. MDP 的五大元素與貝爾曼方程推導？
2. Q-Learning 如何更新？何時收斂？
3. DQN 如何解決 Q-Learning 的限制？
4. Policy Gradient 的數學推導與應用場景？
5. 探索-利用平衡有哪些策略？
6. ε-Greedy 如何設計 ε 衰減？
7. UCB 的數學原理與應用？
8. 強化學習在推薦系統的應用？
9. DQN 訓練不穩定的原因與解法？
10. Policy Gradient 如何降低方差？

---

## ML12 倫理、偏差與公平

1. Demographic Parity 與 Equal Opportunity 差異？
2. GDPR/CCPA 對 ML 流程的實際要求？
3. 偏見來源有哪些？如何偵測？
4. 資料層/模型層/預測層偏見緩解方法？
5. 公平性與準確率如何權衡？
6. 如何用 Python 計算公平性指標？
7. 合規審計流程如何設計？
8. 去除敏感屬性是否足夠？為什麼？
9. AIF360 工具包有哪些功能？
10. 公平性評估在實務中的挑戰？

---

## 解題技巧與常見誤區

- **計算題**：先寫公式再帶數字，避免粗心。
- **推導題**：分步驟寫清楚，標明假設。
- **直覺題**：用圖解、生活例子輔助說明。
- **實作題**：熟悉 numpy、scikit-learn、pytorch 等常用 API。
- **常見誤區**：混淆定義、忽略假設、過度依賴單一指標。

---

## 結語

本題庫涵蓋 ML 理論的經典面試題與解法直覺。建議每題都動手推導、實作與解釋，並多練習口頭表達。祝你面試順利、學習愉快！
