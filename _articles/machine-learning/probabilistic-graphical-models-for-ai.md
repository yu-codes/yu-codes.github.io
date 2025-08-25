---
title: "機率圖模型全解析：Bayesian Network、MRF、EM 與隱變量模型"
date: 2025-05-17 20:00:00 +0800
categories: [Machine Learning]
tags: [機率圖模型, Bayesian Network, Markov Random Field, EM, HMM, GMM]
---

# 機率圖模型全解析：Bayesian Network、MRF、EM 與隱變量模型

機率圖模型（Probabilistic Graphical Models, PGM）是結合機率論與圖論的強大工具，能以視覺化方式描述複雜隨機變數間的依賴關係。從貝氏網路、馬可夫隨機場，到 EM 演算法與隱變量模型，這些理論是現代 AI、NLP、推薦系統、生成模型的基石。本篇將深入剖析 PGM 的數學結構、推理方法、應用場景與 Python 實作，並結合圖解、面試熱點與常見誤區，讓你徹底掌握這門 AI 必修課。

---

## 機率圖模型概觀

### 什麼是機率圖模型？

- 用圖（節點=隨機變數，邊=依賴關係）來表示高維機率分布。
- 兩大類型：
  - **有向圖（Bayesian Network, BN）**：邊有方向，描述因果關係。
  - **無向圖（Markov Random Field, MRF）**：邊無方向，描述對稱依賴。

### 優勢

- 可視化複雜依賴結構
- 降低參數數量（條件獨立性）
- 支援高效推理與學習

---

## Bayesian Network（貝氏網路）

### 結構與數學基礎

- 有向無環圖（DAG），每個節點的機率只依賴其父節點。
- 聯合分布可分解為：
  $$
  P(X_1, ..., X_n) = \prod_{i=1}^n P(X_i | Pa(X_i))
  $$
  其中 $Pa(X_i)$ 為 $X_i$ 的父節點集合。

### 推理與學習

- **推理**：給定部分變數，計算其他變數的條件機率（如貝氏推斷、信念傳播）。
- **結構學習**：從資料自動學習圖結構與參數。

### Python 實作

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = BayesianNetwork([('Cloudy', 'Rain'), ('Rain', 'Sprinkler'), ('Sprinkler', 'WetGrass')])
cpd_cloudy = TabularCPD('Cloudy', 2, [[0.5], [0.5]])
# ...定義其他 CPD...
model.add_cpds(cpd_cloudy)
# ...推理與查詢...
```

---

## Markov Random Field（馬可夫隨機場）

### 結構與數學基礎

- 無向圖，節點間的依賴對稱。
- 聯合分布分解為潛在函數（potential function）之乘積：
  $$
  P(X_1, ..., X_n) = \frac{1}{Z} \prod_{C \in cliques} \psi_C(X_C)
  $$
  其中 $Z$ 為正規化常數。

### 應用場景

- 圖像分割（像素間關聯）
- NLP（詞性標註、命名實體識別）

### Python 實作

```python
from pgmpy.models import MarkovModel
model = MarkovModel()
model.add_nodes_from(['A', 'B', 'C'])
model.add_edges_from([('A', 'B'), ('B', 'C')])
# ...定義潛在函數與推理...
```

---

## EM 演算法核心概念

### EM（Expectation-Maximization）流程

1. **E 步驟**：根據現有參數，計算隱變量的期望（後驗分布）。
2. **M 步驟**：最大化期望下的參數對數似然，更新參數。
3. 重複 E/M 步驟直到收斂。

### 理論推導

- 適用於含有隱變量的最大概似估計問題。
- 常見於 GMM、HMM、主題模型等。

### Python 實作（GMM）

```python
from sklearn.mixture import GaussianMixture
import numpy as np

X = np.random.randn(100, 2)
gmm = GaussianMixture(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)
print("分群結果:", labels[:10])
```

---

## 隱變量模型（HMM, GMM）

### HMM（隱馬可夫模型）

- 適用於序列資料（語音、文字、DNA）。
- 狀態不可見，僅觀察到輸出。
- 典型應用：語音辨識、詞性標註、時間序列預測。

#### HMM 結構

- 狀態轉移機率、觀察機率、初始機率
- 前向-後向演算法、Viterbi 演算法

### GMM（高斯混合模型）

- 用多個高斯分布混合建模資料分布。
- 可自動分群、密度估計、異常偵測。

#### GMM 結構

- 每個分群一組均值、共變異數、權重
- EM 演算法自動學習參數

---

## PGM 在 AI 實務的應用

- NLP：語音辨識、機器翻譯、主題模型（LDA）
- 圖像處理：圖像分割、超像素分群
- 推薦系統：用戶-物品關聯建模
- 生成模型：深度生成模型（如 VAE、GAN 的圖結構擴展）

---

## 理論直覺、圖解與常見誤區

### 直覺圖解

- 有向圖：因果推理、資訊流動有方向
- 無向圖：對稱依賴、局部一致性

### 常見誤區

- 誤以為所有依賴都能用有向圖表示（部分需用無向圖）
- 忽略條件獨立性，導致參數爆炸
- EM 只保證局部最優，初始值敏感

---

## 常見面試熱點整理

| 熱點主題         | 面試常問問題       |
| ---------------- | ------------------ |
| Bayesian Network | 如何分解聯合分布？ |
| MRF              | 何時用無向圖？     |
| EM 演算法        | 推導與收斂性？     |
| HMM/GMM          | 實際應用與推理？   |

---

## 使用注意事項

* PGM 適合高維、依賴複雜的資料建模，但計算量大時需近似推理（如 MCMC、變分法）。
* EM 需多次初始化避免陷入壞局部最優。
* HMM/GMM 需根據資料特性選擇狀態數、分群數。

---

## 延伸閱讀與資源

* [Probabilistic Graphical Models (Stanford)](https://web.stanford.edu/class/cs228/)
* [pgmpy 官方文件](https://pgmpy.org/)
* [StatQuest: HMM, GMM, EM](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
* [深度學習書：PGM 章節](https://www.deeplearningbook.org/contents/graphical.html)

---

## 結語

機率圖模型是 AI 理論與實務的強大工具。掌握 Bayesian Network、MRF、EM 與隱變量模型，不僅能讓你建構更靈活的生成模型，也能在 NLP、推薦、圖像等多領域發揮威力。下一章將帶來經典面試題庫與解法，敬請期待！
