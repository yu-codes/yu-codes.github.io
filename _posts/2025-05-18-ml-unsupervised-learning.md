---
title: "非監督學習大補帖：聚類、密度估計、降維全解析"
date: 2025-05-18 16:00:00 +0800
categories: [Machine Learning]
tags: [非監督學習, 聚類, K-means, DBSCAN, GMM, PCA, t-SNE, UMAP]
---

# 非監督學習大補帖：聚類、密度估計、降維全解析

非監督學習是資料探索、特徵工程與生成模型的基礎。從經典的 K-means、DBSCAN、GMM，到降維利器 PCA、t-SNE、UMAP，這些方法能幫助我們發現資料結構、壓縮高維資訊、提升後續模型表現。本章將深入數學原理、直覺圖解、Python 實作、面試熱點與常見誤區，讓你全面掌握非監督學習。

---

## 聚類：K-means、DBSCAN、Spectral Clustering

### K-means

- 將資料分為 K 群，最小化群內平方誤差。
- 迭代步驟：隨機初始化中心→分配點→更新中心→重複。
- 對初始值敏感，需多次重啟。

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(200, 2)
kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='x')
plt.title("K-means Clustering"); plt.show()
```

### DBSCAN

- 基於密度的聚類，能發現任意形狀的群集。
- 參數：鄰域半徑 eps、最小點數 min_samples。
- 可自動識別噪聲點，不需指定群數。

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=5).fit(X)
plt.scatter(X[:,0], X[:,1], c=db.labels_)
plt.title("DBSCAN Clustering"); plt.show()
```

### Spectral Clustering

- 利用圖論與特徵分解，適合非凸形狀資料。
- 先建鄰接圖，再做特徵分解與 K-means。

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors').fit(X)
plt.scatter(X[:,0], X[:,1], c=sc.labels_)
plt.title("Spectral Clustering"); plt.show()
```

---

## 密度估計 & 混合模型（GMM, EM）

### 密度估計

- 估計資料分布函數，常用於異常偵測、生成模型。
- 方法：直方圖、核密度估計（KDE）、混合模型。

### GMM（Gaussian Mixture Model）

- 用多個高斯分布混合建模資料，參數用 EM 演算法學習。
- 可自動分群、密度估計、異常偵測。

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("GMM Clustering"); plt.show()
```

### EM 演算法

- 交替進行 E 步（計算隱變量期望）與 M 步（最大化參數）。
- 適用於含隱變量的最大概似估計。

---

## 降維：PCA、t-SNE、UMAP 比較

### PCA（主成分分析）

- 線性降維，找最大變異方向。
- 可視化高維資料、特徵壓縮、去除共線性。

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA Projection"); plt.show()
```

### t-SNE

- 非線性降維，保留局部結構，適合視覺化。
- 對超參數敏感，計算量大。

```python
from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1])
plt.title("t-SNE Projection"); plt.show()
```

### UMAP

- 非線性降維，速度快、可保留全域與局部結構。
- 適合大規模資料、互動式視覺化。

```python
import umap

X_umap = umap.UMAP(n_components=2).fit_transform(X)
plt.scatter(X_umap[:,0], X_umap[:,1])
plt.title("UMAP Projection"); plt.show()
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 聚類：市場分群、影像分割、社群偵測
- 密度估計：異常偵測、生成模型
- 降維：資料視覺化、特徵工程、噪聲過濾

### 常見誤區

- K-means 只適合球狀群集，對離群值敏感。
- DBSCAN 參數選擇不當易分錯群。
- GMM 假設群內分布為高斯，實務未必成立。
- t-SNE 僅適合視覺化，無法做新資料投影。
- PCA 只保留線性結構，忽略非線性資訊。

---

## 面試熱點與經典問題

| 主題           | 常見問題                     |
| -------------- | ---------------------------- |
| K-means        | 如何選 K？初始值敏感怎麼辦？ |
| DBSCAN         | 參數如何選？優缺點？         |
| GMM/EM         | EM 步驟推導？何時用 GMM？    |
| PCA/t-SNE/UMAP | 差異與適用場景？             |
| 密度估計       | KDE 與 GMM 差異？            |

---

## 使用注意事項

* 聚類與降維結果需結合領域知識解讀。
* 非監督學習無標準答案，建議多種方法交叉驗證。
* 降維前建議先標準化資料，避免特徵尺度影響。

---

## 延伸閱讀與資源

* [StatQuest: K-means, GMM, PCA, t-SNE](https://www.youtube.com/c/joshstarmer)
* [Scikit-learn Clustering & Decomposition](https://scikit-learn.org/stable/modules/clustering.html)
* [UMAP 官方文件](https://umap-learn.readthedocs.io/en/latest/)
* [t-SNE 理論與實作](https://distill.pub/2016/misread-tsne/)

---

## 經典面試題與解法提示

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

## 結語

非監督學習是資料探索與特徵工程的關鍵。熟悉聚類、密度估計、降維方法，能讓你在資料分析、模型前處理與面試中展現專業素養。下一章將進入特徵工程與選擇，敬請期待！
