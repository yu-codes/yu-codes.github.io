---
title: "生成模型百花齊放：自回歸、VAE、GAN、Diffusion 與應用全解析"
date: 2025-05-19 19:00:00 +0800
categories: [Machine Learning]
tags: [生成模型, 自回歸, VAE, GAN, Diffusion, Flow-based, ControlNet]
---

# 生成模型百花齊放：自回歸、VAE、GAN、Diffusion 與應用全解析

生成模型是現代 AI 最活躍的領域之一，從自回歸、VAE、GAN、Flow-based 到 Diffusion，各有理論基礎與應用場景。本章將深入數學推導、直覺圖解、Python 實作、應用案例、面試熱點與常見誤區，幫助你全面掌握生成模型。

---

## 自回歸 vs. 自編碼 vs. Diffusion

### 自回歸模型（Autoregressive）

- 逐步生成資料，每步條件於前一步（如 GPT、PixelRNN）
- 優點：生成品質高，缺點：推理慢

### 自編碼模型（Autoencoder, VAE）

- 編碼器將資料壓縮為潛在空間，解碼器重建資料
- VAE 加入機率假設，能生成新樣本並量化不確定性

### Diffusion Model

- 逐步將資料加噪聲，再學習反向去噪過程（如 Stable Diffusion）
- 優點：生成多樣性高，缺點：推理慢

---

## GAN, VAE, Flow-based Model 核心 Loss

### GAN（生成對抗網路）

- 生成器與判別器對抗訓練，損失函數為 min-max 遊戲
- 常見問題：mode collapse、不穩定

```python
import torch
import torch.nn as nn

D = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
G = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
loss_fn = nn.BCELoss()
# ...訓練流程略...
```

### VAE（變分自編碼器）

- Evidence Lower Bound (ELBO) 作為損失，包含重建誤差與 KL 散度
- 能生成新樣本並量化潛在空間

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(2, 4)
        self.dec = nn.Linear(2, 2)
    def forward(self, x):
        mu = self.enc(x)
        z = mu  # 簡化，實際應加隨機性
        return self.dec(z)
```

### Flow-based Model

- 可逆變換，精確計算 likelihood（如 RealNVP、Glow）
- 適合密度估計與高品質生成

---

## Diffusion Upsampler、Latent Diffusion & ControlNet

### Diffusion Upsampler

- 用於高解析度圖像生成，先低清再逐步升級

### Latent Diffusion

- 在潛在空間進行 diffusion，提升效率（如 Stable Diffusion）

### ControlNet

- 在 diffusion 過程中加入條件控制（如姿態、邊緣圖），提升可控性

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 影像生成（Stable Diffusion、GAN）、語音合成、文本生成（GPT）、異常偵測、資料增強

### 常見誤區

- GAN 訓練不穩定，需調參與技巧（如 label smoothing、gradient penalty）
- VAE 生成樣本模糊，需調整潛在空間維度
- Diffusion 推理慢，可用 DDIM、Latent Diffusion 加速
- Flow-based 模型參數多，訓練成本高

---

## 面試熱點與經典問題

| 主題       | 常見問題                      |
| ---------- | ----------------------------- |
| GAN        | 損失函數推導？mode collapse？ |
| VAE        | ELBO 結構與 KL 散度作用？     |
| Diffusion  | 正向/反向過程數學原理？       |
| Flow-based | 可逆變換如何設計？            |
| ControlNet | 如何提升生成可控性？          |

---

## 使用注意事項

* 生成模型需大量資料與算力，建議用預訓練權重微調
* GAN/VAE/Diffusion 各有優缺點，根據任務選擇
* 訓練過程需監控生成品質與多樣性

---

## 延伸閱讀與資源

* [GAN 原論文](https://arxiv.org/abs/1406.2661)
* [VAE 原論文](https://arxiv.org/abs/1312.6114)
* [Glow: Flow-based Model](https://arxiv.org/abs/1807.03039)
* [Stable Diffusion 論文](https://arxiv.org/abs/2112.10752)
* [ControlNet 論文](https://arxiv.org/abs/2302.05543)

---

## 經典面試題與解法提示

1. GAN 損失函數與訓練技巧？
2. VAE 的 ELBO 推導與 KL 項作用？
3. Diffusion Model 的正向/反向過程？
4. Flow-based Model 如何保證可逆？
5. ControlNet 如何提升可控性？
6. 生成模型如何評估品質？
7. GAN mode collapse 如何解決？
8. Diffusion 推理加速方法？
9. VAE 潛在空間設計原則？
10. 如何用 Python 實作簡單 GAN/VAE？

---

## 結語

生成模型是 AI 創造力的核心。熟悉自回歸、VAE、GAN、Diffusion、Flow-based 與 ControlNet，能讓你在影像、語音、文本生成等領域發揮深度學習威力。下一章將進入正規化與訓練技巧，敬請期待！
