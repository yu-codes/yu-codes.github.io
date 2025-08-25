---
title: "Fairness, Robustness & 安全：對抗訓練、可重現性與模型安全全攻略"
date: 2025-05-20 22:00:00 +0800
categories: [Machine Learning]
tags: [公平性, Robustness, Adversarial Training, FGSM, PGD, Noise Injection, Feature Smoothing, 可重現性, Seed, Checkpoint]
---

# Fairness, Robustness & 安全：對抗訓練、可重現性與模型安全全攻略

隨著 AI 應用於關鍵領域，模型的公平性、魯棒性與安全性成為不可忽視的議題。本章將深入對抗訓練（Adversarial Training）、FGSM/PGD 防禦、Noise Injection、Feature Smoothing、可重現性（Seed、Determinism、Checkpoint 管理）等主題，結合理論、實作、面試熱點與常見誤區，幫助你打造更可靠的 AI 系統。

---

## Adversarial Training, FGSM / PGD 防禦

### 對抗樣本（Adversarial Examples）

- 對輸入加微小擾動即可欺騙模型，威脅安全性

### FGSM（Fast Gradient Sign Method）

- 單步梯度攻擊，$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))$

### PGD（Projected Gradient Descent）

- 多步 FGSM，效果更強

### 對抗訓練

- 在訓練時加入對抗樣本，提升模型魯棒性

```python
import torch

def fgsm_attack(model, x, y, epsilon=0.1):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = loss_fn(output, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()
```

---

## Noise Injection, Feature Smoothing

### Noise Injection

- 在訓練時對輸入或權重加噪聲，提升魯棒性

### Feature Smoothing

- 在損失中加入特徵平滑項，抑制過度擬合

---

## 可重現性：Seed, Determinism, Checkpoint 管理

### Seed & Determinism

- 設定隨機種子，固定資料分割、初始化，確保結果可重現

```python
import torch, numpy as np, random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Checkpoint 管理

- 定期保存模型、優化器狀態，支持恢復與審計
- 建議保存多份 checkpoint，防止損壞

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融、醫療、司法等高風險領域
- 需防範對抗攻擊、保證公平性與可重現性

### 常見誤區

- 只測試標準資料，忽略對抗樣本
- 未設 seed，導致結果不穩
- Checkpoint 管理混亂，無法追溯模型狀態

---

## 面試熱點與經典問題

| 主題              | 常見問題                      |
| ----------------- | ----------------------------- |
| 對抗訓練          | 原理與實作？                  |
| FGSM/PGD          | 差異與適用場景？              |
| Noise Injection   | 如何提升魯棒性？              |
| 可重現性          | 如何設計 seed 與 checkpoint？ |
| Feature Smoothing | 有何數學原理？                |

---

## 使用注意事項

* 對抗訓練需平衡精度與魯棒性
* Seed/Determinism 設置需覆蓋所有隨機源
* Checkpoint 建議保存多份並記錄超參數

---

## 延伸閱讀與資源

* [Adversarial Training 論文](https://arxiv.org/abs/1412.6572)
* [PGD 論文](https://arxiv.org/abs/1706.06083)
* [Noise Injection 論文](https://arxiv.org/abs/1706.02515)
* [PyTorch reproducibility 官方文件](https://pytorch.org/docs/stable/notes/randomness.html)

---

## 經典面試題與解法提示

1. FGSM/PGD 原理與數學推導？
2. 對抗訓練如何提升魯棒性？
3. Noise Injection/Feature Smoothing 應用場景？
4. 如何設計可重現性流程？
5. Checkpoint 管理的最佳實踐？
6. 對抗樣本如何生成？
7. Seed/Determinism 設置細節？
8. 對抗訓練與標準訓練的 trade-off？
9. 如何用 Python 實作 FGSM？
10. Feature Smoothing 數學原理與實作？

---

## 結語

Fairness、Robustness 與安全是 AI 實務落地的底線。熟悉對抗訓練、Noise Injection、可重現性與 checkpoint 管理，能讓你打造更可靠的 AI 系統並在面試中展現專業素養。下一章將進入經典挑戰題庫，敬請期待！
