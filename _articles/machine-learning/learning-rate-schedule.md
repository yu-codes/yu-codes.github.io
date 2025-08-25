---
title: "學習率策略全解析：Step, Cosine, Cyclical, Warm-up 與 LR Finder"
date: 2025-05-20 14:00:00 +0800
categories: [Machine Learning]
tags: ["learning-rate", "step-decay", "cosine-annealing"]
---

# 學習率策略全解析：Step, Cosine, Cyclical, Warm-up 與 LR Finder

學習率（Learning Rate, LR）是影響模型訓練收斂與泛化的關鍵超參數。從 Constant、Step、Exponential Decay，到 Cosine Annealing、Warm-up、Cyclical、One-Cycle Policy 與 LR Finder，這些策略能顯著提升訓練效率與最終表現。本章將深入原理、實作、面試熱點與常見誤區，幫助你全面掌握學習率調控。

---

## Constant / Step / Exponential Decay

### Constant LR

- 固定學習率，適合簡單任務或預訓練初期

### Step Decay

- 每隔固定 epoch 將學習率乘以一個係數（如 0.1）
- 常用於 ResNet、VGG 等經典架構

### Exponential Decay

- 每個 epoch 以指數方式衰減學習率

```python
import torch
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(30):
    # ...existing code...
    scheduler.step()
```

---

## Cosine Annealing & Warm-up

### Cosine Annealing

- 學習率隨 epoch 呈餘弦曲線下降，訓練後期更平滑
- 適合 Transformer、Vision Transformer 等現代架構

### Warm-up

- 訓練初期用較小學習率，逐步升高，避免梯度爆炸
- 常與 Cosine Annealing 結合

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
# Warm-up 可用自訂 scheduler 或 transformers get_linear_schedule_with_warmup
```

---

## Cyclical／One-Cycle Policy

### Cyclical Learning Rate

- 學習率在訓練過程中週期性上升下降，幫助跳出局部極小
- 代表：CyclicalLR、Triangular、Exp Range

### One-Cycle Policy

- 先升後降，訓練末期快速衰減，提升泛化
- 適合大規模訓練、超參數搜尋

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=100, epochs=10)
```

---

## LR Finder 實戰流程

- 先用指數增長學習率訓練一輪，記錄 loss 與 lr
- 找 loss 最快下降區間，選擇最佳初始學習率

```python
# 參考 fastai 或 pytorch-lr-finder 套件
# ...existing code...
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- Step/Cosine：CV/NLP 主流架構
- Warm-up：Transformer、BERT、GPT
- Cyclical/One-Cycle：超參數搜尋、快速收斂

### 常見誤區

- 學習率設太大導致發散，太小收斂慢
- 忽略 warm-up，導致初期梯度爆炸
- Cyclical/One-Cycle 未正確設置 max_lr

---

## 面試熱點與經典問題

| 主題               | 常見問題               |
| ------------------ | ---------------------- |
| Step vs Cosine     | 差異與適用場景？       |
| Warm-up            | 為何能提升穩定性？     |
| Cyclical/One-Cycle | 原理與優勢？           |
| LR Finder          | 如何選最佳學習率？     |
| 學習率策略         | 對收斂與泛化有何影響？ |

---

## 使用注意事項

* 學習率需根據模型、資料與優化器調整
* 建議用 LR Finder 或 One-Cycle Policy 自動搜尋
* Scheduler 設定需與訓練步數、epoch 匹配

---

## 延伸閱讀與資源

* [PyTorch LR Scheduler 官方文件](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
* [One-Cycle Policy 論文](https://arxiv.org/abs/1708.07120)
* [fastai LR Finder](https://docs.fast.ai/callback.schedule.html#Learner.lr_find)
* [Cyclical Learning Rates 論文](https://arxiv.org/abs/1506.01186)

---

## 經典面試題與解法提示

1. Step Decay、Cosine Annealing、Cyclical LR 原理與差異？
2. Warm-up 如何提升訓練穩定性？
3. One-Cycle Policy 的優勢？
4. LR Finder 如何選最佳學習率？
5. 學習率策略對收斂與泛化的影響？
6. 如何用 Python 實作多種 scheduler？
7. Cyclical/One-Cycle 參數設置原則？
8. Warm-up 需搭配哪些模型？
9. 學習率設錯會有什麼後果？
10. Scheduler 與 optimizer 如何協同設計？

---

## 結語

學習率策略是模型訓練成敗的關鍵。熟悉 Step、Cosine、Cyclical、Warm-up 與 LR Finder，能讓你高效訓練並提升泛化能力。下一章將進入正則化武器庫，敬請期待！