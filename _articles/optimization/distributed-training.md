---
title: "分散式與大規模訓練全攻略：Data/Model/Pipeline Parallel, ZeRO, FSDP, Elastic Training"
date: 2025-05-20 19:00:00 +0800
categories: [模型訓練與優化]
tags: [分散式訓練, Data Parallel, Model Parallel, Pipeline Parallel, ZeRO, FSDP, Megatron-LM, Elastic Training, Checkpoint Sharding]
---

# 分散式與大規模訓練全攻略：Data/Model/Pipeline Parallel, ZeRO, FSDP, Elastic Training

隨著模型規模不斷擴大，單機訓練已無法滿足需求。分散式與大規模訓練技術（Data/Model/Pipeline Parallel, ZeRO, FSDP, Megatron-LM, Elastic Training）成為現代 AI 訓練的核心。本章將深入原理、架構圖解、PyTorch/Hugging Face 實作、資源管理、面試熱點與常見誤區，幫助你掌握大模型訓練的關鍵技術。

---

## Data / Model / Pipeline Parallelism

### Data Parallel

- 每台機器訓練同一模型，分配不同 mini-batch，梯度同步
- 適合大資料集、模型較小

### Model Parallel

- 將模型切分到多台機器/卡上，適合超大模型（如 GPT-3）
- 需手動劃分模型結構

### Pipeline Parallel

- 將模型分為多個 stage，資料流經各 stage，提升硬體利用率
- 常與 Data/Model Parallel 結合

```python
import torch.distributed as dist
# PyTorch DDP 範例
dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## ZeRO Stage 1-3, FSDP, Megatron-LM

### ZeRO (Zero Redundancy Optimizer)

- Stage 1：優化器狀態分散
- Stage 2：加上梯度分散
- Stage 3：參數分散，極致節省記憶體
- 適合超大模型（如 GPT-3、T5）

### FSDP（Fully Sharded Data Parallel）

- PyTorch 官方，參數/梯度/優化器狀態全分片
- 支援動態層、低記憶體佔用

### Megatron-LM

- NVIDIA 開源，支援 Model/Pipeline Parallel，訓練千億參數模型

---

## Gradient Accumulation vs. Micro-Batch

- Gradient Accumulation：多個 mini-batch 累積梯度再更新，等效大 batch
- Micro-Batch：每個 pipeline stage 處理小 batch，提升吞吐

---

## Checkpoint Sharding & Elastic Training

### Checkpoint Sharding

- 將模型權重分片存儲，減少單機 I/O 壓力，加速恢復

### Elastic Training

- 支援動態增減節點，提升容錯與資源利用率
- PyTorch Elastic、DeepSpeed Elastic

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 超大模型訓練（GPT-3、T5、Llama）、企業內部大規模預訓練、雲端分散式訓練

### 常見誤區

- Data Parallel 過多導致通訊瓶頸
- Model/Pipeline Parallel 劃分不均，造成資源閒置
- ZeRO/FSDP 配置錯誤導致 OOM
- Checkpoint Sharding 未測試恢復流程

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Data/Model/Pipeline | 差異與適用場景？ |
| ZeRO         | Stage 1-3 有何不同？ |
| FSDP         | 與 DDP/ZeRO 差異？ |
| Gradient Accumulation | 作用與實作？ |
| Elastic Training | 如何提升容錯？ |

---

## 使用注意事項

* 分散式訓練需正確設置通訊後端與網路
* ZeRO/FSDP 需根據模型大小與硬體調整 stage
* Checkpoint Sharding/Elastic Training 建議多做容錯測試

---

## 延伸閱讀與資源

* [PyTorch Distributed 官方文件](https://pytorch.org/docs/stable/distributed.html)
* [DeepSpeed ZeRO 論文](https://arxiv.org/abs/1910.02054)
* [FSDP 官方文件](https://pytorch.org/docs/stable/fsdp.html)
* [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
* [PyTorch Elastic 官方文件](https://pytorch.org/docs/stable/elastic.html)

---

## 經典面試題與解法提示

1. Data/Model/Pipeline Parallel 差異與組合？
2. ZeRO Stage 1-3 原理與應用？
3. FSDP 與 ZeRO/ DDP 差異？
4. Gradient Accumulation 實作細節？
5. Checkpoint Sharding 如何加速恢復？
6. Elastic Training 如何提升容錯？
7. Megatron-LM 支援哪些並行方式？
8. 分散式訓練常見瓶頸與解法？
9. 如何用 Python 實作 DDP/FSDP？
10. 分散式訓練的資源管理挑戰？

---

## 結語

分散式與大規模訓練是現代 AI 的基石。熟悉 Data/Model/Pipeline Parallel、ZeRO、FSDP、Elastic Training 等技術，能讓你高效訓練超大模型並在面試中展現專業素養。下一章將進入超參數尋優進階，敬請期待！
