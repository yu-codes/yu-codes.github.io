---
title: "循環與序列模型全解析：RNN、LSTM、GRU、Seq2Seq 與時序預測"
date: 2025-05-19 14:00:00 +0800
categories: [Machine Learning]
tags: [RNN, LSTM, GRU, Seq2Seq, Attention, 時序預測, Teacher Forcing]
---

# 循環與序列模型全解析：RNN、LSTM、GRU、Seq2Seq 與時序預測

循環神經網路（RNN）及其變體是處理序列資料（如語音、文本、時間序列）的核心。從 Vanilla RNN 的梯度爆炸/消失，到 LSTM/GRU 的 gating 機制、Seq2Seq+Attention、Bi-directional RNN，再到時序預測技巧（Teacher Forcing、Scheduled Sampling），本章將結合理論、圖解、Python 實作、面試熱點與常見誤區，幫助你全面掌握序列建模。

---

## Vanilla RNN 與梯度爆炸/消失

### RNN 結構

- 每步輸出依賴前一狀態與當前輸入：$h_t = f(Wx_t + Uh_{t-1} + b)$
- 適合序列建模，但長序列訓練困難

### 梯度爆炸/消失

- 反向傳播時，梯度連乘導致指數級增大（爆炸）或趨近 0（消失）
- 影響長期依賴學習

#### 解法

- 梯度裁剪（clipping）、LSTM/GRU gating、殘差連接

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(5, 7, 10)  # batch, seq, feature
out, h = rnn(x)
print("RNN 輸出 shape:", out.shape)
```

---

## LSTM / GRU gating 機制

### LSTM（Long Short-Term Memory）

- 引入輸入、遺忘、輸出閘門，能記憶長期資訊
- 避免梯度消失，提升長序列學習能力

### GRU（Gated Recurrent Unit）

- 結構更簡單，合併部分閘門，訓練更快

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
out_lstm, _ = lstm(x)
out_gru, _ = gru(x)
print("LSTM 輸出 shape:", out_lstm.shape)
print("GRU 輸出 shape:", out_gru.shape)
```

---

## Seq2Seq＋Attention, Bi-directional RNN

### Seq2Seq 架構

- Encoder-Decoder 結構，常用於翻譯、摘要、對話生成
- Encoder 將輸入序列壓縮為上下文向量，Decoder 逐步生成輸出

### Attention 機制

- 讓 Decoder 每步都能關注 Encoder 不同部分，解決長序列資訊瓶頸

### Bi-directional RNN

- 同時考慮前後文，提升序列理解能力

```python
bi_rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)
out_bi, _ = bi_rnn(x)
print("Bi-RNN 輸出 shape:", out_bi.shape)
```

---

## 時序預測：Teacher Forcing、Scheduled Sampling

### Teacher Forcing

- 訓練時用真實標籤作為下一步輸入，加速收斂
- 缺點：推論時誤差累積（Exposure Bias）

### Scheduled Sampling

- 逐步減少 Teacher Forcing 機率，提升模型魯棒性

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 語音辨識、機器翻譯、對話生成、時間序列預測、NLP 任務

### 常見誤區

- 忽略梯度爆炸/消失，導致訓練失敗
- LSTM/GRU 結構選擇不當，模型過大或過簡
- Seq2Seq 未加 Attention，長序列表現差
- Teacher Forcing 機率設置不合理，推論效果不佳

---

## 面試熱點與經典問題

| 主題            | 常見問題                        |
| --------------- | ------------------------------- |
| RNN             | 為何會梯度爆炸/消失？如何解決？ |
| LSTM/GRU        | 閘門結構與優缺點？              |
| Seq2Seq         | Encoder-Decoder 如何運作？      |
| Attention       | 如何幫助長序列建模？            |
| Teacher Forcing | 有何優缺點？                    |

---

## 使用注意事項

* 長序列建議用 LSTM/GRU + Attention
* 訓練時監控梯度，必要時啟用梯度裁剪
* Teacher Forcing/Scheduled Sampling 需根據任務調整

---

## 延伸閱讀與資源

* [Deep Learning Book: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html)
* [PyTorch RNN 官方文件](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
* [Attention is All You Need 論文](https://arxiv.org/abs/1706.03762)
* [Scheduled Sampling 論文](https://arxiv.org/abs/1506.03099)

---

## 經典面試題與解法提示

1. RNN 為何會梯度爆炸/消失？數學推導？
2. LSTM/GRU 閘門結構與公式？
3. Seq2Seq 架構與應用場景？
4. Attention 機制數學原理？
5. Bi-RNN 有何優勢？
6. Teacher Forcing 與 Scheduled Sampling 差異？
7. 如何用 Python 實作 LSTM/GRU？
8. RNN 在 NLP/時序預測的應用？
9. Exposure Bias 是什麼？如何緩解？
10. Seq2Seq 未加 Attention 有何缺點？

---

## 結語

循環與序列模型是處理時序與語言資料的關鍵。熟悉 RNN、LSTM、GRU、Seq2Seq、Attention 與時序預測技巧，能讓你在 NLP、語音、金融等領域發揮深度學習實力。下一章將進入 Attention 機制拆解，敬請期待！
