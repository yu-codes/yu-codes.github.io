---
title: "Attention 機制拆解：Scaled Dot-Product、Multi-Head、QKV 幾何與 Masking"
date: 2025-05-19 15:00:00 +0800
categories: [深度學習]
tags: [Attention, Scaled Dot-Product, Multi-Head, Self-Attention, QKV, Masking]
---

# Attention 機制拆解：Scaled Dot-Product、Multi-Head、QKV 幾何與 Masking

Attention 機制是現代深度學習（尤其是 NLP 和 Vision Transformer）的核心。從 Scaled Dot-Product Attention、Multi-Head & Self-Attention，到 Q/K/V 幾何意義與 Masking 技巧，本章將結合理論推導、圖解、Python 實作、面試熱點與常見誤區，幫助你徹底掌握 Attention。

---

## Scaled Dot-Product Attention

### 數學公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$（Query）、$K$（Key）、$V$（Value）為輸入矩陣
- $d_k$ 為 Key 維度，縮放避免梯度消失/爆炸

### Python 實作

```python
import torch
import torch.nn.functional as F

Q = torch.randn(2, 4, 8)  # batch, seq, d_k
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (8 ** 0.5)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, V)
print("Attention 輸出 shape:", output.shape)
```

---

## Multi-Head & Self-Attention 可視化

### Multi-Head Attention

- 多組 Q/K/V 並行計算，捕捉不同子空間資訊
- 輸出拼接後再線性變換，提升模型表達力

### Self-Attention

- Q/K/V 皆來自同一序列，捕捉序列內部依賴
- 可視化：每個 token 對其他 token 的關注分數

---

## Q/K/V 的幾何意義

- Q：查詢向量，代表當前 token 想「問」什麼
- K：鍵向量，代表每個 token 能「回答」什麼
- V：值向量，攜帶實際資訊
- Attention = 查詢與所有鍵的相似度加權所有值

---

## Masking：Padding vs. Causal

### Padding Mask

- 遮蔽填充（padding）位置，避免影響注意力分數
- 常用於批次處理長度不一的序列

### Causal Mask

- 遮蔽未來資訊，確保自回歸生成時僅依賴過去
- 常用於語言模型（如 GPT）

```python
seq_len = 5
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = torch.randn(seq_len, seq_len)
scores.masked_fill_(mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
print("Causal Masked Attention:", attn)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- NLP（翻譯、摘要、問答）、Vision Transformer、語音辨識、推薦系統

### 常見誤區

- 忽略縮放因子，導致梯度不穩
- Masking 實作錯誤，導致洩漏未來資訊
- Multi-Head 輸出未正確拼接

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Scaled Dot-Product | 為何要縮放？ |
| Multi-Head   | 有何優勢？ |
| Q/K/V        | 幾何意義？ |
| Masking      | Padding vs Causal 差異？ |
| Self-Attention | 如何捕捉長距依賴？ |

---

## 使用注意事項

* 注意 Q/K/V 維度一致性
* Masking 必須正確設計，避免資訊洩漏
* 多頭注意力需拼接後再線性變換

---

## 延伸閱讀與資源

* [Attention is All You Need 論文](https://arxiv.org/abs/1706.03762)
* [PyTorch MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 經典面試題與解法提示

1. Scaled Dot-Product Attention 數學推導？
2. Multi-Head Attention 如何提升表達力？
3. Q/K/V 的幾何直覺？
4. Self-Attention 與傳統 RNN 差異？
5. Masking 有哪些類型？如何實作？
6. 如何用 Python 實作簡單 Attention？
7. 為何要做縮放？有何數值意義？
8. Multi-Head 拼接與線性變換細節？
9. Attention 如何捕捉長距依賴？
10. Masking 實作錯誤會有什麼後果？

---

## 結語

Attention 機制是深度學習的革命。熟悉 Scaled Dot-Product、Multi-Head、QKV 幾何與 Masking，能讓你在 NLP、Vision、生成模型等領域發揮 Transformer 強大威力。下一章將進入 Transformer 家族，敬請期待！
