---
title: "Transformer 家族全解析：結構、位置編碼、複雜度與主流模型比較"
date: 2025-05-19 16:00:00 +0800
categories: [Machine Learning]
tags: [Transformer, Encoder, Decoder, Positional Encoding, BERT, GPT, DeiT, Swin, 計算複雜度]
---

# Transformer 家族全解析：結構、位置編碼、複雜度與主流模型比較

Transformer 架構徹底改變了深度學習格局，從 NLP 到 Vision、語音、推薦系統皆有應用。本章將深入 Encoder/Decoder Block 結構、位置編碼（Sinusoid, ALiBi, RoPE）、參數膨脹與計算複雜度、以及 BERT、GPT、DeiT、Swin 等主流模型差異，結合理論、圖解、Python 實作、面試熱點與常見誤區，幫助你全面掌握 Transformer 家族。

---

## Encoder / Decoder Block 結構

### Encoder Block

- 多層 Self-Attention + Feedforward + 殘差連接 + LayerNorm
- 輸入序列同時處理，捕捉全局依賴

### Decoder Block

- Masked Self-Attention（防洩漏未來資訊）+ Encoder-Decoder Attention + Feedforward
- 適合自回歸生成（如翻譯、摘要）

```python
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)
```

---

## Positional Encoding：Sinusoid, ALiBi, RoPE

### Sinusoid Encoding

- 用正弦/餘弦函數給每個位置唯一編碼，無需學習參數
- 公式：$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$

### ALiBi（Attention with Linear Biases）

- 用線性偏置調整注意力分數，提升長序列泛化

### RoPE（Rotary Position Embedding）

- 用複數旋轉方式編碼位置，提升長距依賴建模

---

## 參數膨脹 vs. 計算複雜度（Quadratic → Linear 改進）

- 標準 Self-Attention 計算複雜度為 $O(n^2)$，n 為序列長度
- 長序列改進：Sparse Attention、Performer、Longformer、FlashAttention 等，複雜度降至 $O(n)$ 或 $O(n \log n)$

---

## BERT、GPT、DeiT、Swin 頂層差異

| 模型 | 架構特點                | 任務        | 代表應用       |
| ---- | ----------------------- | ----------- | -------------- |
| BERT | 雙向 Encoder            | 預訓練+微調 | NLP 理解、問答 |
| GPT  | 單向 Decoder            | 自回歸生成  | NLP 生成、對話 |
| DeiT | ViT 改進，圖像分類      | Encoder     | Vision         |
| Swin | 局部窗口+移動，層次結構 | Encoder     | 影像分割、檢測 |

---

## Python 實作：位置編碼

```python
import torch
import math

def sinusoid_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

print("Sinusoid Encoding:", sinusoid_encoding(5, 8))
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- NLP（BERT、GPT）、Vision（ViT、DeiT、Swin）、語音、推薦系統

### 常見誤區

- 忽略位置編碼，導致序列資訊丟失
- 長序列未優化注意力，計算資源爆炸
- Encoder/Decoder Block 混用，導致模型無法訓練

---

## 面試熱點與經典問題

| 主題                  | 常見問題                      |
| --------------------- | ----------------------------- |
| Encoder/Decoder       | 結構差異與應用？              |
| Positional Encoding   | 為何需要？有何種類？          |
| Self-Attention 複雜度 | 如何優化？                    |
| BERT vs GPT           | 架構與任務差異？              |
| Swin/DeiT             | Vision Transformer 有何創新？ |

---

## 使用注意事項

* 長序列建議用線性/稀疏注意力
* 位置編碼需與模型架構匹配
* Encoder/Decoder Block 須根據任務選擇

---

## 延伸閱讀與資源

* [Attention is All You Need 論文](https://arxiv.org/abs/1706.03762)
* [BERT 論文](https://arxiv.org/abs/1810.04805)
* [GPT 論文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [Swin Transformer 論文](https://arxiv.org/abs/2103.14030)
* [FlashAttention 論文](https://arxiv.org/abs/2205.14135)

---

## 經典面試題與解法提示

1. Encoder/Decoder Block 結構與差異？
2. Sinusoid/ALiBi/RoPE 位置編碼原理？
3. Self-Attention 複雜度如何優化？
4. BERT 與 GPT 架構與訓練差異？
5. DeiT/Swin 在 Vision Transformer 的創新？
6. 長序列 Transformer 如何設計？
7. 位置編碼缺失會有什麼問題？
8. 如何用 Python 實作位置編碼？
9. Encoder/Decoder Block 混用會有什麼後果？
10. FlashAttention 有何優勢？

---

## 結語

Transformer 家族是現代深度學習的核心。熟悉 Encoder/Decoder 結構、位置編碼、計算複雜度與主流模型差異，能讓你在 NLP、Vision、生成模型等領域發揮 Transformer 強大威力。下一章將進入預訓練策略與微調，敬請期待！
