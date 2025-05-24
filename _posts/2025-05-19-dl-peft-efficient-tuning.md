---
title: "參數高效微調（PEFT）全攻略：LoRA、Adapter、Prefix、實務選型與調參"
date: 2025-05-19 18:00:00 +0800
categories: [深度學習]
tags: [PEFT, LoRA, QLoRA, Adapter, Prefix Tuning, P-Tuning, 低秩更新, 量化, 記憶體優化]
---

# 參數高效微調（PEFT）全攻略：LoRA、Adapter、Prefix、實務選型與調參

大模型時代，參數高效微調（PEFT, Parameter-Efficient Fine-Tuning）成為主流。從 LoRA/QLoRA 的低秩更新與量化，到 Adapter、Prefix/P-Tuning v2，這些方法能大幅降低記憶體需求、加速訓練並提升推論友善度。本章將深入原理、實作、選型、調參、面試熱點與常見誤區，幫助你高效微調大模型。

---

## LoRA / QLoRA：低秩更新與量化預處理

### LoRA（Low-Rank Adaptation）

- 僅訓練少量低秩矩陣，主模型參數凍結
- 優點：記憶體佔用低、推論快、易於部署

### QLoRA

- LoRA 結合 4-bit 量化，進一步壓縮記憶體
- 適合消費級 GPU 微調大模型

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
print("LoRA 參數量:", sum(p.numel() for n, p in model.named_parameters() if "lora" in n))
```

---

## Adapter / Prefix / P-Tuning v2

### Adapter

- 在每層插入小型適配器模組，僅訓練 Adapter 參數
- 優點：多任務切換方便，主模型共享

### Prefix Tuning

- 為每個任務學習一組可訓練前綴向量，主模型參數不變
- 適合生成任務、低資源場景

### P-Tuning v2

- 將 Prefix Tuning 擴展到深層 Transformer，提升表現

---

## 差異比較：記憶體佔用、推論友善度

| 方法         | 記憶體佔用 | 推論友善度 | 適用場景         |
|--------------|------------|------------|------------------|
| LoRA/QLoRA   | 極低       | 高         | 通用、消費級 GPU |
| Adapter      | 低         | 高         | 多任務           |
| Prefix/P-Tuning | 低      | 高         | 生成、NLP        |

---

## 實務：選 Rank、α、Target Modules

- Rank（r）：控制低秩矩陣大小，r 越大表現越好但佔用提升
- α（lora_alpha）：調整 LoRA 輸出縮放，需實驗調參
- Target Modules：選擇插入 LoRA/Adapter 的層（如 q_proj, v_proj）

---

## Python 實作：QLoRA 微調

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=bnb_config)
config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
# ...後續 Trainer 設定同 Fine-tune ...
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 大模型微調、企業內部知識庫、個人化應用、低資源設備部署

### 常見誤區

- Rank 設太小導致表現差，太大失去高效優勢
- Target Modules 選擇不當，影響效果
- 量化後未檢查數值穩定性
- Adapter/Prefix 模型未正確切換，導致推論錯誤

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| LoRA/QLoRA   | 原理、優缺點、適用場景？ |
| Adapter      | 結構與多任務優勢？ |
| Prefix/P-Tuning | 如何運作？適用哪些任務？ |
| Rank/α       | 如何選擇？有何 trade-off？ |
| QLoRA        | 量化有何風險？ |

---

## 使用注意事項

* 微調前確認模型支援 PEFT
* Rank/α/Target Modules 需多做實驗調參
* 量化模型需檢查推論精度與穩定性

---

## 延伸閱讀與資源

* [PEFT 官方文件](https://huggingface.co/docs/peft/index)
* [LoRA 論文](https://arxiv.org/abs/2106.09685)
* [QLoRA 論文](https://arxiv.org/abs/2305.14314)
* [AdapterHub](https://adapterhub.ml/)
* [Prefix Tuning 論文](https://arxiv.org/abs/2001.07676)

---

## 經典面試題與解法提示

1. LoRA/QLoRA 的數學原理與優缺點？
2. Adapter 結構與多任務切換？
3. Prefix/P-Tuning 適用場景與限制？
4. Rank/α 如何選擇與調參？
5. QLoRA 量化的數值風險？
6. 如何用 Python 實作 LoRA/QLoRA？
7. PEFT 與全參數微調的比較？
8. 多任務微調如何設計 Adapter？
9. 量化模型推論時需注意什麼？
10. PEFT 技巧在產業落地的挑戰？

---

## 結語

PEFT 技巧讓大模型微調變得高效可行。熟悉 LoRA、QLoRA、Adapter、Prefix/P-Tuning 原理與實作，能讓你在大模型應用、個人化、低資源部署等場景發揮深度學習威力。下一章將進入生成模型百花齊放，敬請期待！
