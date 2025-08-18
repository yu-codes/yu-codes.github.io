---
title: "大型語言模型 Fine-Tuning 完整指南：PEFT、LoRA、資料準備與目錄架構實作"
date: 2025-05-16 12:00:00 +0800
categories: [LLM]
tags: [Fine-Tune, LLM, PEFT, LoRA, Hugging Face, Transformers]
---

# 大型語言模型 Fine-Tuning 完整指南：PEFT、LoRA、資料準備與目錄架構實作

自從 ChatGPT 問世後，大型語言模型（LLM, Large Language Model）的使用越來越普及，但真正能掌握**如何針對特定任務進行微調（fine-tuning）**的工程師仍然稀少。

> 本文將從觀念入門、微調類型、資料準備，到 Hugging Face + PEFT + LoRA 的實作與專案目錄架構，帶你完整掌握 LLM 的微調能力。

---

## 🧠 為何需要微調大型語言模型？

- 🔍 LLM 的通用能力強，但對特定領域知識仍有限（例如金融、醫療、企業內部資料）
- 🎯 Fine-tuning 能讓模型專注在 **特定任務** 或 **特定語調**
- 💡 微調後可節省推論成本、提升回應一致性、支援特定格式生成

---

## 🔄 微調類型概覽（從簡到難）

| 類型                | 說明 |
|---------------------|------|
| Prompt Tuning       | 只訓練 prompt prefix，效能較差但快速輕便 |
| LoRA / Adapter Tuning | 部分參數可訓練，輕量、效率佳（推薦） |
| Full Fine-Tuning    | 訓練所有參數，效果最佳但需高算力        |

---

## 🧰 什麼是 PEFT？什麼是 LoRA？

### ✅ PEFT（Parameter-Efficient Fine-Tuning）

PEFT 是 Hugging Face 提供的工具套件，支援各種 **只調整部分參數的微調策略**，大幅降低記憶體與訓練成本。

支援方法：
- LoRA（最常用）
- Prompt Tuning
- Prefix Tuning
- AdaLoRA

官方 repo: https://github.com/huggingface/peft

---

### ✅ LoRA（Low-Rank Adaptation）

> LoRA 是一種將原始模型參數凍結，只在某些層插入低秩矩陣進行訓練的方法。

- 大幅降低 VRAM 使用量
- 可與原始模型合併或分離儲存
- 對於 LLaMA、Falcon、Mistral、GPT2 等架構特別有效

---

## 🗂️ 專案目錄架構設計（實戰推薦）

```bash
llm-finetune/
├── config/
│   └── lora_config.json          # PEFT 設定
├── data/
│   └── train.jsonl               # 輸入資料集
├── scripts/
│   └── run_finetune.py          # 主訓練腳本
├── checkpoints/
│   └── adapter_model/           # 儲存微調後參數（如 LoRA adapter）
├── logs/
│   └── training.log
├── requirements.txt
└── README.md
```

---

## 🧪 實作流程教學：使用 Hugging Face + PEFT + LoRA

### 1️⃣ 安裝必要套件

```bash
pip install transformers datasets peft accelerate
```

---

### 2️⃣ 建立訓練資料（格式為 jsonl）

```json
{"instruction": "翻譯成英文：我愛你", "input": "", "output": "I love you"}
{"instruction": "解釋什麼是 AI？", "input": "", "output": "AI 是人工智慧的縮寫..."}
```

---

### 3️⃣ 訓練腳本（簡化版本）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

model_name = "tiiuae/falcon-7b"  # 可替換成 llama2、gpt2 等

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# 載入資料（需處理 tokenization）
dataset = load_dataset("json", data_files="data/train.jsonl")["train"]

# 訓練參數
args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="logs",
    save_strategy="epoch"
)

# 開始訓練
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)
trainer.train()
```

---

## 📤 如何使用微調後的 LoRA 模型？

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
lora_model = PeftModel.from_pretrained(base_model, "checkpoints/adapter_model")

# 推論
output = lora_model.generate(...)
```

---

## 📦 模型上傳至 Hugging Face

```bash
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="checkpoints/adapter_model", repo_id="your-username/your-model")
```

---

## ✅ 實務建議與最佳實踐

* 🔒 使用 `bnb` 或 `8bit` 模型節省記憶體（bitsandbytes）
* 🚀 使用 `accelerate` 進行多卡分散訓練
* 📋 小資料集可用 `gradient checkpointing` 降低顯存消耗
* 🧪 測試 prompt 一致性、回答長度與格式準確度

---

## 📘 延伸資源推薦

* [Hugging Face PEFT 官方文件](https://huggingface.co/docs/peft/index)
* [LoRA 原始論文](https://arxiv.org/abs/2106.09685)
* [LLaMA、GPT2 等微調實例](https://github.com/huggingface/transformers/tree/main/examples)

---

## ✅ 結語

微調大型語言模型已經不再是只有大公司能做的事。透過 LoRA 與 PEFT，你可以用單張 GPU、甚至 Colab 就微調出適合自己的任務模型。希望這篇文章能讓你跨出第一步，在 LLM 的世界中打造屬於自己的智慧系統。

