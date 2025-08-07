---
title: "深度學習挑戰題庫：12 章經典口試與白板題"
date: 2025-05-19 23:00:00 +0800
categories: [Machine Learning]
tags: [面試題, 深度學習, 白板題, 口試, 解題技巧]
---

# 深度學習挑戰題庫：12 章經典口試與白板題

本章彙整前述 11 章深度學習主題的經典面試題，每章精選 10-15 題，涵蓋理論推導、實作、直覺解釋與白板題。每題附上解法提示與常見誤區，幫助你在面試與實戰中脫穎而出。

---

## DL1 前菜：感知機 → MLP

1. 感知機學習規則與收斂條件？
2. 為何感知機無法解 XOR 問題？
3. Sigmoid、Tanh、ReLU、GELU 優缺點比較？
4. MLP 為何能逼近任意連續函數？
5. 如何計算 MLP 參數量？
6. 激活函數選擇對訓練有何影響？
7. 如何用 Python 實作簡單感知機/MLP？
8. MLP 過擬合時有哪些解法？
9. ReLU 死神經元問題如何緩解？
10. GELU 為何在 Transformer 中表現佳？

---

## DL2 卷積網路 (CNN) 精要

1. 卷積層參數量如何計算？
2. 池化層有何作用？何時用 Max/Avg？
3. ResNet 殘差連接數學推導？
4. EfficientNet 複合縮放如何設計？
5. Inception Block 為何能捕捉多尺度特徵？
6. TCN 與 RNN 差異？
7. 轉置卷積常見問題與解法？
8. CNN 在 NLP 的應用有哪些？
9. 如何用 Python 實作簡單 CNN？
10. DenseNet 為何特徵重用？

---

## DL3 循環 & 序列模型 (RNN)

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

## DL4 Attention 機制拆解

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

## DL5 Transformer 家族

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

## DL6 預訓練策略 & 微調

1. 預訓練與從零訓練的收斂差異？
2. Feature-based、Fine-tune、Prompt-tune 差異？
3. Llama-2/3 微調的資源瓶頸？
4. AMP、梯度累積的原理與作用？
5. 如何選擇微調策略？
6. 微調過程如何避免過擬合？
7. Prompt-tune 適合哪些場景？
8. 如何用 Python 微調 Llama？
9. 微調資料格式化注意事項？
10. PEFT 技巧有哪些？

---

## DL7 參數高效微調 (PEFT)

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

## DL8 生成模型百花齊放

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

## DL9 正規化 & 訓練技巧

1. BatchNorm、LayerNorm、GroupNorm、RMSNorm 差異？
2. 殘差連接如何幫助深層網路訓練？
3. Dropout 的數學原理與推論差異？
4. Stochastic Depth/DropPath 適用場景？
5. Label Smoothing 有何優缺點？
6. MixUp/CutMix 如何提升泛化？
7. 如何用 Python 實作正規化與資料增強？
8. BatchNorm 在推論時如何運作？
9. Dropout/MixUp 過度使用會有什麼問題？
10. 正規化與訓練技巧如何組合應用？

---

## DL10 加速 & 壓縮實戰

1. AMP 的原理與數值風險？
2. 知識蒸餾如何設計 Teacher/Student？
3. QAT 與 Post-training Quantization 差異？
4. TensorRT/ONNX 如何加速推論？
5. Flash-Attention 計算複雜度與優勢？
6. Edge AI 部署常見挑戰？
7. Streaming 生成的應用與限制？
8. 如何用 Python 實作 AMP/QAT？
9. 量化後精度下降如何調整？
10. 推論優化與壓縮技術如何組合應用？

---

## DL11 多模態 & 視覺語言

1. CLIP 的對比學習損失如何設計？
2. BLIP-2 架構與 Q-Former 作用？
3. LLaVA 如何實現圖文對話？
4. Cross-Attention 權重共享的優缺點？
5. 多模態模型如何對齊影像與文本特徵？
6. CLIP/BLIP-2/LLaVA 輸入格式設計？
7. 多模態應用的資料挑戰？
8. 如何用 Python 實作 Cross-Attention？
9. 權重共享會有什麼風險？
10. 多模態模型在醫療/推薦的應用？

---

## 解題技巧與常見誤區

- **計算題**：先寫公式再帶數字，避免粗心。
- **推導題**：分步驟寫清楚，標明假設。
- **直覺題**：用圖解、生活例子輔助說明。
- **實作題**：熟悉 numpy、torch、transformers 等常用 API。
- **常見誤區**：混淆定義、忽略假設、過度依賴單一指標。

---

## 結語

本題庫涵蓋深度學習經典面試題與解法直覺。建議每題都動手推導、實作與解釋，並多練習口頭表達。祝你面試順利、學習愉快！
