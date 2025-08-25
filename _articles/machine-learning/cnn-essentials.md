---
title: "卷積網路精要：CNN 結構、演進與應用全解析"
date: 2025-05-19 13:00:00 +0800
categories: [Machine Learning]
tags: ["cnn", "convolution", "pooling"]
---

# 卷積網路精要：CNN 結構、演進與應用全解析

卷積神經網路（CNN）是現代深度學習的基石，廣泛應用於影像辨識、語音處理、NLP 等領域。本章將從卷積與池化的直覺出發，帶你理解經典結構（AlexNet、VGG、ResNet、EfficientNet）、各種 Block 差異，以及 CNN 在 NLP/時序資料的變體，結合理論、圖解、Python 實作與面試熱點，幫助你全面掌握 CNN。

---

## 卷積、池化、轉置卷積直覺

### 卷積（Convolution）

- 用小型權重核（filter）滑動提取區域特徵，參數共享、局部連接。
- 優點：大幅減少參數、提升平移不變性。

### 池化（Pooling）

- 降低特徵圖維度，提升模型魯棒性。
- 常見：Max Pooling、Average Pooling。

### 轉置卷積（Transposed Convolution）

- 用於上採樣（如生成模型、語意分割），將特徵圖放大。

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2)
deconv = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
x = torch.randn(1, 3, 32, 32)
y = pool(conv(x))
z = deconv(y)
print("卷積後 shape:", y.shape, "轉置卷積後 shape:", z.shape)
```

---

## 經典結構演進：AlexNet → VGG → ResNet → EfficientNet

### AlexNet

- 2012 年 ImageNet 冠軍，深度 CNN 崛起。
- 多層卷積+ReLU+Dropout，首次用 GPU 訓練。

### VGG

- 結構簡單，堆疊多個 3x3 卷積層。
- 參數量大，易於理解與遷移。

### ResNet

- 引入殘差連接（Residual Connection），解決深層網路梯度消失。
- 可訓練超過 100 層，ImageNet 長青樹。

### EfficientNet

- 用複合縮放（Compound Scaling）同時調整深度、寬度、解析度。
- 參數效率高，效能領先。

---

## Residual / Dense / Inception Block 比較

| Block 類型 | 結構特點   | 優點         | 代表網路            |
| ---------- | ---------- | ------------ | ------------------- |
| Residual   | 跳接殘差   | 解梯度消失   | ResNet              |
| Dense      | 全層連接   | 特徵重用     | DenseNet            |
| Inception  | 多尺度卷積 | 捕捉多種特徵 | GoogLeNet/Inception |

---

## CNN 在 NLP / 時序資料的變體

### TCN（Temporal Convolutional Network）

- 用 1D 卷積處理序列，支援長距依賴。
- 應用：語音、時序預測。

### WaveNet

- Google 提出，生成語音波形，採用因果卷積與殘差結構。

---

## Python 實作：簡單 CNN

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8*14*14, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
print(model)
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 影像辨識、物件偵測、語音處理、NLP（文本分類、情感分析）、時序預測

### 常見誤區

- 忽略 padding/stride 設定，導致特徵圖尺寸錯誤
- 轉置卷積未正確初始化，產生棋盤效應
- 過度堆疊卷積層，未加殘差連接導致梯度消失

---

## 面試熱點與經典問題

| 主題         | 常見問題                  |
| ------------ | ------------------------- |
| 卷積         | 參數共享有何好處？        |
| 池化         | Max vs Avg Pooling 差異？ |
| ResNet       | 殘差連接如何幫助訓練？    |
| EfficientNet | 複合縮放原理？            |
| CNN 變體     | TCN/WaveNet 適用場景？    |

---

## 使用注意事項

* 卷積層後建議加 BatchNorm、激活函數
* 池化層可減少過擬合，但過度會損失細節
* 深層網路建議用殘差/稠密連接提升訓練穩定性

---

## 延伸閱讀與資源

* [Deep Learning Book: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
* [PyTorch CNN 官方文件](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
* [EfficientNet 論文](https://arxiv.org/abs/1905.11946)
* [WaveNet 論文](https://arxiv.org/abs/1609.03499)

---

## 經典面試題與解法提示

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

## 結語

CNN 是深度學習的核心。熟悉卷積、池化、經典結構與變體，能讓你在影像、語音、NLP 等多領域發揮深度學習威力。下一章將進入循環與序列模型，敬請期待！
