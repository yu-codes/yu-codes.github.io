---
title: "訓練監控與 Debug：TensorBoard、Loss 爆炸/消失、Learning Curve 判讀與單元測試"
date: 2025-05-20 21:00:00 +0800
categories: [模型訓練與優化]
tags: [訓練監控, TensorBoard, Weights & Biases, Loss Exploding, Loss Vanishing, Learning Curve, Validation Gap, 單元測試]
---

# 訓練監控與 Debug：TensorBoard、Loss 爆炸/消失、Learning Curve 判讀與單元測試

模型訓練過程中，監控與 Debug 是確保收斂、提升泛化與快速定位問題的關鍵。本章將深入 TensorBoard/Weights & Biases 指標追蹤、Loss 爆炸/消失原因排查、Learning Curve 與 Validation Gap 判讀、Forward/Backward 單元測試，結合理論、實作、面試熱點與常見誤區，幫助你打造穩健的訓練流程。

---

## TensorBoard / Weights & Biases 指標追蹤

- TensorBoard：可視化 loss、accuracy、learning rate、參數分布等
- Weights & Biases（wandb）：雲端追蹤、團隊協作、超參數搜尋

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for epoch in range(10):
    # ...existing code...
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.close()
```

---

## Loss Exploding / Vanishing 原因排查

### Loss 爆炸（Exploding）

- 常見於深層網路、RNN，梯度連乘導致數值爆炸
- 解法：Gradient Clipping、初始化調整、學習率降低

### Loss 消失（Vanishing）

- 梯度趨近 0，參數無法更新
- 解法：選擇合適激活函數（ReLU/LeakyReLU）、初始化優化、殘差連接

---

## Learning Curve & Validation Gap 判讀

- Learning Curve：訓練/驗證 loss 隨 epoch 變化
- Validation Gap：訓練 loss 遠低於驗證 loss，常見於過擬合
- Loss 不降：學習率過小、資料/標籤錯誤、模型容量不足

```python
import matplotlib.pyplot as plt

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.title('Learning Curve'); plt.show()
```

---

## Unit Test Your Forward / Backward

- 單元測試 Forward/Backward，確保模型與 loss 可微、梯度正確
- 可用 torch.autograd.gradcheck、assert 條件

```python
import torch

def test_forward_backward(model, x, y):
    x = x.requires_grad_()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    assert all([p.grad is not None for p in model.parameters() if p.requires_grad]), "Gradient missing!"

# gradcheck 範例
from torch.autograd import gradcheck
# gradcheck(model, (input,))  # 需 double precision
```

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 深度學習訓練監控、模型調參、異常排查、團隊協作

### 常見誤區

- 只看 loss，忽略 accuracy、learning rate 等指標
- Loss 爆炸未及時啟用 Gradient Clipping
- Validation Gap 未調整正則化/資料增強
- 未做單元測試，導致訓練異常難以定位

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| TensorBoard  | 如何追蹤與可視化指標？ |
| Loss 爆炸/消失 | 原因與解法？ |
| Learning Curve | 如何判讀過擬合/欠擬合？ |
| Validation Gap | 產生原因與調整方法？ |
| 單元測試     | 如何驗證模型可微與梯度正確？ |

---

## 使用注意事項

* 訓練監控建議多指標聯合追蹤
* Loss 爆炸/消失需及時調參與 debug
* 單元測試可提前發現模型設計問題

---

## 延伸閱讀與資源

* [TensorBoard 官方文件](https://pytorch.org/docs/stable/tensorboard.html)
* [Weights & Biases 官方文件](https://docs.wandb.ai/)
* [PyTorch gradcheck](https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck)
* [深度學習 Loss 爆炸/消失解法](https://www.deeplearningbook.org/contents/numerical.html)

---

## 經典面試題與解法提示

1. TensorBoard/Weights & Biases 如何追蹤訓練指標？
2. Loss 爆炸/消失的數學原因？
3. Learning Curve 如何判斷過擬合/欠擬合？
4. Validation Gap 如何調整？
5. 如何用 Python 單元測試 Forward/Backward？
6. Loss 不降常見原因？
7. Gradient Clipping 何時啟用？
8. 多指標監控的好處？
9. 單元測試如何設計？
10. 訓練監控與 Debug 的最佳實踐？

---

## 結語

訓練監控與 Debug 是模型穩健訓練的保障。熟悉 TensorBoard、Loss 爆炸/消失排查、Learning Curve 判讀與單元測試，能讓你高效定位問題並提升模型表現。下一章將進入課程學習與自監督，敬請期待！