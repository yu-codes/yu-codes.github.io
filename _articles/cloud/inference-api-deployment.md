---
title: "推論 API 佈署策略：Blue-Green、Canary、KServe、Triton、EKS+Spot"
date: 2025-05-23 19:00:00 +0800
categories: [Cloud Deployment & Services]
tags: [推論 API, Blue-Green, Canary, Shadow, KServe, Triton, TensorRT-LLM, EKS, Spot, DaemonSet, PriorityClass, Checkpoint]
---

# 推論 API 佈署策略：Blue-Green、Canary、KServe、Triton、EKS+Spot

現代 AI 推論服務需兼顧高可用、低延遲與彈性擴縮。從 Blue-Green/Canary/Shadow 部署策略，到 KServe、Triton、TensorRT-LLM 等推論框架，再到 EKS+Spot 的 DaemonSet Drain、PriorityClass、Checkpoint 恢復設計，本章將結合理論、實戰、面試熱點與常見誤區，幫助你打造高效穩健的推論 API 平台。

---

## Blue-Green / Canary / Shadow 部署

### Blue-Green 部署

- 維護兩套環境（藍/綠），流量切換，快速回滾
- 適合大規模升級、零停機部署

### Canary 部署

- 新版本先導入部分流量，逐步擴大
- 監控指標，異常自動回滾

### Shadow 部署

- 新模型僅接收流量，不回應用戶，觀察行為差異
- 適合新模型驗證、風險控制

---

## KServe / Triton / TensorRT-LLM

### KServe

- K8s 原生推論框架，支援多模型、多框架（PyTorch、TF、SKLearn、XGBoost）
- 支援自動擴縮、A/B/Canary、GPU 調度、ModelMesh

### Triton Inference Server

- NVIDIA 高效推論框架，支援多模型、動態批次、TensorRT 加速
- 適合 GPU/多模型高吞吐應用

### TensorRT-LLM

- 專為大語言模型（LLM）優化的推論框架
- 支援分布式推理、低延遲、FP8/INT8 加速

---

## EKS + Spot：DaemonSet Drain、PriorityClass、Checkpoint 恢復

### EKS + Spot 策略

- 利用 Spot 節點降低成本，需設計容錯與自動恢復
- DaemonSet Drain：Spot 回收時自動驅逐 DaemonSet，釋放資源
- PriorityClass：關鍵 Pod 設高優先權，保證資源分配
- Checkpoint 恢復：推論狀態/快取定期保存，Spot 中斷自動恢復

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: inference-critical
value: 1000000
globalDefault: false
description: "Critical inference pods"
```

---

## 設計實戰與最佳實踐

- 推論服務建議用 KServe/Triton，結合 GPU 調度與自動擴縮
- 部署建議用 Blue-Green/Canary，結合監控與自動回滾
- EKS+Spot 建議設 PriorityClass、Checkpoint、自動恢復
- Shadow 部署適合新模型驗證，避免直接影響用戶

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- LLM 推論、推薦系統、即時 API、金融/醫療 AI 服務

### 常見誤區

- Canary 部署未設監控，異常未及時回滾
- Spot 節點未設 PriorityClass，關鍵服務被搶佔
- Checkpoint 未設計，Spot 中斷資料遺失
- Shadow 部署未隔離資源，影響線上服務

---

## 面試熱點與經典問題

| 主題              | 常見問題             |
| ----------------- | -------------------- |
| Blue-Green/Canary | 部署策略與選型？     |
| KServe/Triton     | 功能與差異？         |
| EKS+Spot          | 容錯與恢復設計？     |
| PriorityClass     | 如何設計與應用？     |
| Checkpoint        | 推論狀態保存與恢復？ |

---

## 使用注意事項

* 推論服務建議設監控、告警與自動回滾
* Spot 策略需設 PriorityClass 與自動恢復
* Shadow 部署建議隔離資源與流量

---

## 延伸閱讀與資源

* [KServe 官方文件](https://kserve.github.io/website/)
* [Triton Inference Server](https://github.com/triton-inference-server/server)
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
* [EKS Spot 策略](https://aws.amazon.com/tw/blogs/containers/eks-spot-capacity/)
* [K8s PriorityClass](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

---

## 經典面試題與解法提示

1. Blue-Green/Canary/Shadow 部署差異？
2. KServe/Triton/TensorRT-LLM 選型？
3. EKS+Spot 容錯與恢復設計？
4. PriorityClass 實作細節？
5. Checkpoint 恢復策略？
6. 推論服務自動擴縮與監控？
7. Shadow 部署資源隔離？
8. 多模型推論與 GPU 調度？
9. Canary 部署監控與回滾？
10. 推論 API 佈署常見踩坑？

---

## 結語

推論 API 佈署策略是 AI 服務穩健運營的關鍵。熟悉 Blue-Green/Canary、KServe、Triton、EKS+Spot 策略，能讓你打造高效可用的推論平台。下一章將進入 Auto-Scaling 與成本最佳化，敬請期待！
