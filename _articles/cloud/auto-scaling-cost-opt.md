---
title: "Auto-Scaling & 成本最佳化：HPA/VPA/KPA、Spot Fleet、Savings Plan"
date: 2025-05-23 21:00:00 +0800
categories: [Cloud Deployment & Services]
tags: [Auto-Scaling, HPA, VPA, KPA, Spot Fleet, Spot Capacity, Savings Plan, Committed Use, 成本最佳化]
---

# Auto-Scaling & 成本最佳化：HPA/VPA/KPA、Spot Fleet、Savings Plan

雲端平台的彈性擴縮與成本最佳化是大規模 AI 服務的關鍵。本章將深入 HPA/VPA/KPA（Knative）、Spot Fleet、Spot Capacity Pools、Preemptible Node、Savings Plan、Committed Use 折扣等主題，結合理論、實作、面試熱點與常見誤區，幫助你打造高效低成本的雲端平台。

---

## HPA / VPA / KPA (Knative) 指標選擇

### HPA（Horizontal Pod Autoscaler）

- 根據 CPU/GPU/自訂指標自動調整 Pod 數量
- 適合流量波動大、API 服務、推論平台

### VPA（Vertical Pod Autoscaler）

- 根據歷史資源使用自動調整 Pod requests/limits
- 適合長期運行、資源需求變化大的任務

### KPA（Knative Pod Autoscaler）

- Knative 伺服器無狀態服務自動擴縮，支援 QPS、延遲等指標
- 適合 Serverless、事件驅動應用

---

## Spot Fleet, Spot Capacity Pools, Preemptible Node

### Spot Fleet（AWS）

- 組合多種型號/區域的 Spot 實例，提升可用性與成本效益
- 支援自動補貨、價格上限、容量池切換

### Spot Capacity Pools

- 多個可用區/型號的 Spot 資源池，降低搶佔風險
- 建議設多池、分散部署

### Preemptible Node（GCP）

- 低價但可隨時中斷，適合容錯訓練、批次任務
- 建議設計 checkpoint、自動恢復

---

## Savings Plan & Committed Use 折扣

### Savings Plan（AWS）

- 承諾一定用量，享受大幅折扣（最高 72%）
- 適合長期穩定工作負載

### Committed Use（GCP/Azure）

- 預付資源用量，獲得折扣
- 適合預測性高的訓練/推論平台

---

## 設計實戰與最佳實踐

- API/推論服務建議用 HPA/KPA，批次/長時任務用 VPA
- Spot Fleet/Preemptible 建議多池分散、設 checkpoint
- 長期穩定任務建議用 Savings Plan/Committed Use
- 定期審查資源使用，調整自動擴縮與折扣策略

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 推論 API、批次訓練、資料處理、Serverless、雲端平台

### 常見誤區

- HPA/VPA/KPA 參數設置不當，擴縮不及時
- Spot/Preemptible 未設多池，搶佔風險高
- Savings Plan/Committed Use 過度承諾，資源閒置
- 自動擴縮未結合監控，導致資源浪費

---

## 面試熱點與經典問題

| 主題         | 常見問題           |
| ------------ | ------------------ |
| HPA/VPA/KPA  | 差異與選型？       |
| Spot Fleet   | 原理與設計？       |
| Savings Plan | 適用場景與風險？   |
| 多池部署     | 如何降低搶佔風險？ |
| 成本最佳化   | 如何動態調整策略？ |

---

## 使用注意事項

* 自動擴縮建議結合監控與告警
* Spot/Preemptible 建議設多池與自動恢復
* 折扣策略需根據實際用量定期調整

---

## 延伸閱讀與資源

* [AWS HPA 官方文件](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
* [Knative KPA](https://knative.dev/docs/serving/autoscaling/)
* [AWS Spot Fleet](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-fleet.html)
* [GCP Preemptible VM](https://cloud.google.com/preemptible-vms)
* [AWS Savings Plan](https://aws.amazon.com/savingsplans/)
* [GCP Committed Use](https://cloud.google.com/compute/docs/instances/signing-up-committed-use-discounts)

---

## 經典面試題與解法提示

1. HPA/VPA/KPA 差異與選型？
2. Spot Fleet/Preemptible 多池設計？
3. Savings Plan/Committed Use 折扣策略？
4. 自動擴縮參數調整與監控？
5. Spot/Preemptible 容錯設計？
6. 如何用 YAML 設計 HPA/KPA？
7. Spot Fleet 搶佔風險如何降低？
8. 折扣策略過度承諾風險？
9. 多池部署與資源分散？
10. 成本最佳化常見踩坑？

---

## 結語

Auto-Scaling 與成本最佳化是雲端平台營運的關鍵。熟悉 HPA/VPA/KPA、Spot Fleet、Savings Plan 等策略，能讓你打造高效低成本的 AI 平台。下一章將進入監控與觀測性，敬請期待！
