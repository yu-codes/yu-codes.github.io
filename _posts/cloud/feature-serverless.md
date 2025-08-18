---
title: "Serverless 計算全攻略：Lambda、Fargate、Cloud Functions、冷啟動與事件橋"
date: 2025-05-23 16:00:00 +0800
categories: [Cloud Deployment & Services]
tags: [Serverless, Lambda, Fargate, Cloud Functions, Cloud Run, Azure Functions, Provisioned Concurrency, 冷啟動, EventBridge, PubSub, Event Grid]
---

# Serverless 計算全攻略：Lambda、Fargate、Cloud Functions、冷啟動與事件橋

Serverless 計算讓開發者專注於業務邏輯，無需管理基礎設施。從 Lambda、Fargate、Cloud Functions、Cloud Run、Azure Functions，到 Provisioned Concurrency、冷啟動對策、事件橋（EventBridge/PubSub/Event Grid），本章將結合理論、功能比較、實戰設計、面試熱點與常見誤區，幫助你打造高效彈性的雲端服務。

---

## Lambda / Fargate vs. Cloud Functions / Cloud Run / Azure Functions

### Lambda（AWS）

- 事件驅動無伺服器運算，支援多語言、API Gateway、S3 觸發
- 適合短時、彈性、事件型任務
- 支援 Provisioned Concurrency 降低冷啟動

### Fargate（AWS）

- 無伺服器容器運算，支援 ECS/EKS，適合長時、複雜任務
- 自動調度資源，無需管理節點

### Cloud Functions（GCP）/ Azure Functions

- 事件驅動函數服務，支援多語言、HTTP/事件觸發
- 適合資料處理、API、Webhook

### Cloud Run（GCP）

- 無伺服器容器平台，支援任意語言/框架
- 支援自動擴縮、HTTP 觸發、零管理

---

## Provisioned Concurrency、冷啟動對策

- 冷啟動：首次請求需初始化環境，導致延遲
- Provisioned Concurrency（Lambda）：預先啟動容器，降低冷啟動延遲
- Cloud Run/Functions：可設最小實例數，減少冷啟動
- 建議：高峰時段預熱、分流流量、監控延遲

---

## EventBridge / PubSub / Event Grid

### EventBridge（AWS）

- 事件匯流排，支援多服務事件路由、過濾、轉換
- 適合微服務解耦、事件驅動架構

### PubSub（GCP）

- 分散式訊息服務，支援即時/批次事件傳遞
- 適合資料流、IoT、串流處理

### Event Grid（Azure）

- 事件路由服務，支援多來源/目標、事件過濾
- 適合自動化、Serverless 整合

---

## 設計實戰與最佳實踐

- 短時/事件型任務用 Lambda/Cloud Functions，長時/容器用 Fargate/Cloud Run
- 高流量建議設 Provisioned Concurrency/最小實例數
- 事件橋建議用於微服務解耦、跨服務觸發
- 建議監控冷啟動延遲與失敗率

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- API、Webhook、資料處理、IoT、即時通知、微服務事件流

### 常見誤區

- 冷啟動延遲未優化，影響用戶體驗
- Lambda 濫用於長時任務，成本高
- 事件橋未設過濾，導致流量爆炸
- Serverless 權限設計不當，資安風險

---

## 面試熱點與經典問題

| 主題                          | 常見問題         |
| ----------------------------- | ---------------- |
| Lambda vs Fargate             | 適用場景與差異？ |
| 冷啟動                        | 如何優化？       |
| EventBridge/PubSub/Event Grid | 事件流設計？     |
| Cloud Run/Functions           | 選型與限制？     |
| Provisioned Concurrency       | 設計與調參？     |

---

## 使用注意事項

* Lambda/Functions 建議設超時與重試
* 冷啟動建議預熱與監控
* 事件橋建議設過濾與權限控管

---

## 延伸閱讀與資源

* [AWS Lambda 官方文件](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)
* [AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
* [GCP Cloud Functions](https://cloud.google.com/functions)
* [GCP Cloud Run](https://cloud.google.com/run)
* [Azure Functions](https://learn.microsoft.com/en-us/azure/azure-functions/)
* [EventBridge](https://docs.aws.amazon.com/eventbridge/latest/userguide/what-is-amazon-eventbridge.html)
* [GCP PubSub](https://cloud.google.com/pubsub)
* [Azure Event Grid](https://learn.microsoft.com/en-us/azure/event-grid/)

---

## 經典面試題與解法提示

1. Lambda/Fargate/Cloud Functions/Cloud Run 差異？
2. 冷啟動延遲如何優化？
3. EventBridge/PubSub/Event Grid 設計原則？
4. Provisioned Concurrency 實作細節？
5. Serverless 權限與資安設計？
6. Lambda/Functions 超時與重試設計？
7. Cloud Run/Functions 選型挑戰？
8. 事件橋過濾與流量控制？
9. Serverless 成本優化？
10. Serverless 架構常見踩坑？

---

## 結語

Serverless 計算是現代雲端服務的核心。熟悉 Lambda、Fargate、Cloud Functions、冷啟動與事件橋，能讓你打造高效彈性的雲端平台。下一章將進入 Kubernetes 管理，敬請期待！
