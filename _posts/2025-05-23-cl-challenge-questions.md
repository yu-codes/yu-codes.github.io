---
title: "雲端部署與服務挑戰題庫：14 章經典面試題與解法提示"
date: 2025-05-23 23:59:00 +0800
categories: [雲端部署與服務]
tags: [面試題, 雲端, DevOps, 安全, 觀測性, 白板題, 解題技巧]
---

# 雲端部署與服務挑戰題庫：14 章經典面試題與解法提示

本章彙整前述 13 章雲端部署與服務主題的經典面試題，每章精選 10-15 題，涵蓋理論推導、實作、直覺解釋與白板題。每題附上解法提示與常見誤區，幫助你在面試與實戰中脫穎而出。

---

## CL1 雲端基礎 & IAM

1. VPC/Subnet 劃分與安全設計？
2. Security Group 最小開放原則？
3. IAM Role 權限最小化與跨帳號授權？
4. KMS/Secrets Manager 實作與輪換？
5. Shared Responsibility Model 實務界線？
6. 如何用 Terraform 建立安全 VPC？
7. IAM Policy 審查與自動化？
8. 機密資訊自動輪換策略？
9. VPC/Subnet 設計常見錯誤？
10. 雲端資安審計與合規挑戰？

---

## CL2 AWS AI 生態圈

1. SageMaker Studio 與 JumpStart 差異？
2. Bedrock/Titan/Embedding API 適用場景？
3. Batch Transform vs Endpoint 選型？
4. Endpoint Auto-Scaling 如何設計？
5. JumpStart 模型授權與限制？
6. Bedrock API 權限設計？
7. 多模型部署與 A/B 測試？
8. 如何用 Python 部署 SageMaker Endpoint？
9. Batch Transform 輸入/輸出設計？
10. Bedrock 在企業應用的挑戰？

---

## CL3 GCP AI 生態圈

1. Vertex AI Workbench/Training/Prediction 差異？
2. TPU/GPU/Spot 選型與效能比較？
3. BigQuery ML 適用場景與限制？
4. Generative AI Studio 功能與應用？
5. Spot 訓練如何設計 checkpoint？
6. TPU 程式相容性挑戰？
7. Batch vs Online Prediction 選型？
8. BigQuery ML 如何用 SQL 訓練模型？
9. Generative AI Studio 權限設計？
10. GCP AI 生態圈常見踩坑？

---

## CL4 Azure AI 生態圈

1. ML Workspace/Designer/Endpoint 差異？
2. OpenAI on Azure 與 GPT-4o 佈署？
3. ACI/AKS/Batch 選型與設計原則？
4. Endpoint 多模型部署與 Auto-Scaling？
5. OpenAI on Azure 權限與資安設計？
6. Batch 任務依賴與自動重試？
7. 如何用 Python 部署 Azure Endpoint？
8. Designer 視覺化 Pipeline 實戰？
9. AKS GPU 調度與資源管理？
10. Azure AI 生態圈常見踩坑？

---

## CL5 Serverless 計算

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

## CL6 Kubernetes 管理

1. EKS/GKE/AKS 差異與選型？
2. Karpenter/Cluster Autoscaler 原理與適用場景？
3. Helm/ArgoCD 套件化與 GitOps？
4. Node Pool/資源配額設計？
5. 多雲 K8s 管理挑戰？
6. 如何用 Helm/ArgoCD 部署多環境？
7. Karpenter/Spot 成本優化策略？
8. Helm Chart 參數管理？
9. ArgoCD 權限與審計設計？
10. K8s 管理常見踩坑與解法？

---

## CL7 GPU 叢集調度

1. NVIDIA Device Plugin/MIG 原理與應用？
2. Slurm/Ray/Kubeflow 差異與選型？
3. GPU Sharing/MIG 配置挑戰？
4. Node Affinity/Label 管理異構 GPU？
5. GPU 資源配額與監控？
6. 如何用 YAML 實作 MIG 調度？
7. Slurm/Ray 資源搶佔如何防範？
8. GPU Sharing 權限與隔離？
9. Kubeflow Training Operator 部署細節？
10. GPU 叢集調度常見踩坑？

---

## CL8 推論 API 佈署策略

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

## CL9 Auto-Scaling & 成本最佳化

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

## CL10 監控・觀測性

1. CloudWatch/Stackdriver/Azure Monitor 差異？
2. Prometheus/Grafana 指標設計與告警？
3. Loki/OpenTelemetry 日誌/追蹤設計？
4. Jaeger Trace 分散式追蹤原理？
5. 多層監控如何設計？
6. Trace id 串接與排查？
7. 結構化日誌設計？
8. 告警風暴如何預防？
9. 監控平台多雲整合？
10. 觀測性平台常見踩坑？

---

## CL11 IaC & DevOps 管線

1. Terraform/CloudFormation/Bicep 差異與選型？
2. GitHub Actions/ECR/EKS CI/CD 流程？
3. ArgoCD Sync Policy 功能與應用？
4. Spinnaker 多階段 Pipeline 設計？
5. IaC/DevOps pipeline 回滾與審計？
6. 如何用 YAML/JSON 定義 IaC？
7. ArgoCD 多環境/權限控管？
8. Spinnaker 跨雲部署挑戰？
9. IaC/CI/CD pipeline 常見踩坑？
10. DevOps pipeline 安全與合規？

---

## CL12 安全・合規・網路

1. VPC Peering/PrivateLink 差異與應用？
2. Service Mesh/Istio 流量加密與權限設計？
3. Zero-Trust 原理與落地挑戰？
4. GDPR/HIPAA/ISO 27001 合規要求？
5. IAM 最小權限審查與自動化？
6. 多環境網路隔離策略？
7. 合規審計自動化工具？
8. Service Mesh 可觀測性設計？
9. PrivateLink 配置常見錯誤？
10. 雲端安全與合規常見踩坑？

---

## CL13 多雲 & 混合佈署 (選讀)

1. Anthos/Azure Arc/EKS Anywhere 差異？
2. 雲際資料同步與法規挑戰？
3. Failover/DNS 路由設計？
4. 多雲 IAM/網路隔離？
5. 混合雲資源監控與告警？
6. 多雲部署成本控管？
7. Anthos/Arc 跨雲資源管理？
8. EKS Anywhere 部署挑戰？
9. 多雲/混合雲合規審計？
10. 多雲架構常見踩坑？

---

## 解題技巧與常見誤區

- **計算題**：先寫公式再帶數字，避免粗心。
- **推導題**：分步驟寫清楚，標明假設。
- **直覺題**：用圖解、生活例子輔助說明。
- **實作題**：熟悉 Terraform、K8s YAML、CI/CD、雲端 API 等常用工具。
- **常見誤區**：混淆定義、忽略權限、過度依賴單一雲、缺乏自動化。

---

## 結語

本題庫涵蓋雲端部署與服務經典面試題與解法直覺。建議每題都動手推導、實作與解釋，並多練習口頭表達。祝你面試順利、學習愉快！
