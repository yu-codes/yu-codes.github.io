---
title: "系統設計與 MLOps 挑戰題庫：13 章經典面試題與解法提示"
date: 2025-05-22 23:59:00 +0800
categories: [System Design & MLOps]
tags: [面試題, 系統設計, MLOps, 白板題, 解題技巧]
---

# 系統設計與 MLOps 挑戰題庫：13 章經典面試題與解法提示

本章彙整前述 12 章大型系統設計與 MLOps 主題的經典面試題，每章精選 10-15 題，涵蓋理論推導、實作、直覺解釋與白板題。每題附上解法提示與常見誤區，幫助你在面試與實戰中脫穎而出。

---

## SYS1 系統設計思維起手式

1. CAP/PACELC 理論推導與應用場景？
2. 垂直/水平擴充選型與風險？
3. SLA/SLO/SLI 擬定與監控？
4. 微服務拆分依據與常見失敗案例？
5. 三層架構如何演進到微服務？
6. 如何用圖解說明系統設計思路？
7. SLA/SLO/SLI 實作細節？
8. 微服務拆分後資料一致性如何處理？
9. 擴充策略如何兼顧成本與彈性？
10. 系統設計常見瓶頸與優化？

---

## SYS2 流量洪峰 & 高可用策略

1. L4/L7 負載均衡差異與選型？
2. 熔斷/降級/重試/回溯原理與實作？
3. 多活/熱備/冷備切換策略與風險？
4. 數據同步如何確保一致性？
5. 健康檢查設計細節？
6. 雪崩效應如何預防？
7. 如何用 Python 實作重試與回溯？
8. 多活架構下資料衝突如何解決？
9. 熔斷/降級與監控如何結合？
10. 高可用架構常見瓶頸與優化？

---

## SYS3 線上-離線分離

1. 線上-離線分離的必要性？
2. 推論 call graph 設計細節？
3. 近線/Streaming 特徵更新的優缺點？
4. Lambda/Kappa 架構選型原則？
5. Training-Serving Skew 如何解決？
6. 如何用 Python 實作線上特徵查詢？
7. Streaming 特徵一致性驗證？
8. 離線/線上特徵同步策略？
9. Lambda 架構下結果合併挑戰？
10. Kappa 架構適用限制？

---

## SYS4 Feature Pipeline 實戰

1. 批次/流式特徵差異與選型？
2. CEP 原理與應用場景？
3. Training-Serving Skew 如何解決？
4. 特徵一致性驗證如何設計？
5. Pipeline 設計如何兼顧時效與穩定？
6. 如何用 Python/Spark/Flink 實作特徵管線？
7. CEP 狀態管理與容錯？
8. 特徵驗證自動化策略？
9. Feature Store 如何輔助特徵一致性？
10. 特徵管線常見踩坑與解法？

---

## SYS5 Feature Store 設計

1. Feature Store 作用與設計要點？
2. Offline/Online Table 差異與同步？
3. Entity Key/TTL/Join Graph 設計原則？
4. 特徵一致性如何驗證與同步？
5. 線上/離線特徵同步挑戰？
6. 如何用 Python 定義與查詢特徵？
7. TTL 設置不當會有什麼風險？
8. Join Graph 如何輔助 lineage？
9. 特徵版本控管與回溯查詢？
10. Feature Store 在推薦/金融的應用？

---

## SYS6 模型版本 & Registry

1. Model Registry 作用與設計要點？
2. MLflow/SageMaker Promote 流程？
3. Semantic Versioning 如何設計與管理？
4. 多版本共存與回滾策略？
5. Promote 流程自動化與監控細節？
6. 如何用 Python 註冊與 Promote 模型？
7. 訓練資料/特徵如何追蹤？
8. Model Registry 與 Feature Store 整合？
9. 多環境部署挑戰？
10. 模型審批與審計如何落地？

---

## SYS7 訓練工作流 Orchestration

1. Airflow/Dagster/Kubeflow 差異與選型？
2. 資源預估如何自動化？
3. Spot/On-Demand 切換策略？
4. Checkpoint sharding/resume 設計？
5. DAG 監控與通知如何落地？
6. Pipeline versioning/artifact tracking？
7. Spot 訓練失敗如何自動恢復？
8. 如何用 Python 實作 checkpoint？
9. DAG 依賴與重試設計細節？
10. Pipeline 設計常見踩坑？

---

## SYS8 CI/CD/CT for ML

1. Code/Data/Model CI 差異與設計？
2. 藍綠/Canary/Shadow 部署選型原則？
3. CT pipeline 觸發條件與自動化？
4. Canary 部署監控與回滾設計？
5. pipeline versioning/artifact tracking？
6. 如何用 YAML/CI 工具設計多層驗證？
7. Shadow 部署資源隔離策略？
8. CT pipeline 如何結合資料監控？
9. 部署回滾常見挑戰？
10. CI/CD/CT 在金融/推薦/廣告的應用？

---

## SYS9 容器化 & Kubernetes

1. Pod 資源 requests/limits 設計原則？
2. GPU 調度與資源隔離如何實現？
3. HPA/VPA/Cluster Autoscaler 差異與組合？
4. Node Selector/Affinity 實作細節？
5. 多租戶資源配額如何設計？
6. 如何用 YAML 實作 GPU 調度？
7. HPA/VPA 監控指標如何選擇？
8. Cluster Autoscaler 啟動延遲如何優化？
9. GPU 資源監控與告警？
10. 容器化部署常見踩坑與解法？

---

## SYS10 Kubeflow & 生態系

1. KFServing/KServe 預測路徑設計？
2. Katib 超參數搜尋原理與應用？
3. TorchX/Volcano 批次排程設計？
4. Ray Serve/BentoML 移植取捨？
5. Kubeflow 與 K8s 原生資源整合？
6. KServe 如何支援多模型管理？
7. Katib 如何自動化 HPO？
8. TorchX/Volcano 資源調度挑戰？
9. Ray Serve/BentoML 部署優缺點？
10. Kubeflow 生態系常見踩坑？

---

## SYS11 監控・告警・追蹤

1. Prometheus/Grafana 指標設計與告警？
2. ELK/Loki 日誌收集與查詢？
3. Jaeger 分散式追蹤原理與應用？
4. Model/Data Drift 線上偵測方法？
5. Trace id 如何設計與串接？
6. 指標、日誌、追蹤如何聯動？
7. Drift 偵測自動化挑戰？
8. 如何用 Python 實作 PSI/KS test？
9. 告警風暴如何預防？
10. 監控平台常見瓶頸與優化？

---

## SYS12 成本・安全・合規

1. GPU 成本優化與 Spot 策略？
2. IAM 最小權限設計與帳號隔離？
3. Secrets 管理工具與自動化？
4. GDPR 刪數據全流程設計？
5. 審計日誌如何集中管理與告警？
6. 多租戶成本分帳如何落地？
7. GPU 資源監控與自動調度？
8. IAM/Secrets 常見風險與防範？
9. GDPR 刪數據挑戰與驗證？
10. 合規審計自動化策略？

---

## 解題技巧與常見誤區

- **計算題**：先寫公式再帶數字，避免粗心。
- **推導題**：分步驟寫清楚，標明假設。
- **直覺題**：用圖解、生活例子輔助說明。
- **實作題**：熟悉 Python、K8s YAML、MLflow、CI/CD 工具等常用 API。
- **常見誤區**：混淆定義、忽略監控、過度依賴單一工具、缺乏自動化。

---

## 結語

本題庫涵蓋系統設計與 MLOps 經典面試題與解法直覺。建議每題都動手推導、實作與解釋，並多練習口頭表達。祝你面試順利、學習愉快！
