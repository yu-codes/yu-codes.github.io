---
title: "模型版本與 Registry：MLflow、SageMaker、Semantic Versioning 與 Promote 流程"
date: 2025-05-22 17:00:00 +0800
categories: [大型系統設計與MLOps]
tags: [模型版本, Model Registry, MLflow, SageMaker, Semantic Versioning, Promote, Staging, Production]
---

# 模型版本與 Registry：MLflow、SageMaker、Semantic Versioning 與 Promote 流程

模型版本控管與 Registry 是 MLOps 的核心，確保模型可追溯、可回滾、可審計，支援多環境部署與自動化運維。本章將深入 MLflow Model Registry、SageMaker Model Package、Semantic Model Versioning、Promote → Staging → Production 流程，結合理論、實作、面試熱點與常見誤區，幫助你打造穩健的模型生命週期管理。

---

## MLflow Model Registry、SageMaker Model Package

### MLflow Model Registry

- 支援模型註冊、版本控管、狀態流轉（Staging/Production/Archived）
- 支援模型審批、註解、審計日誌
- 可與 CI/CD、監控、A/B 測試整合

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = "runs:/<run_id>/model"
client.create_registered_model("my_model")
client.create_model_version("my_model", model_uri, "<run_id>")
client.transition_model_version_stage("my_model", 1, stage="Staging")
```

### SageMaker Model Package

- AWS 雲端原生，支援模型註冊、審批、部署
- 支援多框架（XGBoost、PyTorch、TensorFlow）、多環境 Promote
- 整合 SageMaker Pipeline、Endpoint、A/B 測試

---

## Semantic Model Versioning

- 採用語意化版本（如 1.2.0），標示重大/功能/修正變更
- 支援模型回溯、依賴追蹤、兼容性管理
- 建議結合 Git tag、Artifact hash、訓練參數記錄

---

## Promote → Staging → Production 流程

- Promote：模型註冊後進入 Staging，進行驗證、A/B 測試
- Staging：通過驗證後 Promote 至 Production，正式服務流量
- Production：線上服務，支援回滾、審計、監控
- 支援多環境（Dev/Test/Prod）、自動化部署、灰度發布

---

## 設計實戰與最佳實踐

- 模型註冊需記錄訓練資料、特徵、超參數、環境
- Promote 流程建議自動化（CI/CD）、結合監控與驗證
- 支援多版本共存、A/B/Canary 測試、快速回滾
- 建議結合 Feature Store、Data Lineage 追蹤全流程

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融風控、推薦系統、醫療診斷、廣告排序、AI SaaS

### 常見誤區

- 模型版本未記錄訓練資料與特徵，難以回溯
- Promote 流程未自動化，部署易出錯
- 多版本共存未隔離，導致流量混亂
- Semantic Versioning 濫用，無法反映實際變更

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Model Registry | 作用與設計要點？ |
| MLflow/SageMaker | 註冊與 Promote 流程？ |
| Semantic Versioning | 如何設計與管理？ |
| 多版本共存   | 如何隔離與回滾？ |
| Promote 流程 | 自動化與監控細節？ |

---

## 使用注意事項

* 模型註冊需記錄全流程資訊，便於審計與回溯
* Promote 流程建議結合自動化測試與監控
* 多版本共存需設計流量分配與隔離策略

---

## 延伸閱讀與資源

* [MLflow Model Registry 官方文件](https://mlflow.org/docs/latest/model-registry.html)
* [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
* [Semantic Versioning](https://semver.org/)
* [MLOps Model Lifecycle](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#model_registry)

---

## 經典面試題與解法提示

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

## 結語

模型版本與 Registry 是 MLOps 的基石。熟悉 MLflow、SageMaker、Semantic Versioning 與 Promote 流程，能讓你打造穩健可追溯的模型生命週期管理。下一章將進入訓練工作流 Orchestration，敬請期待！
