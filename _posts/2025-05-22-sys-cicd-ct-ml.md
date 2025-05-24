---
title: "CI/CD/CT for ML：Code/Data/Model CI、藍綠/Canary 部署與持續訓練"
date: 2025-05-22 19:00:00 +0800
categories: [大型系統設計與MLOps]
tags: [CI/CD, CT, MLOps, Data Validation, Model CI, Canary, Blue-Green, Shadow Deployment, Continuous Training]
---

# CI/CD/CT for ML：Code/Data/Model CI、藍綠/Canary 部署與持續訓練

現代 MLOps 需結合軟體工程 CI/CD 與機器學習特有的資料/模型驗證與持續訓練（CT）。本章將深入 Code/Data/Model CI、藍綠/Canary/Shadow 部署、CT 觸發條件，結合理論、實作、面試熱點與常見誤區，幫助你打造自動化、可追溯的 ML 交付流程。

---

## Code CI → Data Validation CI → Model CI

### Code CI

- 程式碼自動化測試、靜態分析、格式檢查
- 工具：GitHub Actions、GitLab CI、Jenkins

### Data Validation CI

- 新資料自動驗證 schema、缺失值、分布異常
- 工具：Great Expectations、Deequ、dbt tests

### Model CI

- 模型訓練、評估、效能回歸測試
- 支援自動化訓練、指標驗證、artifact tracking

```yaml
# GitHub Actions CI 範例
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest
```

---

## CD：藍綠 / Canary / Shadow Deployment

### 藍綠部署（Blue-Green Deployment）

- 同時維護兩套環境，流量切換，快速回滾
- 適合大規模升級、零停機

### Canary 部署

- 先將新版本流量導入小部分，逐步擴大
- 監控指標，異常自動回滾

### Shadow Deployment

- 新模型僅接收流量，不回應用戶，觀察行為差異
- 適合新模型驗證、風險控制

---

## CT (Continuous Training) 觸發條件

- 新資料到達、資料分布漂移、模型效能下降自動觸發再訓練
- 支援自動化 pipeline、監控與告警
- 工具：Kubeflow Pipelines、SageMaker Pipelines、Airflow

---

## 設計實戰與最佳實踐

- CI/CD pipeline 建議結合 Code/Data/Model 驗證
- 部署策略根據業務風險選擇藍綠/Canary/Shadow
- CT 需設計資料監控、效能監控與自動回滾
- pipeline versioning、artifact tracking 建議自動化

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融風控、推薦系統、廣告排序、AI SaaS、雲端平台

### 常見誤區

- 只做 Code CI，忽略資料/模型驗證
- Canary 部署未設監控，異常未及時回滾
- CT pipeline 無資料監控，模型效能漂移未發現
- Shadow 部署未隔離資源，影響線上服務

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Code/Data/Model CI | 差異與設計要點？ |
| 藍綠/Canary/Shadow | 部署策略與選型？ |
| CT           | 觸發條件與自動化？ |
| pipeline versioning | 如何追蹤與回溯？ |
| 部署回滾     | 如何設計？ |

---

## 使用注意事項

* CI/CD pipeline 建議多層驗證與自動化回滾
* Canary/Shadow 部署需設監控與隔離
* CT pipeline 需結合資料/模型監控

---

## 延伸閱讀與資源

* [MLOps CI/CD 實踐](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
* [Canary Deployment 解釋](https://martinfowler.com/bliki/CanaryRelease.html)
* [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
* [Great Expectations](https://docs.greatexpectations.io/docs/)

---

## 經典面試題與解法提示

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

## 結語

CI/CD/CT for ML 是 MLOps 的自動化核心。熟悉 Code/Data/Model CI、藍綠/Canary/Shadow 部署與持續訓練，能讓你打造高效可追溯的 ML 交付流程。下一章將進入容器化與 Kubernetes，敬請期待！
