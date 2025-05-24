---
title: "訓練工作流 Orchestration：Airflow、Dagster、Kubeflow、資源預估與 Checkpoint"
date: 2025-05-22 18:00:00 +0800
categories: [大型系統設計與MLOps]
tags: [Orchestration, Airflow, Dagster, Kubeflow, Pipeline, Spot, On-Demand, Checkpoint, Resume]
---

# 訓練工作流 Orchestration：Airflow、Dagster、Kubeflow、資源預估與 Checkpoint

訓練工作流 Orchestration 是 MLOps 的核心，確保模型訓練、資料處理、部署等流程自動化、可追溯、可恢復。本章將深入 Airflow、Dagster、Kubeflow Pipelines DAG 設計，資源佔用預估、Spot/On-Demand 策略、Checkpoint sharding & Resume，結合理論、實作、面試熱點與常見誤區，幫助你打造高效穩健的 ML pipeline。

---

## Airflow / Dagster / Kubeflow Pipelines DAG

### Airflow

- 最主流的 Workflow Orchestrator，支援 DAG、排程、重試、監控
- 適合 ETL、資料前處理、訓練排程、模型部署
- 支援多種 Operator（Bash、Python、KubernetesPod、BigQuery 等）

### Dagster

- 強調型別安全、資產導向、開發體驗佳
- 支援資料資產追蹤、測試、分區管理
- 適合 ML pipeline、資料治理、資產 lineage

### Kubeflow Pipelines

- 雲原生 ML pipeline，支援容器化、GPU、分散式訓練
- 支援 pipeline versioning、artifact tracking、可視化 UI

```python
# Airflow DAG 訓練任務範例
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_model():
    # ...模型訓練邏輯...
    pass

with DAG('ml_training', start_date=datetime(2023,1,1), schedule_interval='@daily') as dag:
    t1 = PythonOperator(task_id='train', python_callable=train_model)
```

---

## 資源佔用預估、Spot vs. On-Demand

### 資源佔用預估

- 根據資料量、模型大小、訓練步數預估 CPU/GPU/記憶體需求
- 可用歷史紀錄、profiling 工具（如 nvidia-smi, top, kubectl）

### Spot vs. On-Demand

- Spot：低價但可能隨時中斷，適合容錯訓練、非即時任務
- On-Demand：穩定但價格高，適合關鍵訓練、即時部署
- 混合策略：先用 Spot，失敗自動切換 On-Demand

---

## Checkpoint sharding & Resume

- 訓練過程定期保存 checkpoint，支援斷點續訓
- Checkpoint sharding：將大模型 checkpoint 拆分多檔，提升儲存與恢復效率
- Resume：Spot 中斷後自動恢復訓練，減少資源浪費
- 工具：PyTorch Lightning、DeepSpeed、TensorFlow、Kubeflow Checkpoint Operator

```python
# PyTorch checkpoint 範例
import torch
def save_ckpt(model, optimizer, path):
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, path)
def load_ckpt(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
```

---

## 設計實戰與最佳實踐

- DAG 設計需考慮依賴、重試、監控、通知
- 資源預估建議自動化，結合監控與動態調度
- Spot 訓練需設計 checkpoint 與自動恢復
- Pipeline versioning、artifact tracking 建議結合 MLflow、Kubeflow

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 大規模模型訓練、資料前處理、ML pipeline、雲端自動化部署

### 常見誤區

- DAG 無監控，失敗難追蹤
- Spot 訓練未設 checkpoint，資源浪費
- 資源預估不準，導致排程失敗或成本爆炸
- Pipeline versioning/追蹤未落地，難以回溯

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Airflow/Dagster/Kubeflow | 差異與選型？ |
| 資源預估     | 如何自動化？ |
| Spot/On-Demand | 適用場景與切換策略？ |
| Checkpoint   | 如何設計 sharding 與 resume？ |
| DAG 設計     | 依賴、重試、監控細節？ |

---

## 使用注意事項

* Pipeline 設計需結合監控、通知與自動恢復
* Spot 訓練建議結合 checkpoint 與自動切換
* DAG versioning、artifact tracking 建議自動化

---

## 延伸閱讀與資源

* [Airflow 官方文件](https://airflow.apache.org/docs/)
* [Dagster 官方文件](https://docs.dagster.io/)
* [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
* [PyTorch Checkpoint 範例](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
* [Spot vs On-Demand 策略](https://aws.amazon.com/ec2/spot/)

---

## 經典面試題與解法提示

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

## 結語

訓練工作流 Orchestration 是 MLOps 的核心。熟悉 Airflow、Dagster、Kubeflow、資源預估與 checkpoint 策略，能讓你打造高效穩健的 ML pipeline。下一章將進入 CI/CD/CT for ML，敬請期待！
