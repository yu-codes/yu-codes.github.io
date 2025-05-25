---
title: "GPU 叢集調度全攻略：NVIDIA Plugin、MIG、Slurm、Ray、GPU Sharing"
date: 2025-05-23 18:00:00 +0800
categories: [雲端部署與服務]
tags: [GPU, 叢集調度, NVIDIA Device Plugin, MIG, Node Affinity, Slurm, Ray, Kubeflow, GPU Sharing, Multi-Instance GPU]
---

# GPU 叢集調度全攻略：NVIDIA Plugin、MIG、Slurm、Ray、GPU Sharing

現代 AI 訓練與推論高度依賴 GPU 叢集調度。從 Kubernetes NVIDIA Device Plugin、MIG、Node Affinity，到 Slurm、Ray、Kubeflow Training Operator、Multi-Instance GPU 與 GPU Sharing，本章將結合理論、功能比較、實戰設計、面試熱點與常見誤區，幫助你打造高效可擴展的 GPU 平台。

---

## NVIDIA Device Plugin、MIG、Node Affinity

### NVIDIA Device Plugin

- K8s GPU 調度標配，支援自動發現與分配 GPU
- 支援 nvidia.com/gpu 資源限制，Pod 可指定卡數
- 監控 GPU 使用率，防止資源閒置

### MIG（Multi-Instance GPU）

- 支援 A100/Hopper 等 GPU 一卡多用，分割為多個獨立實例
- 適合多租戶、推論混合訓練場景
- K8s 支援 MIG Profile 調度，需設 Node Affinity

### Node Affinity

- 控制 Pod 部署到特定 GPU 型號/配置節點
- 適合異構 GPU 叢集、資源隔離

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
nodeSelector:
  gpu: "a100"
```

---

## Slurm, Ray, Kubeflow Training Operator

### Slurm

- HPC/AI 標準排程器，支援 GPU/CPU 任務、資源配額、佇列管理
- 適合超算中心、科研機構、混合雲

### Ray

- 分散式運算框架，支援動態 GPU 資源分配、彈性調度
- 適合大規模分散式訓練、推論、RL

### Kubeflow Training Operator

- K8s 原生訓練排程，支援 PyTorchJob、TFJob、MPIJob
- 整合 GPU 調度、資源監控、彈性伸縮

---

## Multi-Instance GPU 與 GPU Sharing

- Multi-Instance GPU（MIG）：一卡多用，提升 GPU 利用率
- GPU Sharing：多 Pod/Job 共用單張 GPU（如 MIG、KubeShare、GPU Operator）
- 適合推論服務、低負載多租戶場景
- 建議監控 GPU 使用率與資源隔離

---

## 設計實戰與最佳實踐

- 大型訓練建議用 Slurm/Kubeflow，推論/混合場景用 Ray/MIG
- Node Affinity/Label 管理異構 GPU
- GPU Sharing 建議設資源配額與監控
- 定期審查 GPU 使用率，優化排程策略

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 大規模 AI 訓練、推論服務、科研 HPC、雲端 GPU 叢集

### 常見誤區

- GPU 調度未設 affinity，Pod 排程失敗
- MIG 配置錯誤，資源閒置或衝突
- Slurm/Ray 未設資源配額，任務搶佔
- GPU Sharing 權限設計不當，資料外洩風險

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| NVIDIA Plugin/MIG | 原理與應用場景？ |
| Slurm vs Ray | 差異與選型？ |
| Kubeflow Training Operator | 功能與優勢？ |
| GPU Sharing  | 如何實現與監控？ |
| Node Affinity | 異構 GPU 管理？ |

---

## 使用注意事項

* GPU/MIG 配置建議自動化與監控
* Slurm/Ray/Kubeflow 建議設資源配額
* GPU Sharing 需設權限與隔離策略

---

## 延伸閱讀與資源

* [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
* [MIG 官方文件](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
* [Slurm 官方文件](https://slurm.schedmd.com/documentation.html)
* [Ray 官方文件](https://docs.ray.io/en/latest/)
* [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)

---

## 經典面試題與解法提示

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

## 結語

GPU 叢集調度是 AI 訓練與推論平台的核心。熟悉 NVIDIA Plugin、MIG、Slurm、Ray、GPU Sharing，能讓你打造高效可擴展的 GPU 平台。下一章將進入推論 API 佈署策略，敬請期待！
