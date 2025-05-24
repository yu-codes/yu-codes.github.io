---
title: "容器化與 Kubernetes：Pod 資源、GPU 調度、HPA/VPA/Autoscaler 全攻略"
date: 2025-05-22 20:00:00 +0800
categories: [大型系統設計與MLOps]
tags: [Kubernetes, 容器化, Pod, GPU 調度, HPA, VPA, Cluster Autoscaler, Node Selector, NVIDIA Plugin]
---

# 容器化與 Kubernetes：Pod 資源、GPU 調度、HPA/VPA/Autoscaler 全攻略

容器化與 Kubernetes 是現代 AI 與大規模系統部署的基石。本章將深入 Pod Spec 資源限制、Node Selector、GPU 調度（NVIDIA Device Plugin, Share-GPU）、自動擴縮（HPA/VPA/Cluster Autoscaler），結合理論、實作、面試熱點與常見誤區，幫助你打造高效可擴展的運算平台。

---

## Pod Spec：資源限制 & Node Selector

- 每個 Pod 可設置 CPU/記憶體 requests/limits，防止資源爭用
- Node Selector/Node Affinity 控制 Pod 部署到特定節點（如 GPU/SSD 節點）
- 建議設置 liveness/readiness probe，提升服務穩定性

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-infer
spec:
  containers:
    - name: infer
      image: myrepo/ml-infer:latest
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
  nodeSelector:
    gpu: "true"
```

---

## GPU 調度 (NVIDIA Device Plugin, Share-GPU)

- NVIDIA Device Plugin 支援 GPU 自動發現與分配
- 支援 GPU requests/limits，Pod 可指定需幾張卡
- Share-GPU（如 MIG、KubeShare）支援多 Pod 共用單張 GPU
- 建議監控 GPU 使用率，防止資源閒置或爭用

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## HPA / VPA / Cluster Autoscaler

### HPA（Horizontal Pod Autoscaler）

- 根據 CPU/GPU/自訂指標自動調整 Pod 數量
- 適合流量波動大、需自動擴縮的服務

### VPA（Vertical Pod Autoscaler）

- 根據歷史資源使用自動調整 Pod requests/limits
- 適合長期運行、資源需求變化大的任務

### Cluster Autoscaler

- 根據 Pod 排程需求自動擴縮節點數量
- 支援多雲、混合雲環境

---

## 設計實戰與最佳實踐

- 建議結合 HPA+VPA+Cluster Autoscaler，兼顧彈性與成本
- GPU 任務建議設置 node affinity 與資源限制
- 定期監控資源使用，調整 requests/limits
- 多租戶環境建議設置 namespace 資源配額

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- AI 推論服務、分散式訓練、批次 ETL、彈性 API 平台

### 常見誤區

- Pod 未設資源限制，導致資源爭用或 OOM
- GPU 調度未設 affinity，Pod 排程失敗
- HPA/VPA 參數設置不當，擴縮不及時
- Cluster Autoscaler 未設預留，節點啟動延遲

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Pod 資源限制 | 如何設計？ |
| GPU 調度     | NVIDIA Plugin/Share-GPU 原理？ |
| HPA/VPA      | 適用場景與設計細節？ |
| Cluster Autoscaler | 如何自動擴縮？ |
| Node Selector | 有何作用與限制？ |

---

## 使用注意事項

* Pod/Node 資源建議定期審查與調整
* GPU 調度需結合監控與資源隔離
* HPA/VPA/Autoscaler 需根據業務特性調參

---

## 延伸閱讀與資源

* [Kubernetes 官方文件](https://kubernetes.io/docs/home/)
* [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
* [KubeShare GPU Sharing](https://github.com/ICLUE/kubeshare)
* [HPA 官方文件](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
* [VPA 官方文件](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
* [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler)

---

## 經典面試題與解法提示

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

## 結語

容器化與 Kubernetes 是現代 AI 與大規模系統部署的基石。熟悉 Pod 資源、GPU 調度、HPA/VPA/Autoscaler，能讓你打造高效可擴展的運算平台。下一章將進入 Kubeflow 與生態系，敬請期待！
