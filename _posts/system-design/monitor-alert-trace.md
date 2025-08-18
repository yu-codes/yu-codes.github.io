---
title: "監控・告警・追蹤全攻略：Prometheus、Grafana、ELK、Loki、Drift 偵測"
date: 2025-05-22 22:00:00 +0800
categories: [System Design & MLOps]
tags: [監控, 告警, 追蹤, Prometheus, Grafana, ELK, Loki, Jaeger, Model Drift, Data Drift]
---

# 監控・告警・追蹤全攻略：Prometheus、Grafana、ELK、Loki、Drift 偵測

現代大型系統與 MLOps 平台需具備完善的監控、告警與追蹤能力，才能保障服務穩定、快速定位異常並持續優化。本章將深入 Prometheus/Grafana 指標流、ELK/Loki/Jaeger 日誌與追蹤、Model/Data Drift 線上偵測，結合理論、實作、面試熱點與常見誤區，幫助你打造高可觀測性的智能平台。

---

## Prometheus / Grafana 指標流

- Prometheus：時序資料庫，支援多種 exporter，拉取式監控
- Grafana：可視化儀表板，支援多資料源、告警規則
- 常見指標：CPU/GPU/記憶體、QPS、延遲、錯誤率、模型推論指標

```yaml
# Prometheus 指標監控範例
scrape_configs:
  - job_name: 'ml-infer'
    static_configs:
      - targets: ['localhost:9100']
```

---

## Log / Trace：ELK / Loki / Jaeger

### ELK（Elasticsearch, Logstash, Kibana）

- 日誌收集、搜尋、可視化，支援結構化/非結構化日誌
- 適合大規模日誌分析、異常排查

### Loki

- 輕量級日誌系統，與 Grafana 深度整合
- 適合雲原生、Kubernetes 環境

### Jaeger

- 分散式追蹤，支援 OpenTracing 標準
- 可視化請求鏈路、延遲瓶頸、跨服務追蹤

---

## Model Drift / Data Drift 線上偵測

- Model Drift：模型預測分布與訓練時顯著不同，需再訓練
- Data Drift：輸入資料分布變化，可能影響模型效能
- 線上偵測方法：統計檢定（KS test, PSI）、滑動窗口監控、異常告警
- 工具：Evidently AI、Alibi Detect、Prometheus 指標自訂

```python
# PSI (Population Stability Index) 偵測範例
import numpy as np
def psi(expected, actual, buckets=10):
    expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=buckets)[0] / len(actual)
    return np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
```

---

## 設計實戰與最佳實踐

- 指標、日誌、追蹤三位一體，結合告警與自動化修復
- 監控指標需覆蓋資源、服務、模型、資料多層級
- Drift 偵測建議自動化，異常自動通知與觸發 retrain
- 日誌與追蹤建議結合 trace id，便於全鏈路排查

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- AI 推論服務、API 平台、分散式系統、金融/醫療監控

### 常見誤區

- 只監控資源，忽略模型/資料指標
- 告警未設抑制，導致告警風暴
- 日誌未結構化，異常難定位
- Drift 偵測僅離線，未線上自動化

---

## 面試熱點與經典問題

| 主題               | 常見問題         |
| ------------------ | ---------------- |
| Prometheus/Grafana | 指標設計與告警？ |
| ELK/Loki           | 日誌收集與查詢？ |
| Jaeger             | 分散式追蹤原理？ |
| Drift 偵測         | 如何線上自動化？ |
| Trace id           | 如何設計與串接？ |

---

## 使用注意事項

* 指標、日誌、追蹤需結合 trace id 與自動化告警
* Drift 偵測建議滑動窗口與多指標聯合
* 日誌建議結構化，便於搜尋與分析

---

## 延伸閱讀與資源

* [Prometheus 官方文件](https://prometheus.io/docs/introduction/overview/)
* [Grafana 官方文件](https://grafana.com/docs/)
* [ELK Stack](https://www.elastic.co/what-is/elk-stack)
* [Loki 官方文件](https://grafana.com/docs/loki/latest/)
* [Jaeger 官方文件](https://www.jaegertracing.io/docs/)
* [Evidently AI](https://docs.evidentlyai.com/)
* [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/stable/)

---

## 經典面試題與解法提示

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

## 結語

監控、告警與追蹤是大型系統與 MLOps 的生命線。熟悉 Prometheus、Grafana、ELK、Loki、Jaeger 與 Drift 偵測，能讓你打造高可觀測性、易維運的智能平台。下一章將進入成本、安全與合規，敬請期待！
