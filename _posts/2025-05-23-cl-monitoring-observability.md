---
title: "監控・觀測性全攻略：CloudWatch、Stackdriver、Prometheus、Loki、Trace"
date: 2025-05-23 22:00:00 +0800
categories: [雲端部署與服務]
tags: [監控, 觀測性, CloudWatch, Stackdriver, Azure Monitor, Prometheus, Grafana, Loki, OpenTelemetry, Jaeger]
---

# 監控・觀測性全攻略：CloudWatch、Stackdriver、Prometheus、Loki、Trace

雲端平台的高可用與穩定運營離不開完善的監控與觀測性。從 CloudWatch、Stackdriver、Azure Monitor，到 Prometheus+Grafana、Loki、OpenTelemetry、Jaeger Trace，本章將結合理論、功能比較、實戰設計、面試熱點與常見誤區，幫助你打造高可觀測性的雲端平台。

---

## CloudWatch / Stackdriver / Azure Monitor

### CloudWatch（AWS）

- 雲端原生監控平台，支援指標、日誌、事件、告警
- 整合 Lambda、ECS、EKS、RDS、S3 等服務
- 支援自訂指標、Log Insights、Dashboard

### Stackdriver（GCP）

- 現稱 Cloud Monitoring/Logging，支援多雲監控、日誌、告警
- 整合 GKE、BigQuery、Cloud Run、Vertex AI
- 支援 Trace、Profiler、Error Reporting

### Azure Monitor

- 整合 VM、AKS、App Service、Database 監控
- 支援 Log Analytics、Application Insights、告警規則

---

## Prometheus + Grafana Dashboards

- Prometheus：開源時序資料庫，支援多 exporter，拉取式監控
- Grafana：可視化儀表板，支援多資料源、告警、即時查詢
- 適合 K8s、AI 推論、API 服務等多場景

```yaml
# Prometheus scrape config 範例
scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['localhost:9100']
```

---

## Loki / OpenTelemetry / Jaeger Trace

### Loki

- 輕量級日誌系統，與 Grafana 深度整合
- 適合 K8s、Serverless、雲原生環境

### OpenTelemetry

- 開放標準，支援指標、日誌、追蹤三合一
- 整合多雲、K8s、Serverless、AI 服務

### Jaeger

- 分散式追蹤，支援 OpenTracing 標準
- 可視化請求鏈路、延遲瓶頸、跨服務追蹤

---

## 設計實戰與最佳實踐

- 關鍵服務建議多層監控（資源、應用、API、模型）
- 日誌建議結構化，便於搜尋與異常排查
- Trace 建議結合 trace id，實現全鏈路追蹤
- 告警建議設抑制規則，避免告警風暴

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- AI 推論服務、API 平台、K8s 叢集、Serverless、資料管線

### 常見誤區

- 只監控資源，忽略應用/模型/資料指標
- 日誌未結構化，異常難定位
- Trace 未串接 trace id，跨服務排查困難
- 告警未設抑制，導致告警風暴

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| CloudWatch/Stackdriver | 功能與差異？ |
| Prometheus/Grafana | 指標設計與告警？ |
| Loki/OpenTelemetry | 日誌/追蹤設計？ |
| Jaeger Trace | 分散式追蹤原理？ |
| 多層監控     | 如何設計？ |

---

## 使用注意事項

* 監控建議多層級、結構化、可視化
* Trace id 建議全鏈路貫穿
* 告警建議設抑制與自動化修復

---

## 延伸閱讀與資源

* [AWS CloudWatch](https://docs.aws.amazon.com/cloudwatch/)
* [GCP Stackdriver](https://cloud.google.com/stackdriver)
* [Azure Monitor](https://learn.microsoft.com/en-us/azure/azure-monitor/)
* [Prometheus 官方文件](https://prometheus.io/docs/introduction/overview/)
* [Grafana 官方文件](https://grafana.com/docs/)
* [Loki 官方文件](https://grafana.com/docs/loki/latest/)
* [OpenTelemetry](https://opentelemetry.io/docs/)
* [Jaeger 官方文件](https://www.jaegertracing.io/docs/)

---

## 經典面試題與解法提示

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

## 結語

監控與觀測性是雲端平台穩定運營的關鍵。熟悉 CloudWatch、Stackdriver、Prometheus、Loki、Trace，能讓你打造高可觀測性、易維運的雲端平台。下一章將進入 IaC 與 DevOps 管線，敬請期待！
