---
title: "IaC & DevOps 管線全攻略：Terraform、CloudFormation、ArgoCD、Spinnaker"
date: 2025-05-23 23:00:00 +0800
categories: [雲端部署與服務]
tags: [IaC, Terraform, CloudFormation, Bicep, DevOps, ArgoCD, Spinnaker, GitHub Actions, ECR, EKS, Rolling, Sync Policy]
---

# IaC & DevOps 管線全攻略：Terraform、CloudFormation、ArgoCD、Spinnaker

現代雲端部署離不開基礎設施即程式碼（IaC）與自動化 DevOps 管線。從 Terraform、CloudFormation、Bicep，到 GitHub Actions、ECR、EKS Rolling、ArgoCD Sync Policy、Spinnaker 多階段，本章將結合理論、功能比較、實戰設計、面試熱點與常見誤區，幫助你打造高效可維運的雲端交付流程。

---

## Terraform / CloudFormation / Bicep

### Terraform

- 開源 IaC 工具，支援多雲（AWS/GCP/Azure）、模組化、狀態管理
- 適合跨雲、複雜基礎設施自動化

### CloudFormation

- AWS 原生 IaC，支援資源堆疊、變更集、回滾
- 適合 AWS 單雲、與 AWS 服務深度整合

### Bicep

- Azure 原生 IaC，語法簡潔、與 ARM 深度整合
- 適合 Azure 平台自動化

---

## GitHub Actions → ECR → EKS Rolling

- GitHub Actions：CI/CD 自動化，支援測試、建置、部署
- ECR（Elastic Container Registry）：Docker 映像倉庫
- EKS Rolling：K8s 滾動升級，零停機部署新版本
- Pipeline：程式碼提交 → 測試 → 建置映像 → 推送 ECR → 部署 EKS

```yaml
# GitHub Actions CI/CD 範例
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t $IMAGE .
      - name: Push to ECR
        run: docker push $IMAGE
      - name: Deploy to EKS
        run: kubectl rollout restart deployment/my-app
```

---

## ArgoCD Sync Policy、Spinnaker 多階段

### ArgoCD

- GitOps 工具，支援自動同步 K8s 資源、權限控管、審計
- Sync Policy：自動/手動同步、回滾、健康檢查
- 支援多環境、多分支、PR 驗證

### Spinnaker

- 多雲持續交付平台，支援多階段 Pipeline、藍綠/Canary 部署
- 整合 K8s、Lambda、EC2、GCE、Cloud Run 等

---

## 設計實戰與最佳實踐

- IaC 建議用 Terraform/CloudFormation/Bicep 管理基礎設施
- CI/CD pipeline 建議自動化測試、建置、部署、回滾
- ArgoCD 建議設多環境、多分支、權限控管
- Spinnaker 適合多雲、多階段、複雜部署

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- AI 平台、微服務、K8s 叢集、Serverless、跨雲部署

### 常見誤區

- IaC/CI/CD pipeline 未版本控管，難以回溯
- ArgoCD 權限設計不當，GitOps 失控
- Spinnaker pipeline 過度複雜，維運困難
- IaC/DevOps pipeline 未設自動回滾，部署失敗難恢復

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| Terraform/CloudFormation/Bicep | 差異與選型？ |
| GitHub Actions/ECR/EKS | CI/CD 流程設計？ |
| ArgoCD Sync Policy | 功能與應用？ |
| Spinnaker Pipeline | 多階段設計？ |
| IaC/DevOps pipeline | 回滾與審計？ |

---

## 使用注意事項

* IaC/CI/CD pipeline 建議結合版本控管、審計與自動回滾
* ArgoCD/Spinnaker 建議設多環境、權限控管
* Pipeline 設計需兼顧彈性、可維運性與安全

---

## 延伸閱讀與資源

* [Terraform 官方文件](https://developer.hashicorp.com/terraform/docs)
* [AWS CloudFormation](https://docs.aws.amazon.com/cloudformation/)
* [Azure Bicep](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/)
* [GitHub Actions](https://docs.github.com/en/actions)
* [ArgoCD 官方文件](https://argo-cd.readthedocs.io/en/stable/)
* [Spinnaker 官方文件](https://spinnaker.io/docs/)

---

## 經典面試題與解法提示

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

## 結語

IaC 與 DevOps 管線是雲端平台自動化與可維運的基石。熟悉 Terraform、CloudFormation、ArgoCD、Spinnaker，能讓你打造高效穩健的雲端交付流程。下一章將進入安全、合規與網路，敬請期待！
