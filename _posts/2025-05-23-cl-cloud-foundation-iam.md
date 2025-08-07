---
title: "雲端基礎 & IAM：VPC、Subnet、SG、IAM Role、資安守門與責任模型"
date: 2025-05-23 12:00:00 +0800
categories: [Cloud Deployment & Services]
tags: [雲端, IAM, VPC, Subnet, Security Group, KMS, Secrets Manager, STS, Well-Architected, Shared Responsibility]
---

# 雲端基礎 & IAM：VPC、Subnet、SG、IAM Role、資安守門與責任模型

雲端部署的基礎在於網路、權限與資安設計。從 VPC、Subnet、Security Group、IAM Role，到 Shared Responsibility Model、Well-Architected Framework、KMS、Secrets Manager、STS 等資安守門，本章將結合理論、圖解、實戰、面試熱點與常見誤區，幫助你建立穩健的雲端基礎。

---

## 核心資源：VPC / Subnet / SG / IAM Role

### VPC（Virtual Private Cloud）

- 雲端中的邏輯隔離網路，類似自家資料中心
- 可自訂 IP 範圍、子網（Subnet）、路由表、NAT Gateway

### Subnet

- VPC 內的子網，劃分公有/私有區域
- 公有子網：可連外網（有 IGW），私有子網：僅內部通訊

### Security Group（SG）

- 虛擬防火牆，控制進出流量（IP/Port/協議）
- 預設拒絕所有，需明確開放

### IAM Role

- 權限最小化原則，授權給 EC2/Lambda/SageMaker 等服務
- 支援跨帳號、臨時授權（AssumeRole）

---

## Shared Responsibility Model 與 Well-Architected

### Shared Responsibility Model

- 雲端供應商負責基礎設施安全，客戶負責資料/應用安全
- 例：AWS 管理硬體、網路，客戶管理 OS、應用、資料加密

### Well-Architected Framework

- AWS/GCP/Azure 提供的最佳實踐指引
- 五大支柱：卓越運營、安全、可靠、效能、成本最佳化

---

## 資安守門：KMS、Secrets Manager、STS

### KMS（Key Management Service）

- 管理加密金鑰，支援資料加密/解密、輪換、審計
- 整合 S3、RDS、EBS、Lambda 等服務

### Secrets Manager

- 集中管理 API Key、DB 密碼等機密資訊
- 支援自動輪換、加密、存取審計

### STS（Security Token Service）

- 提供臨時憑證，支援跨帳號、短期授權
- 降低長期憑證外洩風險

---

## 設計實戰與最佳實踐

- VPC/Subnet 劃分公私區，敏感服務僅部署於私有子網
- Security Group 僅開放必要 Port，禁止 0.0.0.0/0
- IAM Role 採最小權限，定期審查與輪換
- 機密資訊集中於 Secrets Manager，嚴禁硬編碼
- 關鍵資料建議全程加密，金鑰託管於 KMS

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 金融、醫療、SaaS、AI 平台、企業雲端遷移

### 常見誤區

- Security Group 開放過寬，導致資安風險
- IAM 權限過大，未設最小權限
- 機密資訊硬編碼於程式碼
- 忽略 Shared Responsibility，誤以為雲端全權負責

---

## 面試熱點與經典問題

| 主題                  | 常見問題             |
| --------------------- | -------------------- |
| VPC/Subnet            | 劃分原則與安全設計？ |
| Security Group        | 如何設計最小開放？   |
| IAM Role              | 權限最小化與跨帳號？ |
| KMS/Secrets           | 加密與機密管理？     |
| Shared Responsibility | 客戶/雲端責任界線？  |

---

## 使用注意事項

* 定期審查 Security Group、IAM Policy
* 機密資訊建議自動輪換與審計
* VPC/Subnet 設計需兼顧安全與可擴展性

---

## 延伸閱讀與資源

* [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
* [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
* [AWS KMS 官方文件](https://docs.aws.amazon.com/kms/latest/developerguide/)
* [Secrets Manager 官方文件](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html)
* [Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/)

---

## 經典面試題與解法提示

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

## 結語

雲端基礎與 IAM 是安全部署的第一步。熟悉 VPC、Subnet、SG、IAM Role、KMS、Secrets Manager 與責任模型，能讓你打造穩健的雲端平台。下一章將進入 AWS AI 生態圈，敬請期待！
