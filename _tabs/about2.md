---
# the default layout is 'page'
icon: fas fa-info-circle
order: 5
title: Resume2
---
<!-- ===== 單欄 Header 區 ===== -->
<div class="row">
  <div class="col-12" markdown="1">

## YuHan · Backend / AI Cloud Engineer

  </div>
</div>

<!-- ====== 既有雙欄區 ====== -->
<div class="row g-4" markdown="1">

  <!-- 左欄 -->
  <aside class="col-md-4" markdown="1">

### Contact

- **Email** dylan.jhou1120@gmail.com
- **Phone** (886) 956-897-210
- **Linkedin**

### Technical Skills

- **Languages** Python ★ · Java
- **Backend** FastAPI ★ · Spring Boot
- **Database** PostgreSQL · MongoDB · Redis
- **Testing** Pytest

### Tools & Platforms

- **VCS** Git · Github
- **CI/CD** Github Action
- **Container** Docker
- **Cloud Service** AWS ★ · GCP

### Certifications

- **AWS Certified Development Associate (DVA-C02)**
- **AWS Certified AI Practitioner (AIF-C01)**

### Campus Leadership

- **College of Science Student Association** — Vice President (AY 2017 – 2018)
- **Geography Student Association** — President (AY 2017 – 2018)
- **Nantou Regional Alumni Association** — Deputy Camp Director, Winter Camp (Spring 2018)
- **Department Volleyball Team** — Captain (2024)
  </aside>

  <!-- 右欄 -->
  <main class="col-md-8" markdown="1">

### Project/ Experience

**AI 技術文章摘要管線專案 (2025 / 05)**

*FastAPI · Hugging Face Transformers · PostgreSQL · Notion API · LINE Messaging API · Docker*

- 於 4 週內獨立完成端到端開發：爬取目標網站技術文章 → AI 摘要 → 資料持久化 → 即時推播。
- 使用 **FastAPI** 非同步爬蟲，將文章內容送往 **Hugging Face** 模型產生 JSON 摘要，延遲低於 3 s/篇。
- 透過 **Notion API** 與 **PostgreSQL** 雙寫入，支援 REST 查詢與後續資料分析；容器化部署一鍵上雲。
- 整合 **LINE Bot**，自動推送最新摘要給訂閱者，省去人工整理流程，提升閱讀效率與即時性。

**Software Engineering Intern @ 新創公司 (2021 / 08 – 2023 / 11)**

*Python · Gmail API · Google Vision API · Google Sheets API · Google Chat Webhooks · Docker*

- **Order Recognition & Placement System**：獨立設計、開發並上線，自動化整個 Email 下訂流程。
- 以 **Gmail API** 擷取訂單郵件 → **Vision API OCR** 解析發票內容，關鍵欄位擷取準確率 > 95 %。
- 透過 **Google Sheets API** 即時比對公司內部 SKU／價格表，避免商品或金額錯誤。
- 調用公司內部 **REST API** 完成下單，平均處理時間由人工 2 h 降至 < 10 min。
- 建立 **Google Chat webhook** 推播，秒級通知營運團隊，錯失訂單事件歸零，節省 ≈ 20 h/週人力。

### Education
**National Taiwan University (NTU)** — *M.S. in Civil Engineering*
Taipei, Taiwan | Sep 2021 – Feb 2025

- **Thesis:** *Development of Machine‑Learning‑Based Weather Analog Methods for Predicting Extreme Precipitation Events*
*Relevant coursework: Machine Learning, Distributed Systems, Cloud Computing, Numerical Weather Prediction*

**National Taiwan University (NTU)** — B.S. in Civil Engineering
Taipei, Taiwan | Sep 2018 – Aug 2021

- **Project – VegPrice‑LSTM** Built a multivariate LSTM model (Python / TensorFlow) that combines weather, calendar, and market factors to forecast daily wholesale vegetable prices.
- **Project – StructViz** Developed an interactive Unity 3D application (C#) that visualizes beam‑and‑column deformation under various loads and boundary conditions for structural‑mechanics classes.
- **Competition – Asia Cup University Mechanics Contest** Served as structural analyst in NTU’s team, focusing on truss design and load testing throughout the regional mechanics tournament.

  </main>

</div>
