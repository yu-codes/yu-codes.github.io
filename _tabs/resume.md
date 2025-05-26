---
title: Resume          # 供導覽列與 <title> 顯示
icon: fas fa-info-circle
order: 1
layout: page           # 預設就是 page，可保留
---

<!-- ===== 單欄 Header ===== -->
<div class="row">
  <div class="col-12" markdown="1">

## YuHan · Backend / AI Cloud Engineer
M.S. in Civil Engineering from National Taiwan University, with two years of experience in backend development and AI applications. Proficient in building containerized backends using Python and FastAPI, and familiar with Hugging Face, FAISS, and RAG system workflows. Hands-on experience with AWS (certified) and GCP. Graduate thesis focused on using machine learning methods to predict extreme rainfall events, strengthening skills in data processing and model development. Independently built an AI-powered article summarization system and an automated order-processing service, delivering end-to-end implementation from data ingestion to model inference, API integration, and message delivery. Passionate about continuing to work in AI engineering and model deployment.

  </div>
</div>

<!-- ===== 雙欄 ===== -->
<div class="row g-4"><aside class="col-md-4" markdown="1">

### Contact
- **Email** dylan.jhou1120@gmail.com  
- **Phone** +886 956‑897‑210  
- **LinkedIn**

### Technical Skills
- **Languages** Python ★ · Java  
- **Backend** FastAPI ★ · Spring Boot  
- **Database** PostgreSQL · MongoDB · Redis  
- **Testing** Pytest  

### Tools & Platforms
- **VCS** Git · GitHub  
- **CI/CD** GitHub Actions  
- **Container** Docker  
- **Cloud Service** AWS ★ · GCP  

### Certifications
- [AWS Certified Developer – Associate (2025)](https://www.credly.com/badges/cf591085-60b2-443a-ab63-037804787827/public_url)
- [AWS Certified AI Practitioner (2024)](https://www.credly.com/badges/5dae0a6b-0bc3-432e-8b9c-2d7d6f660d5b/public_url)
- [AWS Certified AI Practitioner Early Adopter](https://www.credly.com/badges/538b46bc-fa22-4d95-a1ee-0aa4e2ea6a0c/public_url)

### Campus Leadership
- **College of Science Student Association** — Vice President (AY 2017 – 2018)  
- **Geography Student Association** — President (AY 2017 – 2018)  
- **Nantou Regional Alumni Association** — Deputy Camp Director, Winter Camp (Spring 2018)  
- **Department Volleyball Team** — Captain (2024)

</aside><main class="col-md-8" markdown="1">

### Projects / Experience
**Side-Project – LINE Chatbot with RAG Backend (2025/05)**  
*FastAPI · LangChain · Hugging Face · PostgreSQL + pgvector · Docker · LINE Messaging API*

- Designed & implemented a **Retrieval-Augmented Generation pipeline**: LangChain orchestrates embeddings → pgvector search → LLM answer synthesis, exposed via **FastAPI REST / webhook** routes.  
- Built **LINE Messaging Bot** interface: verifies signatures, forwards user queries to RAG backend, streams answers back to chat.  
- Developed **embedding worker container**: chunks PDFs, generates sentence-transformer vectors, and bulk-loads into **PostgreSQL 16 + pgvector**.  
- **Containerised the entire stack** with multi-stage Dockerfiles & `docker-compose`; one command launches API, worker, DB, and **ngrok tunnel**.  
- Organised codebase for clarity: `app/` (API), `rag/` (retrieval), `crawler/` (fetch/schedule), `data/` (vectors/files), `models/` (GGUF), with start scripts & docs.

**Software Engineering Intern @ Startup (2021/08 – 2023/11)**  
*Python · Gmail API · Google Vision API · Google Sheets API · Google Chat Webhooks · Docker*

- **Order Recognition & Placement System**：Independently designed, developed, and launched a fully automated email ordering system.  
- Used **Gmail API** to fetch orders → parsed with **Vision API OCR**, achieving key field accuracy > 95 %.  
- Verified SKU/pricing via **Google Sheets API** to prevent order errors; reduced full process from 2 h to < 10 min.  
- **Google Chat webhook** instant alerts; missed order incidents reduced to zero, saving ≈ 20 h/week of manual effort.

### Education
**National Taiwan University (NTU)** — *M.S. in Civil Engineering*  
Taipei, Taiwan | Sep 2021 – Feb 2025  
- **Thesis:** *Development of Machine‑Learning‑Based Weather Analog Methods for Predicting Extreme Precipitation Events*  

**National Taiwan University (NTU)** — *B.S. in Civil Engineering*  
Taipei, Taiwan | Sep 2016 – Aug 2021  
- **Project – VegPrice‑LSTM**(2021)：Multivariate LSTM model for forecasting daily wholesale vegetable prices (Python/TensorFlow).  
- **Project – StructViz**(2021)：Unity 3D (C#) application visualizing beam and column deformation to support structural mechanics education.  
- **Competition – Asia Cup University Mechanics Contest** (2019)：Won 2nd place; responsible for truss design and load testing.
</main></div>
