Markdown
# 🎯 TalentAlign AI: Enterprise Resume Screener (RAG Pipeline)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-orange.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/LLM-Llama%203.1%20(Groq)-green.svg" alt="Llama 3.1">
</p>

An advanced, production-ready **Retrieval-Augmented Generation (RAG)** pipeline designed to automate corporate talent acquisition. This system parses high volumes of multi-domain technical resumes (`.docx`), computes mathematical semantic alignment against custom Job Descriptions, and delivers granular fitment analytics with zero hallucination.

---

## 🚀 Key Architecture & Features

- 🧠 **Semantic Embedding Matrix:** Utilizes `all-MiniLM-L6-v2` via **SentenceTransformers** to map complex candidate experiences into high-dimensional vector spaces.
- 🗄️ **High-Performance Vector Store:** Powered by **ChromaDB** using Cosine Similarity metrics to fetch the most contextually relevant candidate text blocks.
- 🤖 **Deterministic LLM Orchestration:** Integrated with **Llama 3.1 (8B Instant)** via **LangChain & Groq API** for ultra-fast context assessment and zero-drift reasoning.
- 🎨 **Enterprise Dashboard UI:** A clean, metric-driven **Streamlit** interface featuring secure asynchronous document ingestion and dynamic top-k candidate filtering.
- 🛡️ **Production-Grade Security:** Fully decoupled architecture using **Streamlit Secrets Management** to prevent raw API key exposure in public version control.

---

## 🏗️ System Blueprint (Data Flow)

[ Bulk .docx Resumes ] ➔ [ Recursive Text Splitter ] ➔ [ SentenceTransformers ]
│
▼
[ Markdown Report Table ] 🗲 [ Llama 3.1 LLM ] 🗲 [ ChromaDB Vector Engine ]


---

## 🛠️ Tech Stack & Dependencies

- **Orchestration:** LangChain Core, LangChain Community, LangChain Groq
- **Vector Database:** ChromaDB
- **Embeddings Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Frontend Dashboard:** Streamlit
- **Document Processing:** Docx2txt
- **Data Engineering:** Pandas

---

## 📦 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/KAVINGUPTA09/RAG-RESUME-SCREENER.git](https://github.com/KAVINGUPTA09/RAG-RESUME-SCREENER.git)
   cd RAG-RESUME-SCREENER
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Local Environment / Secrets:
Create a local Streamlit secrets file .streamlit/secrets.toml:

Ini, TOML
GROQ_API_KEY = "your_groq_api_key_here"
Boot the Dashboard:

Bash
streamlit run app.py
📊 Sample Pipeline Output
When a multi-domain Job Description (e.g., Healthcare MIS / EDI 837 / HIPAA Analyst) is executed against the vector space, the system generates a precise evaluation matrix:

Rank	Candidate Name	Match Score (1-100)	Fitment Reason
1	Bharatha BA	85	Strong core experience in SDLC methodologies, requirements gathering, and creating functional specifications using MS Visio.
2	Venkat_BA	78	Proficient in creating complex UML diagrams and implementing enterprise Cloud HCM modules.
3	MounikaReddy	75	Extensive healthcare domain expertise, specifically with HIPAA EDI transactions (837/835) and implementation configurations.
📂 Repository Structure
├── app.py                     # Main Streamlit Application UI & Logic
├── requirements.txt           # Production Library Dependencies
├── RAG_RESUME_SCREENER.ipynb  # Core Architecture R&D Notebook
└── README.md                  # System Documentation
🛡️ Core Data Privacy Disclaimer
This enterprise screener adheres strictly to data compliance standards. The pipeline does not store personal candidate identifiers globally; data ingestion and vector tokenization happen strictly within the user's localized session environment.


