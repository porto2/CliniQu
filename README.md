<p align="center">
  <img src="banner.svg" alt="ClinIQ Banner"/>
</p>



![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

 

![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=for-the-badge&logo=fastapi)

 

![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)

 

![Railway](https://img.shields.io/badge/Deploy-Railway-purple?style=for-the-badge&logo=railway)



> Clinical Notes NLP Pipeline — FastAPI backend untuk analisis catatan klinis dengan AI. Upload CSV atau PDF, dapat laporan PDF profesional lengkap dengan AI explanation, donut chart, dan clinical decision support.

---

## Endpoints
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Health check |
| POST | `/analyze/csv` | Upload file CSV |
| POST | `/analyze/pdf` | Upload file PDF |

## Format CSV
    note_id,text
    N001,"Patient is a 65-year-old male with hypertension..."
    N002,"ICU admission for septic shock..."

## Deploy ke Railway
1. Push repo ke GitHub
2. Buat project baru di Railway
3. Connect repo
4. Set env vars: GROQ_API_KEY dan SERPDEV_API_KEY

## Run Lokal
    pip install -r requirements.txt
    uvicorn main:app --reload

## Tech Stack
- FastAPI + Uvicorn
- Groq LLM llama-3.3-70b
- HuggingFace Transformers NER
- FAISS RAG vector search
- SerpDev web search fallback
- ReportLab + Matplotlib PDF generation
