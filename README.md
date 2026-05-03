# ClinIQ — Modern Care by AI Power Service

Clinical Notes NLP Pipeline — FastAPI backend untuk analisis catatan klinis dengan AI.

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
