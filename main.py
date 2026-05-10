import os
import re
import json
import tempfile
import requests
import numpy as np
import pandas as pd
import faiss
import fitz

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from contextlib import asynccontextmanager

from pdf_generator import generate_pdf
from model import (
    KNOWLEDGE_BASE, SYNTHETIC_DATA, MEDICAL_LATIN,
    NER_CLINICAL_LABELS, CLINICAL_SYNONYMS,
)

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
SERPDEV_API_KEY = os.environ.get("SERPDEV_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL       = "d4data/biomedical-ner-all"
SERPDEV_URL     = "https://serpapi.com/search"

groq_client  = None
embed_model  = None
ner_pipeline = None
faiss_index  = None
classifier   = None
vectorizer   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, embed_model, ner_pipeline, faiss_index
    global classifier, vectorizer

    print("Loading models...")
    groq_client  = Groq(api_key=GROQ_API_KEY)
    embed_model  = SentenceTransformer(EMBED_MODEL)
    ner_pipeline = pipeline("ner", model=NER_MODEL, tokenizer=NER_MODEL, aggregation_strategy="simple")

    kb_embeddings = embed_model.encode(KNOWLEDGE_BASE, convert_to_numpy=True)
    faiss_index   = faiss.IndexFlatL2(kb_embeddings.shape[1])
    faiss_index.add(kb_embeddings)

    vectorizer, classifier = train_classifier()
    print("Models loaded.")
    yield


app = FastAPI(title="ClinIQ API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def train_classifier():
    records = []
    for label, notes in SYNTHETIC_DATA.items():
        for text in notes:
            records.append({"label": label, "text": text})
    df = pd.DataFrame(records)
    df["text_en"], df["lang"] = zip(*df["text"].apply(preprocess_text))
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X   = vec.fit_transform(df["text_en"].tolist())
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, df["label"].tolist())
    return vec, clf


def normalize_latin(text):
    for latin, english in MEDICAL_LATIN.items():
        text = re.sub(latin, english, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = normalize_latin(text)
    try:
        lang = detect(text)
        if lang == "id":
            translated = GoogleTranslator(source="id", target="en").translate(text)
            return translated, "id"
        return text, "en"
    except Exception:
        return text, "unknown"

def retrieve_from_rag(query, top_k=4):
    query_vec          = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vec, top_k)
    return [KNOWLEDGE_BASE[i] for i in indices[0]]

def search_web(query, num_results=3):
    params = {
        "q":       f"clinical guidelines {query}",
        "api_key": SERPDEV_API_KEY,
        "num":     num_results,
        "hl":      "en",
    }
    try:
        response = requests.get(SERPDEV_URL, params=params, timeout=10)
        results  = response.json().get("organic_results", [])
        return [r.get("snippet", "") for r in results if r.get("snippet")]
    except Exception:
        return []

def get_context(query):
    rag = retrieve_from_rag(query)
    if rag:
        return {"source": "RAG", "context": rag}
    web = search_web(query)
    if web:
        return {"source": "Web", "context": web}
    return {"source": "None", "context": []}

STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with"}

def normalize_entity(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'(\d+)\s*(mg|ml|mmhg|l/min|mmol)', r'\1\2', text, flags=re.IGNORECASE)
    return text

def inject_rule_based_entities(text, entities):
    text_lower = text.lower()
    for term, label in CLINICAL_SYNONYMS.items():
        if term in text_lower:
            existing = [e.lower() for e in entities.get(label, [])]
            if not any(term in e or e in term for e in existing):
                entities.setdefault(label, [])
                entities[label].append(term)
    bp_matches = re.findall(r'\b(\d{2,3})\s*/\s*(\d{2,3})\b', text)
    for systolic, _ in bp_matches:
        if int(systolic) < 90:
            existing = [e.lower() for e in entities.get("Sign_symptom", [])]
            if "hypotension" not in existing:
                entities.setdefault("Sign_symptom", [])
                entities["Sign_symptom"].append("hypotension")
            break
    return entities

def extract_entities(text):
    results  = ner_pipeline(text)
    entities = {}
    for ent in results:
        label = ent["entity_group"]
        word  = normalize_entity(ent["word"])
        if len(word) < 3 or word.lower() in STOPWORDS or word.startswith("##"):
            continue
        entities.setdefault(label, [])
        if word.lower() not in [e.lower() for e in entities[label]]:
            entities[label].append(word)
    return inject_rule_based_entities(text, entities)

def classify_note(text):
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]

def summarize_note(text, lang="en"):
    lang_instruction = "Respond in Bahasa Indonesia." if lang == "id" else "Respond in English."
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior clinical documentation specialist. "
                    "Summarize the clinical note in exactly 3 sentences. "
                    "Sentence 1: primary diagnosis and patient background. "
                    "Sentence 2: current medications and interventions. "
                    "Sentence 3: key findings and clinical concerns. "
                    f"{lang_instruction}"
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def generate_explanation(note_id, category, entities, summary, lang="en"):
    retrieval        = get_context(summary)
    context_docs     = "\n".join(f"- {c}" for c in retrieval["context"])
    source_label     = retrieval["source"]
    lang_instruction = "Respond in Bahasa Indonesia." if lang == "id" else "Respond in English."
    prompt = f"""
Clinical Note ID  : {note_id}
Category          : {category}
Extracted Entities: {json.dumps(entities, indent=2)}
Summary           : {summary}

Evidence-Based Reference [{source_label}]:
{context_docs}

Provide structured clinical decision support:
1. KEY MEDICAL CONCERNS
2. MEDICATION REVIEW
3. CLINICAL RISKS (flag HIGH RISK explicitly)
4. RECOMMENDED ACTIONS

{lang_instruction}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ClinIQ, an AI clinical decision support system. "
                    "Outputs are used by licensed clinicians as decision aids only, not final diagnoses. "
                    "Flag HIGH RISK findings explicitly. Be conservative and prioritize patient safety."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip(), source_label

def process_note(note_id, text):
    text_en, lang = preprocess_text(text)
    entities       = extract_entities(text_en)
    category       = classify_note(text_en)
    summary        = summarize_note(text_en, lang)
    explanation, context_source = generate_explanation(note_id, category, entities, summary, lang)
    return {
        "note_id":        note_id,
        "lang":           lang,
        "text_en":        text_en,
        "category":       category,
        "entities":       entities,
        "summary":        summary,
        "explanation":    explanation,
        "context_source": context_source,
    }


@app.get("/")
def root():
    return {"status": "ClinIQ API is running"}


@app.post("/analyze/csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    df = pd.read_csv(tmp_path)
    if "note_id" not in df.columns or "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have columns: note_id, text")
    results = []
    for _, row in df.iterrows():
        result = process_note(str(row["note_id"]), str(row["text"]))
        results.append(result)
    pdf_path = generate_pdf(results)
    return FileResponse(pdf_path, media_type="application/pdf", filename="cliniq_report.pdf")


@app.post("/analyze/pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    doc   = fitz.open(tmp_path)
    pages = [page.get_text() for page in doc]
    raw   = "\n".join(pages)
    notes = re.split(r'\n{2,}|={3,}|-{3,}', raw)
    notes = [n.strip() for n in notes if len(n.strip()) > 50]
    results = []
    for i, text in enumerate(notes):
        result = process_note(f"PDF-{i+1:03d}", text)
        results.append(result)
    pdf_path = generate_pdf(results)
    return FileResponse(pdf_path, media_type="application/pdf", filename="cliniq_report.pdf")
