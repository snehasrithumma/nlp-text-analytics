import datetime
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import spacy
import time
import json
import psutil
from pathlib import Path
import os
import uuid
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "spacy_analysis_log.jsonl"

nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def log_to_file(data):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)  # in MB


@app.get("/")
def root():
    return {"message": "Backend is working!"}

@app.post("/analyzespacy")
async def analyze_text(text: str = Form(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    doc = nlp(text)
    duration = round(time.time() - start_time, 4)

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    keywords = list(set([token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]))
    memory = get_memory_usage()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_summary": text[:100] + ("..." if len(text) > 100 else ""),
        "entities": entities,
        "keywords": keywords,
        "duration_sec": duration,
        "memory_MB": memory,
        "engine": "spaCy - en_core_web_sm"
    }

    log_to_file(log_entry)

    return {
        "entities": entities,
        "keywords": keywords,
        "summary": "To be added later",
    }

# Load Hugging Face pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

@app.post("/analyze")
async def analyze_text(text: str = Form(...)):
    # Load tokenizer and model from your saved checkpoint
    checkpoint_path = "./huggingface_models/ner_model/checkpoint-1125"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

    # Create ner pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1  # CPU
    )
    
    # Sentiment
    sentiment = sentiment_pipeline(text)[0]

    # Summary (limit length for performance)
    summary = summarizer_pipeline(text[:1024], max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    def clean_token(token):
        return {
            "entity": token["entity_group"],
            "score": float(token["score"]), 
            "word": token["word"]
        }

    # NER
    tokens = ner_pipeline(text)
    tokens = [clean_token(t) for t in tokens]

    candidate_labels = ["technology", "finance", "health", "education", "sports", "politics", "business"]
    classification = zero_shot_pipeline(text[:512], candidate_labels)

    return {
        "sentiment": sentiment,
        "summary": summary,
        "entities": tokens,
        "topics": classification
    }

# Yelp Review
@app.post("/yelp")
async def yelp_review(text: str = Form(...)):
    # Load tokenizer and model from your saved checkpoint
    checkpoint_path = "./huggingface_models/yelp_review_classifier/checkpoint-250"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    # Create the pipeline
    classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Inference example
    result = classification_pipeline(text)
    return result