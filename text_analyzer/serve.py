"""Unified text analysis FastAPI service.

POST /analyze
    { "text": "your text here" }
returns:
    {
        "language": {...},
        "sentiment": {...},
        "emotions": [...],
        "toxicity": {...},
        "entities": [...],
        "sarcasm": {...},
        "topic": {...},
        "intent": {...},
        "urgency": {...},
        "elapsed_ms": ...
    }

Usage:
    uvicorn text_analyzer.serve:app --host 0.0.0.0 --port 8000
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .emotion import EmotionAnalyzer
from .sentiment import SentimentAnalyzer
from .toxicity import ToxicityAnalyzer
from .language import LanguageDetector
from .ner import NERAnalyzer
from .sarcasm import SarcasmDetector
from .zeroshot import ZeroShotClassifier


app = FastAPI(
    title="LECTOR — Linguistic Instrument",
    description="Multi-aspect text analysis: emotion, sentiment, toxicity, NER, sarcasm, language, topic, intent, urgency",
    version="1.0.0",
)

# CORS — allow the frontend to call from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    include: Optional[List[str]] = None  # subset of analyzers to run


class _State:
    emotion = None
    sentiment = None
    toxicity = None
    language = None
    ner = None
    sarcasm = None
    zeroshot = None


@app.on_event("startup")
def load_models():
    print("Loading models — this takes ~30 seconds...")
    t0 = time.time()
    _State.language = LanguageDetector()
    _State.sentiment = SentimentAnalyzer()
    _State.emotion = EmotionAnalyzer()
    _State.toxicity = ToxicityAnalyzer()
    _State.ner = NERAnalyzer()
    _State.sarcasm = SarcasmDetector()
    _State.zeroshot = ZeroShotClassifier()
    print(f"All models loaded in {time.time() - t0:.1f}s")


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)


@app.get("/")
def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "endpoints": ["/analyze (POST)", "/health (GET)"]}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": all(
            getattr(_State, k) is not None
            for k in ["emotion", "sentiment", "toxicity", "language", "ner", "sarcasm", "zeroshot"]
        ),
    }


# ─── Script-based language override (handles Tamil/Telugu/etc that the
#     20-language model doesn't know natively) ─────────────────────────
_SCRIPT_RANGES = [
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("pa", 0x0A00, 0x0A7F),  # Punjabi (Gurmukhi)
    ("or", 0x0B00, 0x0B7F),  # Odia
    ("si", 0x0D80, 0x0DFF),  # Sinhala
    ("my", 0x1000, 0x109F),  # Burmese
    ("km", 0x1780, 0x17FF),  # Khmer
    ("ko", 0xAC00, 0xD7AF),  # Korean Hangul
]


def _script_detect(text: str):
    """Return (lang_code, ratio) if script-specific characters dominate, else None."""
    counts = {}
    letters = 0
    for ch in text:
        c = ord(ch)
        if ch.isalpha():
            letters += 1
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= c <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts or letters == 0:
        return None
    code, n = max(counts.items(), key=lambda kv: kv[1])
    ratio = min(n / max(letters, 1), 1.0)
    if ratio >= 0.3:
        return code, ratio
    return None


def _detect_language(text: str) -> dict:
    """Detect language with script-based override for unsupported languages."""
    raw = _State.language.analyze(text)
    override = _script_detect(text)
    if override:
        code, ratio = override
        if code != raw.get("language"):
            return {"language": code, "score": round(ratio, 4)}
    return raw


def _analyse_one(text: str) -> dict:
    """Run all analyzers on a single text snippet."""
    return {
        "sentiment": _State.sentiment.analyze(text),
        "emotions": _State.emotion.analyze(text),
        "toxicity": _State.toxicity.analyze(text),
        "entities": _State.ner.analyze(text),
        "sarcasm": _State.sarcasm.analyze(text),
        "topic": _State.zeroshot.topic(text),
        "intent": _State.zeroshot.intent(text),
        "urgency": _State.zeroshot.urgency(text),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        return {"error": "empty text"}

    t0 = time.time()
    out = _analyse_one(text)
    out["language"] = _State.language.analyze(text)
    out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return out


# ─── Sentence-level analysis ───────────────────────────────────────────
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_sentences(text: str) -> list:
    """Cheap sentence splitter. Splits on .!? followed by whitespace + capital."""
    text = text.strip()
    if not text:
        return []
    # Split on paragraph breaks first, then on sentence boundaries
    chunks = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                chunks.append(s)
    return chunks


@app.post("/analyze_paragraph")
def analyze_paragraph(req: AnalyzeRequest):
    """Split paragraph into sentences and analyse each individually.

    Returns:
      {
        "language":  {...},
        "sentences": [
          { "text": "...", "sentiment": {...}, "emotions": [...], ... },
          ...
        ],
        "summary": { dominant_sentiment, dominant_emotion, ... },
        "elapsed_ms": ...
      }
    """
    text = req.text.strip()
    if not text:
        return {"error": "empty text"}

    t0 = time.time()
    sentences = split_sentences(text)
    if not sentences:
        return {"error": "no sentences found"}

    per_sent = []
    for s in sentences:
        analysis = _analyse_one(s)
        analysis["text"] = s
        per_sent.append(analysis)

    # Language detected once on full text (more reliable on more context)
    language = _State.language.analyze(text)

    # Build a small summary across sentences
    sentiments = [s["sentiment"]["label"] for s in per_sent]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    if pos > neg and pos > neu:
        dom_sent = "positive"
    elif neg > pos and neg > neu:
        dom_sent = "negative"
    else:
        dom_sent = "neutral"

    # Top emotion across sentences (sum scores)
    emo_sum = {}
    for s in per_sent:
        for e in s.get("emotions", []) or []:
            emo_sum[e["label"]] = emo_sum.get(e["label"], 0.0) + e["score"]
    top_emotion = max(emo_sum.items(), key=lambda kv: kv[1])[0] if emo_sum else None

    any_toxic = any(s.get("toxicity", {}).get("is_toxic") for s in per_sent)
    any_sarcastic = any(s.get("sarcasm", {}).get("is_sarcastic") for s in per_sent)

    return {
        "language": language,
        "sentences": per_sent,
        "summary": {
            "sentence_count": len(per_sent),
            "dominant_sentiment": dom_sent,
            "sentiment_breakdown": {"positive": pos, "negative": neg, "neutral": neu},
            "dominant_emotion": top_emotion,
            "any_toxic": any_toxic,
            "any_sarcastic": any_sarcastic,
        },
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
