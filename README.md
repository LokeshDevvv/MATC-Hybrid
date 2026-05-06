# MATC-Hybrid

**Multi-Aspect Text Classification through Pre-trained Expert Fusion**

A hybrid text-analysis system that combines a contextual neural backbone (DistilBERT) with seven specialised pre-trained models — each an expert in one aspect of language — through a single trained fusion classifier on top.

- **Final accuracy on AG News:** 88.40% (macro-F1: 88.28%)
- **Trainable parameters:** 566K (only the fusion head; everything else is frozen)
- **Live web app** with sentence-by-sentence Grammarly-style highlighting
- **Languages supported in analysis:** ~200 (Tamil, Hindi, Spanish, Japanese, Arabic, etc.)

---

## What it does

Reads a paragraph and tells you, **for each sentence**:

- Sentiment (positive / negative / neutral)
- Top emotions out of 28 (joy, anger, gratitude, ...)
- Topic, intent, urgency
- Whether it's sarcastic or toxic
- All named entities (people, organisations, places, dates, money, percentages, ...)
- Language

Plus a final classification combining all signals through a learned fusion head.

---

## Quick start (5 commands)

```bash
# 1. Clone
git clone https://github.com/LokeshDevvv/MATC-Hybrid.git
cd MATC-Hybrid

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn fasttext

# 4. Run the web service
uvicorn text_analyzer.serve:app --host 127.0.0.1 --port 8001

# 5. Open in browser
# http://localhost:8001/
```

**First run will take ~10 minutes** — Hugging Face downloads ~3 GB of model weights to `~/.cache/huggingface/`. Subsequent runs start in ~25 seconds.

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | none (will run on CPU) | NVIDIA GPU, 6 GB VRAM |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free (for model cache) | 10 GB |
| Python | 3.10+ | 3.10 or 3.11 |

The system is designed to run on a single laptop. CUDA is auto-detected; without a GPU, inference is just slower (~3–4× slower).

---

## Software requirements

```
torch>=2.1.0
transformers>=4.36.0
fastapi>=0.100.0
uvicorn>=0.20.0
fasttext              # for the language detector
huggingface-hub
scikit-learn
numpy, pandas, tqdm
```

The full list with pinned versions is in `requirements.txt`.

If `fasttext` fails to install on Windows, use:
```bash
pip install fasttext-wheel
```

---

## Architecture in one picture

```
                          INPUT TEXT
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐         ┌─────────┐          ┌──────────┐
   │DistilBERT│         │ 7 expert │          │ Sentence │
   │ (frozen) │         │ branches │          │ splitter │
   │   768 d  │         │ (frozen) │          │ (regex)  │
   └─────┬────┘         └─────┬────┘          └─────┬────┘
         │                    │                     │
         │   ┌────────────────┴───────────┐         │
         │  emo(28)+sent(3)+tox(6)+       │         │
         │  topic(10)+ner(4)+lang(20)+    │         │
         │  sarc(1) = 72-d aux            │         │
         │   └───────────┬────────────────┘         │
         │               │                          │
         ▼               ▼                          │
       ┌────────────────────────┐                   │
       │  Concatenate to 840-d  │                   │
       │  + BatchNorm           │                   │
       └──────────┬─────────────┘                   │
                  │                                 │
                  ▼                                 │
       ┌──────────────────────┐                     │
       │ Fusion MLP (566K)    │  ← only            │
       │ 840 → 512 → 256 → C  │    trainable        │
       └──────────┬───────────┘                     │
                  │                                 │
                  ▼                                 ▼
            CLASS PREDICTION              PER-SENTENCE OUTPUT
                                          (web UI)
```

---

## Project layout

```
MATC-Hybrid/
├── README.md                     # This file
├── PROJECT_REPORT.md             # Full academic report (3K words)
├── VIVA_REPORT.md                # Viva Q&A bank (8K words, 66 questions)
├── PROGRESS.md                   # Build log
│
├── frontend/
│   └── index.html                # Web UI (vanilla HTML/CSS/JS)
│
├── text_analyzer/                # Backend service
│   ├── serve.py                  # FastAPI server
│   ├── cli.py                    # Command-line one-shot interface
│   ├── emotion.py                # ModernBERT-large GoEmotions
│   ├── sentiment.py              # tabularisai multilingual sentiment
│   ├── toxicity.py               # unitary toxic-bert
│   ├── ner.py                    # tner roberta-large OntoNotes-5
│   ├── language.py               # facebook fasttext (200 langs)
│   ├── sarcasm.py                # cardiffnlp twitter irony
│   └── zeroshot.py               # MoritzLaurer DeBERTa-v3-large
│
├── models/
│   └── hybrid_matc.py            # Fusion-head architecture
│
├── results/                      # Trained fusion-head checkpoints
│   ├── hybrid_v3_ag_news.pt      # Best model (use this)
│   └── ablation_results.json     # Per-branch ablation table
│
├── extract_features_v2.py        # Pre-compute feature vectors → cache/
├── extend_aux.py                 # Add NER/language/sarcasm to aux features
├── train_hybrid.py               # Train the fusion head on cached features
├── run_ablations.py              # Run all 9 ablation configurations
├── hybrid_inference.py           # End-to-end inference
│
├── tests/
│   └── edge_cases.py             # Edge-case test suite
│
└── requirements.txt
```

---

## Running it

### Option A — Live web app (most common)

```bash
uvicorn text_analyzer.serve:app --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001/** in any browser.

The UI:
- Paste a paragraph, click Analyze
- See per-sentence colour-coded analysis
- Toggle "Auto" for live as-you-type analysis
- Try the sample-text chips for quick demos

### Option B — Command line

```bash
python -m text_analyzer.cli "Your text here"
```

Returns structured JSON.

Interactive mode (no argument):
```bash
python -m text_analyzer.cli
```

### Option C — REST API

```bash
# Single text
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Whole paragraph (per-sentence breakdown)
curl -X POST http://localhost:8001/analyze_paragraph \
  -H "Content-Type: application/json" \
  -d '{"text": "I waited 45 minutes. The waiter was rude. But the dessert was amazing."}'
```

Auto-generated API docs: **http://localhost:8001/docs**

---

## Reproducing the AG News result

The trained fusion head (`results/hybrid_v3_ag_news.pt`) is already in the repo. To reproduce its training from scratch:

```bash
# 1. Extract spine + aux features for all training samples (~25 min on GPU, longer on CPU)
python extract_features_v2.py --dataset ag_news --limit 5000 --batch_size 32

# 2. Add the 3 extra aux branches (~5 min)
python extend_aux.py

# 3. Train the fusion head (~3 min)
python train_hybrid.py --dataset v3_ag_news --limit_suffix _5000 --epochs 30

# 4. Run all 9 ablation configurations (~30 min)
python run_ablations.py
```

Expected result: **~88% accuracy / ~88% macro-F1** on AG News test set.

---

## Test it

Single text:
```bash
python -m text_analyzer.cli "Apple announced a 40% revenue jump on Tuesday."
```

Edge-case suite:
```bash
python tests/edge_cases.py     # ~50 tests covering empty input, multilingual,
                                # sarcasm, toxicity, code-switching, etc.
```

---

## Known limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Latency: ~25 s per paragraph | Slow demos | Future: batch+parallel inference (~1.5 s achievable) |
| Single random seed | Variance unknown | Future: 3–5 seed runs |
| Trained on 5K subset of AG News | Slight accuracy ceiling | Future: full 120K dataset |
| Fusion classifier is English-only | Trained classifier output won't make sense for non-English text | The analysis pipeline (sentiment, language, etc.) still works for any language |
| Sarcasm model occasionally false-fires | One signal of eight | Used as a hint, not a hard rule |

These are documented in `VIVA_REPORT.md` § 12.

---

## Documents

| File | What it covers | Length |
|------|----------------|--------|
| `README.md` | This file — quick start + architecture | ~250 lines |
| `PROJECT_REPORT.md` | Academic-style report with 21 sections (Abstract → References) | ~500 lines |
| `VIVA_REPORT.md` | Viva preparation: 66 possible questions answered in plain English | ~760 lines |
| `PROGRESS.md` | Original build log of the underlying MATC-Net project | ~200 lines |

---

## Citation

```bibtex
@misc{matc-hybrid-2026,
  author = {Lokesh S and NC Gautham and Shanjo J Benadict},
  title  = {MATC-Hybrid: Multi-Aspect Text Classification through Pre-trained Expert Fusion},
  year   = {2026},
  url    = {https://github.com/LokeshDevvv/MATC-Hybrid}
}
```

---

## License

Educational and research use. Individual underlying models retain their original licenses (most are MIT or Apache 2.0; check Hugging Face model cards for specifics).
