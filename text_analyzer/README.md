# Text Analyzer

Multi-aspect text analysis using pre-trained Hugging Face models. No training required — all models are downloaded on first use.

## What it analyzes

| Aspect | Model | Output |
|---|---|---|
| Language | papluca/xlm-roberta-base-language-detection | One of 20 languages |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment-latest | positive / negative / neutral |
| Emotions | SamLowe/roberta-base-go_emotions | Up to 5 emotions from 28 labels (multi-label) |
| Toxicity | unitary/toxic-bert | 6 sublabels: toxic, obscene, threat, insult, identity_hate, severe_toxic |
| Entities (NER) | dslim/bert-base-NER | PER, ORG, LOC, MISC |
| Sarcasm | helinivan/english-sarcasm-detector | Binary (works best on headline-style English) |
| Topic | MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | Zero-shot from 10 default topics |
| Intent | (same model) | Zero-shot: complaint, compliment, question, request, statement, greeting, farewell |
| Urgency | (same model) | Zero-shot: urgent, important, normal, low priority |

## Run as a service

```bash
uvicorn text_analyzer.serve:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: server is down"}'
```

## Run from the command line

```bash
python -m text_analyzer.cli "your text here"
python -m text_analyzer.cli                  # interactive mode
```

## Run a single analyzer for testing

```bash
python -m text_analyzer.emotion
python -m text_analyzer.sentiment
python -m text_analyzer.toxicity
python -m text_analyzer.language
python -m text_analyzer.ner
python -m text_analyzer.sarcasm
python -m text_analyzer.zeroshot
```

## Performance

- **Cold start**: ~25s to load all 7 models
- **Per request**: ~7s on CPU, ~1-2s on a single GPU
- **Memory**: ~3 GB total model footprint

## Known limitations

- Sarcasm model is trained on news headlines — degrades on conversational text.
- NER recognises only PER/ORG/LOC/MISC, no fine-grained types like dates/money.
- Emotion threshold default is 0.3 — tune in `emotion.py` if you want more or fewer labels per text.
- All models are English-first except language detection. For multilingual emotion/sentiment, swap to multilingual variants.
