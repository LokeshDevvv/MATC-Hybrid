"""Single-shot CLI for the text analyzer (no FastAPI required).

Usage:
    python -m text_analyzer.cli "your text here"
    python -m text_analyzer.cli  # interactive mode
"""

import json
import sys
import time

from .emotion import EmotionAnalyzer
from .sentiment import SentimentAnalyzer
from .toxicity import ToxicityAnalyzer
from .language import LanguageDetector
from .ner import NERAnalyzer
from .sarcasm import SarcasmDetector
from .zeroshot import ZeroShotClassifier


def build_pipeline():
    print("Loading models...")
    t0 = time.time()
    pipeline = {
        "language": LanguageDetector(),
        "sentiment": SentimentAnalyzer(),
        "emotion": EmotionAnalyzer(),
        "toxicity": ToxicityAnalyzer(),
        "ner": NERAnalyzer(),
        "sarcasm": SarcasmDetector(),
        "zeroshot": ZeroShotClassifier(),
    }
    print(f"All models loaded in {time.time() - t0:.1f}s\n")
    return pipeline


def analyze_text(text, p):
    t0 = time.time()
    out = {
        "text": text,
        "language": p["language"].analyze(text),
        "sentiment": p["sentiment"].analyze(text),
        "emotions": p["emotion"].analyze(text),
        "toxicity": p["toxicity"].analyze(text),
        "entities": p["ner"].analyze(text),
        "sarcasm": p["sarcasm"].analyze(text),
        "topic": p["zeroshot"].topic(text),
        "intent": p["zeroshot"].intent(text),
        "urgency": p["zeroshot"].urgency(text),
    }
    out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return out


def main():
    p = build_pipeline()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(json.dumps(analyze_text(text, p), indent=2, ensure_ascii=False))
    else:
        print("Interactive mode. Type text and press Enter (Ctrl+C to exit).")
        try:
            while True:
                text = input("\n> ").strip()
                if not text:
                    continue
                print(json.dumps(analyze_text(text, p), indent=2, ensure_ascii=False))
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")


if __name__ == "__main__":
    main()
