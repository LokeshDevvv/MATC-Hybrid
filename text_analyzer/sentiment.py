"""Sentiment analyzer using cardiffnlp/twitter-roberta-base-sentiment-latest.

Returns positive / negative / neutral with confidence.
"""

from transformers import pipeline

MODEL_ID = "tabularisai/multilingual-sentiment-analysis"

# tabularisai outputs 5 classes: Very Negative / Negative / Neutral / Positive / Very Positive
# We collapse to 3 classes for the UI but keep the raw label too.
LABEL_MAP = {
    # title case (model output)
    "Very Negative": "negative",
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "Very Positive": "positive",
    # lowercase
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive",
    # legacy LABEL_X
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class SentimentAnalyzer:
    def __init__(self, device=-1):
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=MODEL_ID,
            device=device,
        )

    def analyze(self, text: str):
        result = self.pipe(text)[0]
        raw = result["label"]
        # Try exact match first, then lowercase, then title-case
        label = (
            LABEL_MAP.get(raw)
            or LABEL_MAP.get(raw.lower())
            or LABEL_MAP.get(raw.title())
            or raw.lower()
        )
        return {"label": label, "score": round(result["score"], 4)}


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    samples = [
        "I love this product, it works perfectly!",
        "Terrible experience, would not recommend.",
        "The package arrived on Tuesday.",
        "Mixed feelings - some parts were great but others were poor.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Sentiment: {analyzer.analyze(s)}")
