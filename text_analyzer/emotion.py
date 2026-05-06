"""Emotion analyzer using SamLowe/roberta-base-go_emotions.

Detects 28 fine-grained emotions in a multi-label fashion.
"""

from transformers import pipeline

MODEL_ID = "cirimus/modernbert-large-go-emotions"


class EmotionAnalyzer:
    def __init__(self, device=-1, threshold=0.3, top_k=5):
        self.threshold = threshold
        self.top_k = top_k
        self.pipe = pipeline(
            task="text-classification",
            model=MODEL_ID,
            top_k=None,
            device=device,
        )

    def analyze(self, text: str):
        results = self.pipe(text)[0]
        # Filter by threshold and take top_k
        filtered = [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
            if r["score"] >= self.threshold
        ]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered[: self.top_k]


if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    samples = [
        "I can't believe how good this is, thank you so much!",
        "I waited 45 minutes and the soup was cold. Worst service ever.",
        "It's just an ordinary day, nothing special.",
        "I'm so scared of what's about to happen tomorrow.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Emotions: {analyzer.analyze(s)}")
