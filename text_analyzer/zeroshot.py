"""Zero-shot classifier using MoritzLaurer/deberta-v3-base-zeroshot-v2.0.

Classify text against any custom set of labels with no training.
Used for topic, intent, urgency, complaint type, etc.
"""

from transformers import pipeline

MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

DEFAULT_TOPICS = [
    "technology",
    "politics",
    "sports",
    "business",
    "entertainment",
    "health",
    "science",
    "education",
    "travel",
    "food",
]

DEFAULT_INTENTS = [
    "complaint",
    "compliment",
    "question",
    "request",
    "statement",
    "greeting",
    "farewell",
]

DEFAULT_URGENCY = ["urgent", "important", "normal", "low priority"]


class ZeroShotClassifier:
    def __init__(self, device=-1):
        self.pipe = pipeline(
            task="zero-shot-classification",
            model=MODEL_ID,
            device=device,
        )

    def analyze(self, text: str, candidate_labels=None, multi_label=False):
        labels = candidate_labels or DEFAULT_TOPICS
        result = self.pipe(text, candidate_labels=labels, multi_label=multi_label)
        return [
            {"label": l, "score": round(s, 4)}
            for l, s in zip(result["labels"], result["scores"])
        ]

    def topic(self, text: str):
        return self.analyze(text, DEFAULT_TOPICS, multi_label=False)[0]

    def intent(self, text: str):
        return self.analyze(text, DEFAULT_INTENTS, multi_label=False)[0]

    def urgency(self, text: str):
        return self.analyze(text, DEFAULT_URGENCY, multi_label=False)[0]


if __name__ == "__main__":
    classifier = ZeroShotClassifier()
    samples = [
        "The new iPhone has a better camera than last year's model.",
        "I want to book a flight to Tokyo next month.",
        "URGENT: server down, all customers affected, need help now!",
        "The President signed a new bill into law today.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Topic:   {classifier.topic(s)}")
        print(f"Intent:  {classifier.intent(s)}")
        print(f"Urgency: {classifier.urgency(s)}")
