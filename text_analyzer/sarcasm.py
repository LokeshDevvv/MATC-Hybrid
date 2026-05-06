"""Irony / sarcasm detector using cardiffnlp/twitter-roberta-base-irony.

Binary classification: ironic vs. not ironic. Trained on Twitter data,
so it works on conversational text — much better than headline-trained models.

Output labels from this model:
  LABEL_0 → non_irony
  LABEL_1 → irony
"""

from transformers import pipeline

MODEL_ID = "cardiffnlp/twitter-roberta-base-irony"


class SarcasmDetector:
    def __init__(self, device=-1):
        self.pipe = pipeline(
            task="text-classification",
            model=MODEL_ID,
            device=device,
        )

    def analyze(self, text: str):
        result = self.pipe(text)[0]
        label = result["label"]
        is_sarcastic = label in (
            "LABEL_1", "1", "irony", "ironic", "sarcastic", "SARCASTIC"
        )
        return {
            "is_sarcastic": bool(is_sarcastic),
            "score": round(float(result["score"]), 4),
        }


if __name__ == "__main__":
    detector = SarcasmDetector()
    samples = [
        "Oh great, another Monday morning meeting. Just what I needed.",
        "I really enjoyed the concert last night, it was wonderful.",
        "Wow, traffic on Friday afternoon, what a surprise.",
        "The package arrived on time and the product was exactly as described.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Sarcasm: {detector.analyze(s)}")
