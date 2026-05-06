"""Toxicity analyzer using unitary/toxic-bert.

Detects: toxic, severe_toxic, obscene, threat, insult, identity_hate.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "unitary/toxic-bert"


class ToxicityAnalyzer:
    def __init__(self, device=None, threshold=0.5):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(
            self.device
        )
        self.model.eval()
        self.labels = list(self.model.config.id2label.values())

    @torch.no_grad()
    def analyze(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        logits = self.model(**inputs).logits
        scores = torch.sigmoid(logits)[0].cpu().tolist()
        result = {self.labels[i]: round(scores[i], 4) for i in range(len(self.labels))}
        result["is_toxic"] = result.get("toxic", 0.0) >= self.threshold
        return result


if __name__ == "__main__":
    analyzer = ToxicityAnalyzer()
    samples = [
        "I love this community, everyone is so helpful!",
        "You are an idiot and I hate you.",
        "This is just a normal sentence about coding.",
        "Get out of here, nobody wants you around.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Toxicity: {analyzer.analyze(s)}")
