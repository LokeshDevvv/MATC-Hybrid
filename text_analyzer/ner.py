"""NER analyzer using tner/roberta-large-ontonotes5.

Extracts 18 fine-grained entity types from OntoNotes 5:
  PERSON, ORG, GPE, LOC, FAC, NORP, EVENT, WORK_OF_ART, LAW, LANGUAGE,
  PRODUCT, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL.
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_ID = "tner/roberta-large-ontonotes5"

# Map OntoNotes raw tags to display-friendly types
# Some tags are uppercase like "person", "organization", etc. depending on the model.
TYPE_DISPLAY = {
    "PERSON": "PER", "person": "PER",
    "ORG": "ORG", "organization": "ORG",
    "GPE": "LOC", "geopolitical_area": "LOC",
    "LOC": "LOC", "location": "LOC",
    "FAC": "LOC", "facility": "LOC",
    "NORP": "GROUP", "group": "GROUP",
    "PRODUCT": "PROD", "product": "PROD",
    "EVENT": "EVENT", "event": "EVENT",
    "WORK_OF_ART": "WORK", "work_of_art": "WORK",
    "LAW": "LAW", "law": "LAW",
    "LANGUAGE": "LANG", "language": "LANG",
    "DATE": "DATE", "date": "DATE",
    "TIME": "TIME", "time": "TIME",
    "MONEY": "MONEY", "money": "MONEY",
    "PERCENT": "PERCENT", "percent": "PERCENT",
    "QUANTITY": "QTY", "quantity": "QTY",
    "ORDINAL": "ORDINAL", "ordinal": "ORDINAL",
    "CARDINAL": "NUM", "cardinal": "NUM",
}


class NERAnalyzer:
    def __init__(self, device=-1):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        self.pipe = pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max",
            device=device,
        )

    def analyze(self, text: str):
        results = self.pipe(text)
        entities = []
        for r in results:
            span_text = text[r["start"] : r["end"]].strip()
            if not span_text:
                continue
            raw_type = r.get("entity_group") or r.get("entity") or "MISC"
            display = TYPE_DISPLAY.get(raw_type, raw_type)
            entities.append(
                {
                    "text": span_text,
                    "type": display,
                    "score": round(float(r["score"]), 4),
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                }
            )
        return entities


if __name__ == "__main__":
    analyzer = NERAnalyzer()
    samples = [
        "Apple announced a new iPhone in Cupertino last Tuesday.",
        "Tim Cook met Elon Musk in San Francisco to discuss AI.",
        "The Eiffel Tower in Paris was built in 1889.",
        "Microsoft Research opened a new lab in Bangalore.",
    ]
    for s in samples:
        print(f"\nText: {s}")
        print(f"Entities: {analyzer.analyze(s)}")
