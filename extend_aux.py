"""Extend cached auxiliary features with 3 new branches: NER, language, sarcasm.

Inputs:  cache/v2_ag_news_{split}_5000.pt  with h_aux (47-d)
Outputs: cache/v3_ag_news_{split}_5000.pt  with h_aux (72-d)

72-d = emotion(28) + sentiment(3) + toxicity(6) + topic(10) +
       ner(4) + language(20) + sarcasm(1)
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from data.dataset import _load_raw_data
from text_analyzer.ner import NERAnalyzer
from text_analyzer.language import LanguageDetector
from text_analyzer.sarcasm import SarcasmDetector


DATASET = "ag_news"
LIMIT = 5000

# Fixed orderings for new branches
NER_TYPES = ["PER", "ORG", "LOC", "MISC"]   # 4-d
LANGUAGES = [                                 # 20-d (papluca/xlm-roberta order)
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]
# sarcasm = 1-d (probability of LABEL_1 = sarcastic)


def ner_vec(ner_analyzer, text: str):
    """4-d vector: count of each entity type, capped at 5 for stability, normalized."""
    entities = ner_analyzer.analyze(text)
    counts = {t: 0 for t in NER_TYPES}
    for e in entities:
        if e["type"] in counts:
            counts[e["type"]] += 1
    return [min(counts[t], 5) / 5.0 for t in NER_TYPES]


def language_vec(lang_detector, text: str):
    """20-d full probability distribution over papluca's 20 languages."""
    pipe = lang_detector.pipe
    out = pipe(text, top_k=None)  # list of {label, score}
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    score_map = {item["label"]: item["score"] for item in out}
    return [float(score_map.get(lang, 0.0)) for lang in LANGUAGES]


def sarcasm_vec(sarcasm_detector, text: str):
    """1-d: probability that text is sarcastic."""
    pipe = sarcasm_detector.pipe
    out = pipe(text, top_k=None)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    score_map = {item["label"]: item["score"] for item in out}
    # LABEL_1 = sarcastic
    return [float(score_map.get("LABEL_1", 0.0))]


def main():
    gpu = 0 if torch.cuda.is_available() else -1
    print("Loading 3 missing branches: NER, language, sarcasm...")
    ner = NERAnalyzer(device=gpu)
    lang = LanguageDetector(device=gpu)
    sarc = SarcasmDetector(device=gpu)

    train_texts, _, test_texts, _ = _load_raw_data(DATASET)
    train_texts = train_texts[:LIMIT]
    test_texts = test_texts[:LIMIT]

    for split, texts in [("train", train_texts), ("test", test_texts)]:
        in_path = f"cache/v2_{DATASET}_{split}_{LIMIT}.pt"
        out_path = f"cache/v3_{DATASET}_{split}_{LIMIT}.pt"
        if not os.path.exists(in_path):
            print(f"Skipping {split}: {in_path} not found")
            continue

        d = torch.load(in_path, weights_only=True)
        h_spine = d["h_matc"].float()
        h_aux_old = d["h_aux"].float()
        labels = d["labels"]
        print(f"\n=== {split}: extending aux from {h_aux_old.shape} ===")

        new_features = []
        for text in tqdm(texts, desc=f"{split} new aux"):
            text = str(text)
            v = ner_vec(ner, text) + language_vec(lang, text) + sarcasm_vec(sarc, text)
            new_features.append(v)
        new_arr = np.array(new_features, dtype=np.float32)
        print(f"  new aux: shape {new_arr.shape}, "
              f"min={new_arr.min():.3f}, max={new_arr.max():.3f}")

        h_aux_full = torch.cat(
            [h_aux_old, torch.from_numpy(new_arr)], dim=-1
        )
        print(f"  combined aux: {h_aux_full.shape}")

        torch.save(
            {
                "h_matc": h_spine,         # DistilBERT spine
                "h_aux": h_aux_full,       # 72-d
                "labels": labels,
            },
            out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
