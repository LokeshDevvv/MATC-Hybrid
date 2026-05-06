"""Replace broken MATC-Net spine embeddings with DistilBERT in cached files.

The aux branches are fine — only the spine needs replacement.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from data.dataset import _load_raw_data

SPINE_MODEL = "distilbert-base-uncased"
DATASET = "ag_news"
LIMIT = 5000
MAX_LEN = 128
BATCH_SIZE = 32


@torch.no_grad()
def extract_cls(model, tokenizer, texts, device):
    model.eval()
    embeds = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="DistilBERT"):
        batch = [str(t) for t in texts[i : i + BATCH_SIZE]]
        enc = tokenizer(batch, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        embeds.append(cls.cpu())
    return torch.cat(embeds, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {SPINE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SPINE_MODEL)
    spine = AutoModel.from_pretrained(SPINE_MODEL).to(device)

    train_texts, _, test_texts, _ = _load_raw_data(DATASET)
    train_texts = train_texts[:LIMIT]
    test_texts = test_texts[:LIMIT]

    for split, texts in [("train", train_texts), ("test", test_texts)]:
        in_path = f"cache/{DATASET}_{split}_{LIMIT}.pt"
        out_path = f"cache/v2_{DATASET}_{split}_{LIMIT}.pt"
        if not os.path.exists(in_path):
            print(f"Skipping {split}: {in_path} not found")
            continue
        d = torch.load(in_path, weights_only=True)
        print(f"Loaded {in_path}: aux shape {d['h_aux'].shape}")

        new_spine = extract_cls(spine, tokenizer, texts, device)
        print(f"  spine NaN: {torch.isnan(new_spine).any().item()}, shape: {new_spine.shape}")

        torch.save(
            {
                "h_matc": new_spine.float(),  # keep field name for compatibility
                "h_aux": d["h_aux"].float(),
                "labels": d["labels"],
            },
            out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
