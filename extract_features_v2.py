"""Pre-compute hybrid features using DistilBERT spine (since MATC-Net checkpoint is broken).

Spine: DistilBERT CLS embedding (768-d)
Aux:   emotion (28) + sentiment (3) + toxicity (6) + topic (10) = 47-d

Usage:
    python extract_features_v2.py --dataset ag_news --limit 5000 --batch_size 32
"""

import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from data.dataset import DATASET_INFO, _load_raw_data
from text_analyzer.emotion import EmotionAnalyzer
from text_analyzer.sentiment import SentimentAnalyzer
from text_analyzer.toxicity import ToxicityAnalyzer
from text_analyzer.zeroshot import ZeroShotClassifier, DEFAULT_TOPICS

from extract_features import (
    EMOTION_LABELS_ORDER,
    SENTIMENT_LABELS_ORDER,
    TOXICITY_LABELS_ORDER,
    _emotion_dense_vector,
    _norm_label,
    extract_aux_features,
)


SPINE_MODEL = "distilbert-base-uncased"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def extract_distilbert_embeddings(model, tokenizer, texts, max_len, batch_size, device):
    model.eval()
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT spine"):
        batch = [str(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(batch, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        # Use [CLS] token (index 0 of last_hidden_state)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        embeds.append(cls.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = DATASET_INFO[args.dataset]

    # Load DistilBERT spine
    print(f"Loading {SPINE_MODEL} spine...")
    tokenizer = AutoTokenizer.from_pretrained(SPINE_MODEL)
    spine = AutoModel.from_pretrained(SPINE_MODEL).to(device)

    # Load pre-trained branches (frozen, on GPU)
    gpu_dev = 0 if torch.cuda.is_available() else -1
    print("Loading pre-trained auxiliary branches...")
    emo = EmotionAnalyzer(device=gpu_dev)
    sent = SentimentAnalyzer(device=gpu_dev)
    tox = ToxicityAnalyzer()
    zs = ZeroShotClassifier(device=gpu_dev)

    train_texts, train_labels, test_texts, test_labels = _load_raw_data(args.dataset)
    if args.limit:
        train_texts = train_texts[: args.limit]
        train_labels = train_labels[: args.limit]
        test_texts = test_texts[: args.limit]
        test_labels = test_labels[: args.limit]

    for split, texts, labels in [
        ("train", train_texts, train_labels),
        ("test", test_texts, test_labels),
    ]:
        suffix = f"_{args.limit}" if args.limit else ""
        out = os.path.join(args.cache_dir, f"v2_{args.dataset}_{split}{suffix}.pt")
        if os.path.exists(out):
            print(f"\n{out} exists, skipping.")
            continue

        print(f"\n=== Extracting features for {split} ({len(texts)} samples) ===")
        t0 = time.time()
        h_spine = extract_distilbert_embeddings(spine, tokenizer, texts,
                                                args.max_seq_length, args.batch_size, device)
        h_aux = extract_aux_features(emo, sent, tox, zs, texts)
        labels_arr = np.array(labels, dtype=np.int64)

        torch.save(
            {
                "h_matc": torch.from_numpy(h_spine),  # naming kept for compatibility
                "h_aux": torch.from_numpy(h_aux),
                "labels": torch.from_numpy(labels_arr),
            },
            out,
        )
        print(f"Saved {out}: spine={h_spine.shape}, aux={h_aux.shape} "
              f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
