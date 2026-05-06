"""Pre-compute hybrid features for all samples in a dataset.

Extracts:
  - h_matc: 768-d MATC-Net spine embedding
  - h_aux:  47-d concatenated auxiliary probabilities (emotion + sentiment + toxicity + topic)

Caches as a single .pt file per split. The fusion head is then trained
on these cached features (very fast — no upstream model calls during training).
"""

import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer

from utils.helpers import load_config, get_device
from data.dataset import DATASET_INFO, _load_raw_data
from data.label_graph import build_label_graph
from models.matc_net import MATCNet

from text_analyzer.emotion import EmotionAnalyzer, MODEL_ID as EMO_MODEL
from text_analyzer.sentiment import SentimentAnalyzer
from text_analyzer.toxicity import ToxicityAnalyzer
from text_analyzer.zeroshot import ZeroShotClassifier, DEFAULT_TOPICS

# Stable label ordering for the 28 GoEmotions classes (matches model output)
EMOTION_LABELS_ORDER = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
SENTIMENT_LABELS_ORDER = ["negative", "neutral", "positive"]
TOXICITY_LABELS_ORDER = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
]
# DEFAULT_TOPICS used as-is for zeroshot (10 topics)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ag_news",
                   choices=["ag_news", "imdb", "20newsgroups", "bbc"])
    p.add_argument("--matc_checkpoint", type=str,
                   default="results/best_model_ag_news.pt")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="optional: limit number of samples per split for testing")
    return p.parse_args()


def _emotion_dense_vector(pipe_output, label_order):
    """pipe_output is a list of dicts; convert to fixed-order vector."""
    score_map = {item["label"]: item["score"] for item in pipe_output}
    return [float(score_map.get(lbl, 0.0)) for lbl in label_order]


@torch.no_grad()
def extract_matc_embeddings(matc_net, tokenizer, texts, max_len, batch_size, device):
    """Run MATC-Net on all texts, return (N, 768) numpy array of h_fused."""
    matc_net.eval()
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="MATC-Net"):
        batch = [str(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(batch, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        _, _, h_fused = matc_net(enc["input_ids"], enc["attention_mask"])
        embeds.append(h_fused.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def _norm_label(s: str) -> str:
    s = s.lower()
    return {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(s, s)


def extract_aux_features(emo, sent, tox, zs, texts):
    """Run pre-trained branches and concatenate per-sample feature vectors."""
    aux = []
    for text in tqdm(texts, desc="Aux branches"):
        text = str(text)

        # Emotion: pipe with top_k=None returns [[{label, score}, ...]]
        emo_result = emo.pipe(text)[0]
        emo_vec = _emotion_dense_vector(emo_result, EMOTION_LABELS_ORDER)

        # Sentiment: pipe with top_k=None on sentiment-analysis task returns flat list
        sent_result = sent.pipe(text, top_k=None)
        sent_map = {_norm_label(item["label"]): item["score"] for item in sent_result}
        sent_vec = [float(sent_map.get(lbl, 0.0)) for lbl in SENTIMENT_LABELS_ORDER]

        # Toxicity: returns dict
        tox_result = tox.analyze(text)
        tox_vec = [float(tox_result.get(lbl, 0.0)) for lbl in TOXICITY_LABELS_ORDER]

        # Zero-shot topic
        zs_result = zs.analyze(text, candidate_labels=DEFAULT_TOPICS, multi_label=False)
        zs_map = {item["label"]: item["score"] for item in zs_result}
        zs_vec = [float(zs_map.get(lbl, 0.0)) for lbl in DEFAULT_TOPICS]

        aux.append(emo_vec + sent_vec + tox_vec + zs_vec)
    return np.array(aux, dtype=np.float32)


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    device = get_device()
    config = load_config(args.config)

    info = DATASET_INFO[args.dataset]

    # Load MATC-Net (frozen)
    edge_index = build_label_graph(args.dataset)
    matc = MATCNet(
        config=config,
        num_classes=info["num_classes"],
        edge_index=edge_index,
        class_names=info["class_names"],
    ).to(device)
    state = torch.load(args.matc_checkpoint, map_location=device, weights_only=True)
    matc.load_state_dict(state)
    print(f"Loaded MATC-Net checkpoint from {args.matc_checkpoint}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load pre-trained branches (frozen, on GPU if available)
    gpu_dev = 0 if torch.cuda.is_available() else -1
    print("\nLoading pre-trained auxiliary branches...")
    emo = EmotionAnalyzer(device=gpu_dev)
    sent = SentimentAnalyzer(device=gpu_dev)
    tox = ToxicityAnalyzer()
    zs = ZeroShotClassifier(device=gpu_dev)

    # Load raw data
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
        out = os.path.join(args.cache_dir, f"{args.dataset}_{split}{suffix}.pt")
        if os.path.exists(out):
            print(f"\n{out} exists, skipping.")
            continue

        print(f"\n=== Extracting features for {split} ({len(texts)} samples) ===")
        t0 = time.time()
        h_matc = extract_matc_embeddings(matc, tokenizer, texts,
                                         args.max_seq_length, args.batch_size, device)
        h_aux = extract_aux_features(emo, sent, tox, zs, texts)
        labels_arr = np.array(labels, dtype=np.int64)

        torch.save(
            {
                "h_matc": torch.from_numpy(h_matc),
                "h_aux": torch.from_numpy(h_aux),
                "labels": torch.from_numpy(labels_arr),
            },
            out,
        )
        print(f"Saved {out}: h_matc={h_matc.shape}, h_aux={h_aux.shape} "
              f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
