"""End-to-end inference for HybridMATC.

Given raw text, runs:
  - MATC-Net spine -> 768-d
  - Pre-trained branches (emotion + sentiment + toxicity + topic) -> 47-d
  - Fusion head -> class logits

Usage:
  python hybrid_inference.py --text "Apple announced a new iPhone today" \
                             --dataset ag_news \
                             --matc_checkpoint results/best_model_ag_news.pt \
                             --hybrid_checkpoint results/hybrid_ag_news.pt
"""

import argparse
import torch
import numpy as np
from transformers import RobertaTokenizer

from utils.helpers import load_config, get_device
from data.dataset import DATASET_INFO
from data.label_graph import build_label_graph
from models.matc_net import MATCNet
from models.hybrid_matc import HybridMATC, AUX_TOTAL_DIM

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
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--matc_checkpoint", type=str,
                   default="results/best_model_ag_news.pt")
    p.add_argument("--hybrid_checkpoint", type=str,
                   default="results/hybrid_ag_news.pt")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--max_seq_length", type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def encode_text(text, matc, tokenizer, max_len, device):
    enc = tokenizer(text, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    _, _, h = matc(enc["input_ids"], enc["attention_mask"])
    return h  # (1, 768)


def aux_features(text, emo, sent, tox, zs):
    emo_result = emo.pipe(text)[0]
    emo_vec = _emotion_dense_vector(emo_result, EMOTION_LABELS_ORDER)

    sent_result = sent.pipe(text, top_k=None)
    sent_map = {_norm_label(item["label"]): item["score"] for item in sent_result}
    sent_vec = [float(sent_map.get(lbl, 0.0)) for lbl in SENTIMENT_LABELS_ORDER]

    tox_result = tox.analyze(text)
    tox_vec = [float(tox_result.get(lbl, 0.0)) for lbl in TOXICITY_LABELS_ORDER]

    zs_result = zs.analyze(text, candidate_labels=DEFAULT_TOPICS)
    zs_map = {item["label"]: item["score"] for item in zs_result}
    zs_vec = [float(zs_map.get(lbl, 0.0)) for lbl in DEFAULT_TOPICS]

    return np.array(emo_vec + sent_vec + tox_vec + zs_vec, dtype=np.float32)


def main():
    args = parse_args()
    device = get_device()
    info = DATASET_INFO[args.dataset]

    # Load MATC-Net spine
    config = load_config(args.config)
    edge_index = build_label_graph(args.dataset)
    matc = MATCNet(
        config=config,
        num_classes=info["num_classes"],
        edge_index=edge_index,
        class_names=info["class_names"],
    ).to(device)
    matc.load_state_dict(torch.load(args.matc_checkpoint, map_location=device,
                                    weights_only=True))
    matc.eval()
    print(f"Loaded MATC-Net from {args.matc_checkpoint}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load pre-trained aux branches
    gpu = 0 if torch.cuda.is_available() else -1
    print("Loading auxiliary branches...")
    emo = EmotionAnalyzer(device=gpu)
    sent = SentimentAnalyzer(device=gpu)
    tox = ToxicityAnalyzer()
    zs = ZeroShotClassifier(device=gpu)

    # Load hybrid head
    hybrid = HybridMATC(num_classes=info["num_classes"]).to(device)
    hybrid.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device,
                                      weights_only=True))
    hybrid.eval()
    print(f"Loaded hybrid head from {args.hybrid_checkpoint}")

    # Encode + classify
    h_matc = encode_text(args.text, matc, tokenizer, args.max_seq_length, device)
    h_aux_np = aux_features(args.text, emo, sent, tox, zs)
    h_aux = torch.from_numpy(h_aux_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = hybrid(h_matc, h_aux)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    print(f"\nInput: \"{args.text}\"")
    print("\nClass probabilities:")
    for i, prob in enumerate(probs.tolist()):
        bar = "#" * int(prob * 40)
        print(f"  {info['class_names'][i]:20s} {prob:.4f}  {bar}")

    top = probs.argmax().item()
    print(f"\nPredicted: {info['class_names'][top]} ({probs[top].item():.4f})")


if __name__ == "__main__":
    main()
