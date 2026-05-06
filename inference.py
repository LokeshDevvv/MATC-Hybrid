"""Single-text inference for MATC-Net.

Usage:
    python inference.py --dataset ag_news --model_path results/best_model_ag_news.pt \
                        --text "Apple announced a new iPhone today"

    python inference.py --dataset imdb --model_path results/best_model_imdb.pt \
                        --text "This movie was absolutely fantastic. Great acting!"
"""

import argparse
import torch
from transformers import RobertaTokenizer

from utils.helpers import load_config, get_device
from data.dataset import DATASET_INFO
from data.label_graph import build_label_graph
from models.matc_net import MATCNet


def parse_args():
    parser = argparse.ArgumentParser(description="MATC-Net Inference")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text to classify")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ag_news", "imdb", "20newsgroups", "bbc"],
                        help="Dataset the model was trained on (determines classes)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model weights")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--ablation", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict(text, model, tokenizer, class_names, device, max_seq_length=512):
    """Classify a single text and return predictions with confidence scores."""
    model.eval()

    encoding = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits, z, h_fused = model(input_ids, attention_mask)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    # Top predictions
    sorted_indices = probabilities.argsort(descending=True)
    results = []
    for idx in sorted_indices:
        results.append({
            "class": class_names[idx.item()],
            "confidence": probabilities[idx].item(),
        })

    return results


def main():
    args = parse_args()

    config = load_config(args.config)
    device = get_device()

    info = DATASET_INFO[args.dataset]
    num_classes = info["num_classes"]
    class_names = info["class_names"]

    edge_index = build_label_graph(args.dataset)

    model = MATCNet(
        config=config,
        num_classes=num_classes,
        edge_index=edge_index,
        class_names=class_names,
        ablation=args.ablation,
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    print(f"Model loaded from {args.model_path}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    results = predict(args.text, model, tokenizer, class_names, device,
                      args.max_seq_length)

    print(f"\nInput: \"{args.text}\"")
    print(f"\nPredictions:")
    for r in results:
        bar = "#" * int(r["confidence"] * 40)
        print(f"  {r['class']:30s} {r['confidence']:.4f}  {bar}")

    print(f"\nPredicted class: {results[0]['class']} ({results[0]['confidence']:.4f})")


if __name__ == "__main__":
    main()
