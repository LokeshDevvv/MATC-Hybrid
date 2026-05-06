"""Standalone evaluation script for MATC-Net.

Usage:
    python evaluate.py --dataset ag_news --model_path results/best_model_ag_news.pt
    python evaluate.py --dataset ag_news --model_path results/best_model_ag_news.pt --visualize all
"""

import argparse
import os

import numpy as np
import torch

from utils.helpers import set_seed, load_config, get_device
from data.dataset import get_dataloaders
from data.label_graph import build_label_graph
from models.matc_net import MATCNet
from training.evaluator import Evaluator
from utils.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MATC-Net")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ag_news", "imdb", "20newsgroups", "bbc"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model weights (.pt)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_mamba", "no_transformer", "no_supcon",
                                 "no_gat", "no_gating"])
    parser.add_argument("--visualize", type=str, default="all",
                        choices=["all", "confusion", "tsne", "gate", "none"],
                        help="Which visualizations to generate")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    if args.max_seq_length is not None:
        config.model.max_seq_length = args.max_seq_length

    device = get_device()

    dataset_name = args.dataset
    save_name = f"{dataset_name}_{args.ablation}" if args.ablation else dataset_name

    # Data (only need test_loader)
    _, _, test_loader, num_classes, class_names = get_dataloaders(
        dataset_name,
        batch_size=args.batch_size,
        max_seq_length=config.model.max_seq_length,
    )

    # Label graph
    edge_index = build_label_graph(dataset_name)

    # Model
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
    print(f"Loaded model from {args.model_path}")

    # Evaluate
    evaluator = Evaluator(model, device, class_names, save_dir=args.save_dir)
    metrics, preds, labels, embeddings, gate_values, cm = evaluator.evaluate(
        test_loader, dataset_name=save_name
    )

    # Visualizations
    if args.visualize != "none":
        viz = Visualizer(save_dir=args.save_dir)
        labels_arr = np.array(labels)

        if args.visualize in ("all", "confusion"):
            viz.plot_confusion_matrix(cm, class_names, save_name)

        if args.visualize in ("all", "tsne"):
            viz.plot_tsne(embeddings, labels_arr, class_names, save_name)

        if args.visualize in ("all", "gate") and gate_values is not None:
            viz.plot_gate_distribution(gate_values, save_name)

    print(f"\nEvaluation complete. Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
