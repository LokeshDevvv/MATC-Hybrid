"""Main training entry point for MATC-Net.

Usage:
    python train.py --dataset ag_news --epochs 30 --batch_size 32 --max_seq_length 128
    python train.py --dataset imdb --epochs 20 --batch_size 16 --max_seq_length 512
    python train.py --dataset 20newsgroups --epochs 30 --batch_size 32 --max_seq_length 256
    python train.py --dataset bbc --epochs 30 --batch_size 16 --max_seq_length 256

Ablation studies:
    python train.py --dataset ag_news --ablation no_mamba
    python train.py --dataset ag_news --ablation no_transformer
    python train.py --dataset ag_news --ablation no_supcon
    python train.py --dataset ag_news --ablation no_gat
    python train.py --dataset ag_news --ablation no_gating
"""

import argparse
import os
import sys

import torch

from utils.helpers import set_seed, load_config, get_device, count_parameters
from data.dataset import get_dataloaders
from data.label_graph import build_label_graph
from models.matc_net import MATCNet
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MATC-Net")
    parser.add_argument("--dataset", type=str, default="ag_news",
                        choices=["ag_news", "imdb", "20newsgroups", "bbc"],
                        help="Dataset to train on")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Override max_seq_length from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_mamba", "no_transformer", "no_supcon",
                                 "no_gat", "no_gating"],
                        help="Ablation study variant")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # Config
    config = load_config(args.config)

    # Override config with CLI args
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_seq_length is not None:
        config.model.max_seq_length = args.max_seq_length
    if args.lr is not None:
        config.training.learning_rate = args.lr

    device = get_device()

    # Build dataset name for saving (include ablation suffix)
    dataset_name = args.dataset
    if args.ablation:
        save_name = f"{dataset_name}_{args.ablation}"
    else:
        save_name = dataset_name

    # Data
    print(f"\nLoading dataset: {dataset_name}")
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        dataset_name,
        batch_size=config.training.batch_size,
        max_seq_length=config.model.max_seq_length,
        num_workers=args.num_workers,
    )

    # Label graph
    edge_index = build_label_graph(dataset_name)

    # Model
    print(f"\nBuilding MATC-Net (ablation={args.ablation})...")
    model = MATCNet(
        config=config,
        num_classes=num_classes,
        edge_index=edge_index,
        class_names=class_names,
        ablation=args.ablation,
    )
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
    )
    history = trainer.train(dataset_name=save_name)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)

    best_model_path = os.path.join(args.save_dir, f"best_model_{save_name}.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    evaluator = Evaluator(model, device, class_names, save_dir=args.save_dir)
    metrics, preds, labels, embeddings, gate_values, cm = evaluator.evaluate(
        test_loader, dataset_name=save_name
    )

    # Visualizations
    viz = Visualizer(save_dir=args.save_dir)
    viz.plot_training_curves(history, save_name)
    viz.plot_confusion_matrix(cm, class_names, save_name)
    viz.plot_tsne(embeddings, labels_array_from_list(labels), class_names, save_name)
    if gate_values is not None:
        viz.plot_gate_distribution(gate_values, save_name)

    print(f"\nAll results saved to {args.save_dir}/")
    print("Done!")


def labels_array_from_list(labels):
    """Convert list of labels to numpy array."""
    import numpy as np
    return np.array(labels)


if __name__ == "__main__":
    main()
