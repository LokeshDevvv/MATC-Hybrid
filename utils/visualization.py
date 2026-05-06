"""Visualization utilities for MATC-Net."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


class Visualizer:
    """Generate all MATC-Net visualizations."""

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, history: dict, dataset_name: str):
        """Plot training loss and validation accuracy curves.

        Args:
            history: dict with keys 'train_loss_ce', 'train_loss_supcon',
                     'train_loss_total', 'val_accuracy', 'val_f1'
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history["train_loss_total"]) + 1)

        # Loss curves
        ax = axes[0]
        ax.plot(epochs, history["train_loss_total"], label="Total Loss", linewidth=2)
        ax.plot(epochs, history["train_loss_ce"], label="CE Loss", linestyle="--")
        ax.plot(epochs, history["train_loss_supcon"], label="SupCon Loss", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Losses — {dataset_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curve
        ax = axes[1]
        ax.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2, color="green")
        if "val_f1" in history:
            ax.plot(epochs, history["val_f1"], label="Val F1 (Macro)", linewidth=2, color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"Validation Metrics — {dataset_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{dataset_name}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to {path}")

    def plot_confusion_matrix(self, cm, class_names, dataset_name: str):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {dataset_name}")
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{dataset_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix to {path}")

    def plot_tsne(self, embeddings: np.ndarray, labels: np.ndarray,
                  class_names: list, dataset_name: str):
        """Plot t-SNE visualization of fused embeddings.

        Args:
            embeddings: (N, d_model) array of fused representations
            labels: (N,) array of integer labels
            class_names: list of class name strings
        """
        print("Computing t-SNE embeddings...")
        n_samples = min(len(embeddings), 5000)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sub_embeddings = embeddings[indices]
        sub_labels = labels[indices]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        coords = tsne.fit_transform(sub_embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=sub_labels, cmap="tab10" if len(class_names) <= 10 else "tab20",
            s=8, alpha=0.6
        )

        handles = []
        for i, name in enumerate(class_names):
            mask = sub_labels == i
            if mask.any():
                handles.append(
                    ax.scatter([], [], c=[plt.cm.tab10(i / 10) if len(class_names) <= 10
                                          else plt.cm.tab20(i / 20)], label=name, s=30)
                )
        ax.legend(handles=handles, loc="best", fontsize=7, markerscale=2)
        ax.set_title(f"t-SNE of Fused Embeddings — {dataset_name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{dataset_name}_tsne.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved t-SNE plot to {path}")

    def plot_gate_distribution(self, gate_values: np.ndarray, dataset_name: str):
        """Plot histogram of GCF gate values.

        Args:
            gate_values: (N, d_model) or (N,) array of gate values between 0 and 1
        """
        if gate_values.ndim > 1:
            gate_values = gate_values.mean(axis=-1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(gate_values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(x=0.5, color="red", linestyle="--", label="Equal fusion (0.5)")
        ax.axvline(x=gate_values.mean(), color="green", linestyle="--",
                   label=f"Mean = {gate_values.mean():.3f}")
        ax.set_xlabel("Gate Value (1 = Mamba, 0 = Transformer)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"GCF Gate Value Distribution — {dataset_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{dataset_name}_gate_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved gate distribution to {path}")

    def plot_ablation_comparison(self, ablation_results: dict, dataset_name: str):
        """Plot bar chart comparing ablation study results.

        Args:
            ablation_results: dict mapping variant name to accuracy, e.g.
                {'Full MATC-Net': 0.94, 'No Mamba': 0.91, ...}
        """
        variants = list(ablation_results.keys())
        accuracies = list(ablation_results.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if v == "Full MATC-Net" else "#3498db" for v in variants]
        bars = ax.bar(variants, accuracies, color=colors, edgecolor="black", alpha=0.85)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Accuracy")
        ax.set_title(f"Ablation Study — {dataset_name}")
        ax.set_ylim(min(accuracies) - 0.05, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{dataset_name}_ablation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved ablation chart to {path}")

    def plot_dataset_comparison(self, results: dict):
        """Plot accuracy comparison across datasets.

        Args:
            results: dict mapping dataset name to accuracy
        """
        datasets = list(results.keys())
        accuracies = list(results.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(datasets, accuracies, color="#3498db", edgecolor="black", alpha=0.85)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Accuracy")
        ax.set_title("MATC-Net Accuracy Across Datasets")
        ax.set_ylim(min(accuracies) - 0.05, 1.0)
        plt.tight_layout()
        path = os.path.join(self.save_dir, "comparison_table.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison chart to {path}")

    def save_metrics(self, metrics: dict, dataset_name: str):
        """Save metrics to JSON file."""
        path = os.path.join(self.save_dir, f"{dataset_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {path}")
