"""Evaluation module for MATC-Net — metrics, confusion matrix, and analysis."""

import time
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    """Evaluate a trained MATC-Net model and produce all metrics."""

    def __init__(self, model, device, class_names=None, save_dir="results"):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, test_loader, dataset_name: str = "dataset"):
        """Run full evaluation on test set.

        Returns:
            metrics: dict with accuracy, precision, recall, f1, per-class metrics
            all_preds: list of predictions
            all_labels: list of true labels
            all_embeddings: np.ndarray of fused embeddings (N, d_model)
            all_gate_values: np.ndarray of gate values or None
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_embeddings = []
        all_gate_values = []

        print(f"\nEvaluating on {dataset_name} test set...")
        latencies = []

        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            start_time = time.time()
            logits, z, h_fused = self.model(input_ids, attention_mask)
            elapsed = time.time() - start_time
            latencies.append(elapsed / input_ids.size(0))

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_embeddings.append(h_fused.cpu().numpy())

            gate_vals = self.model.get_gate_values()
            if gate_vals is not None:
                all_gate_values.append(gate_vals.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if all_gate_values:
            all_gate_values = np.concatenate(all_gate_values, axis=0)
        else:
            all_gate_values = None

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        avg_latency_ms = np.mean(latencies) * 1000

        # Classification report
        target_names = self.class_names if self.class_names else None
        report = classification_report(
            all_labels, all_preds, target_names=target_names, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "avg_latency_ms": avg_latency_ms,
            "per_class": {},
        }

        class_names = self.class_names or [str(i) for i in range(len(precision))]
        for i, name in enumerate(class_names):
            metrics["per_class"][name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Print results
        print(f"\n{'='*60}")
        print(f"Results for {dataset_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {macro_precision:.4f} (macro)")
        print(f"Recall:    {macro_recall:.4f} (macro)")
        print(f"F1-Score:  {macro_f1:.4f} (macro)")
        print(f"Latency:   {avg_latency_ms:.2f} ms/sample")
        print(f"\n{report}")

        # Save metrics
        metrics_path = os.path.join(self.save_dir, f"{dataset_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        return metrics, all_preds, all_labels, all_embeddings, all_gate_values, cm
