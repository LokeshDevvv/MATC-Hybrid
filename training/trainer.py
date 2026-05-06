"""Training loop for MATC-Net with combined CE + SupCon loss."""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.contrastive_head import SupConLoss


class Trainer:
    """Handles the complete training pipeline for MATC-Net."""

    def __init__(self, model, config, train_loader, val_loader, device,
                 save_dir="results"):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=config.contrastive.temperature)
        self.alpha = config.training.alpha  # CE weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        total_steps = len(train_loader) * config.training.max_epochs
        warmup_steps = config.training.warmup_steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.scaler = GradScaler() if device.type == "cuda" else None
        self.use_amp = device.type == "cuda"

        # Training history
        self.history = {
            "train_loss_total": [],
            "train_loss_ce": [],
            "train_loss_supcon": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rates": [],
        }

    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_supcon = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits, z, h_fused = self.model(input_ids, attention_mask)
                    loss_ce = self.ce_loss(logits, labels)

                    if z is not None:
                        loss_supcon = self.supcon_loss(z, labels)
                    else:
                        loss_supcon = torch.tensor(0.0, device=self.device)

                    loss_total = self.alpha * loss_ce + (1 - self.alpha) * loss_supcon

                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, z, h_fused = self.model(input_ids, attention_mask)
                loss_ce = self.ce_loss(logits, labels)

                if z is not None:
                    loss_supcon = self.supcon_loss(z, labels)
                else:
                    loss_supcon = torch.tensor(0.0, device=self.device)

                loss_total = self.alpha * loss_ce + (1 - self.alpha) * loss_supcon

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss_total.item()
            total_ce += loss_ce.item()
            total_supcon += loss_supcon.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "ce": f"{loss_ce.item():.4f}",
                "sc": f"{loss_supcon.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_supcon = total_supcon / num_batches

        return avg_loss, avg_ce, avg_supcon

    @torch.no_grad()
    def validate(self):
        """Run validation and return accuracy and F1."""
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _, _ = self.model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        return accuracy, f1

    def train(self, dataset_name: str = "dataset"):
        """Full training loop with early stopping.

        Returns:
            history: dict of training metrics per epoch
        """
        max_epochs = self.config.training.max_epochs
        patience = self.config.training.early_stopping_patience
        best_accuracy = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, f"best_model_{dataset_name}.pt")

        print(f"\nStarting training for {max_epochs} epochs...")
        print(f"  Alpha (CE weight): {self.alpha}, SupCon weight: {1 - self.alpha}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Device: {self.device}\n")

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()

            # Train
            avg_loss, avg_ce, avg_supcon = self.train_epoch(epoch)

            # Validate
            val_accuracy, val_f1 = self.validate()

            elapsed = time.time() - start_time

            # Record history
            self.history["train_loss_total"].append(avg_loss)
            self.history["train_loss_ce"].append(avg_ce)
            self.history["train_loss_supcon"].append(avg_supcon)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_f1"].append(val_f1)
            self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])

            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, SC: {avg_supcon:.4f}) | "
                f"Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  -> New best model saved (accuracy: {best_accuracy:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs.")
                    break

        print(f"\nTraining complete. Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best model saved to: {best_model_path}")

        # Save training history
        history_path = os.path.join(self.save_dir, f"{dataset_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
