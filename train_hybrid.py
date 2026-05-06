"""Train the hybrid fusion head on cached features.

Inputs (cached):
    cache/<dataset>_train.pt  -> {h_matc, h_aux, labels}
    cache/<dataset>_test.pt   -> {h_matc, h_aux, labels}

Trains only the BatchNorm + Fusion MLP. All upstream models are frozen.
Outputs:
    results/hybrid_<dataset>.pt -> trained fusion head weights
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.hybrid_matc import HybridMATC, AUX_TOTAL_DIM, count_trainable_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--save_dir", type=str, default="results")
    p.add_argument("--limit_suffix", type=str, default="_5000",
                   help="suffix used by extract_features.py (e.g. _5000), or '' for full")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--ablation", type=str, default=None,
                   choices=[None, "no_aux", "no_spine"],
                   help="Ablation: 'no_aux' = spine only (vanilla MATC-Net features), "
                        "'no_spine' = aux features only (no MATC-Net)")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_cached(path):
    print(f"Loading {path}")
    d = torch.load(path, weights_only=True)
    return d["h_matc"], d["h_aux"], d["labels"]


def make_loader(h_matc, h_aux, labels, batch_size, shuffle):
    ds = TensorDataset(h_matc.float(), h_aux.float(), labels.long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, num_classes, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for h_m, h_a, y in loader:
            h_m, h_a = h_m.to(device), h_a.to(device)
            logits = model(h_m, h_a)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1,
            "preds": np.array(all_preds), "labels": np.array(all_labels)}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = os.path.join(
        args.cache_dir, f"{args.dataset}_train{args.limit_suffix}.pt")
    test_path = os.path.join(
        args.cache_dir, f"{args.dataset}_test{args.limit_suffix}.pt")

    h_matc_tr, h_aux_tr, y_tr = load_cached(train_path)
    h_matc_te, h_aux_te, y_te = load_cached(test_path)

    # Apply ablation by zeroing the disabled branch
    if args.ablation == "no_aux":
        print("ABLATION: zeroing auxiliary branch (spine-only baseline)")
        h_aux_tr = torch.zeros_like(h_aux_tr)
        h_aux_te = torch.zeros_like(h_aux_te)
    elif args.ablation == "no_spine":
        print("ABLATION: zeroing MATC-Net spine (aux-only baseline)")
        h_matc_tr = torch.zeros_like(h_matc_tr)
        h_matc_te = torch.zeros_like(h_matc_te)

    # Carve a validation split out of training data
    n = len(y_tr)
    val_size = int(n * args.val_split)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    train_loader = make_loader(h_matc_tr[tr_idx], h_aux_tr[tr_idx], y_tr[tr_idx],
                               args.batch_size, shuffle=True)
    val_loader = make_loader(h_matc_tr[val_idx], h_aux_tr[val_idx], y_tr[val_idx],
                             args.batch_size, shuffle=False)
    test_loader = make_loader(h_matc_te, h_aux_te, y_te,
                              args.batch_size, shuffle=False)

    print(f"\nTrain: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(y_te)}")
    print(f"Spine dim: {h_matc_tr.shape[1]}, Aux dim: {h_aux_tr.shape[1]}")

    model = HybridMATC(
        spine_dim=h_matc_tr.shape[1],
        aux_dim=h_aux_tr.shape[1],
        num_classes=args.num_classes,
    ).to(device)
    print(f"Trainable params: {count_trainable_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience_counter = 0
    suffix = f"_{args.ablation}" if args.ablation else ""
    save_path = os.path.join(args.save_dir, f"hybrid_{args.dataset}{suffix}.pt")

    print("\n=== Training fusion head ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for h_m, h_a, y in train_loader:
            h_m, h_a, y = h_m.to(device), h_a.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(h_m, h_a)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, args.num_classes)
        print(f"Epoch {epoch:02d} | loss={running_loss/n_batches:.4f} "
              f"| val_acc={val_metrics['accuracy']:.4f} "
              f"| val_f1={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test
    print(f"\nLoading best checkpoint: {save_path}")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, args.num_classes)
    print(f"\n=== TEST RESULTS ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                digits=4))


if __name__ == "__main__":
    main()
