"""Run per-branch ablation: drop each of the 7 aux branches in turn.

Each branch occupies a fixed slice in the 72-d aux vector:
  emotion   [0:28]
  sentiment [28:31]
  toxicity  [31:37]
  topic     [37:47]
  ner       [47:51]
  language  [51:71]
  sarcasm   [71:72]
"""

import os
import subprocess
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from models.hybrid_matc import HybridMATC


BRANCH_SLICES = {
    "emotion":   (0, 28),
    "sentiment": (28, 31),
    "toxicity":  (31, 37),
    "topic":     (37, 47),
    "ner":       (47, 51),
    "language":  (51, 71),
    "sarcasm":   (71, 72),
}

CACHE_PREFIX = "cache/v3_ag_news"
LIMIT = 5000
NUM_CLASSES = 4
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


def load(path):
    d = torch.load(path, weights_only=True)
    return d["h_matc"].float(), d["h_aux"].float(), d["labels"].long()


def make_loader(hm, ha, y, shuffle):
    return DataLoader(TensorDataset(hm, ha, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for hm, ha, y in loader:
            logits = model(hm.to(device), ha.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(y.numpy())
    return {"acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")}


def train_one(spine_dim, aux_dim, train_loader, val_loader, device):
    model = HybridMATC(spine_dim=spine_dim, aux_dim=aux_dim,
                       num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0
    best_state = None
    patience = 0
    for epoch in range(EPOCHS):
        model.train()
        for hm, ha, y in train_loader:
            hm, ha, y = hm.to(device), ha.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(hm, ha), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        m = evaluate(model, val_loader, device)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hm_tr, ha_tr, y_tr = load(f"{CACHE_PREFIX}_train_{LIMIT}.pt")
    hm_te, ha_te, y_te = load(f"{CACHE_PREFIX}_test_{LIMIT}.pt")

    n = len(y_tr)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[: int(0.15 * n)], perm[int(0.15 * n) :]

    spine_dim, aux_dim = hm_tr.shape[1], ha_tr.shape[1]
    print(f"Spine dim: {spine_dim}, full aux dim: {aux_dim}")

    results = {}

    # 1. Full hybrid (all 7 branches)
    print("\n=== Full hybrid (all 7 aux branches) ===")
    train_loader = make_loader(hm_tr[tr_idx], ha_tr[tr_idx], y_tr[tr_idx], True)
    val_loader = make_loader(hm_tr[val_idx], ha_tr[val_idx], y_tr[val_idx], False)
    test_loader = make_loader(hm_te, ha_te, y_te, False)
    model = train_one(spine_dim, aux_dim, train_loader, val_loader, device)
    m = evaluate(model, test_loader, device)
    results["full"] = m
    print(f"  Test: acc={m['acc']:.4f}  f1={m['f1']:.4f}")

    # 2. Spine only (zero out all aux)
    print("\n=== Spine only (DistilBERT, no aux) ===")
    ha_tr_zero = torch.zeros_like(ha_tr)
    ha_te_zero = torch.zeros_like(ha_te)
    tl = make_loader(hm_tr[tr_idx], ha_tr_zero[tr_idx], y_tr[tr_idx], True)
    vl = make_loader(hm_tr[val_idx], ha_tr_zero[val_idx], y_tr[val_idx], False)
    el = make_loader(hm_te, ha_te_zero, y_te, False)
    model = train_one(spine_dim, aux_dim, tl, vl, device)
    m = evaluate(model, el, device)
    results["spine_only"] = m
    print(f"  Test: acc={m['acc']:.4f}  f1={m['f1']:.4f}")

    # 3. Aux only (zero out spine)
    print("\n=== Aux only (no DistilBERT spine) ===")
    hm_tr_zero = torch.zeros_like(hm_tr)
    hm_te_zero = torch.zeros_like(hm_te)
    tl = make_loader(hm_tr_zero[tr_idx], ha_tr[tr_idx], y_tr[tr_idx], True)
    vl = make_loader(hm_tr_zero[val_idx], ha_tr[val_idx], y_tr[val_idx], False)
    el = make_loader(hm_te_zero, ha_te, y_te, False)
    model = train_one(spine_dim, aux_dim, tl, vl, device)
    m = evaluate(model, el, device)
    results["aux_only"] = m
    print(f"  Test: acc={m['acc']:.4f}  f1={m['f1']:.4f}")

    # 4. Drop one branch at a time
    for branch, (lo, hi) in BRANCH_SLICES.items():
        print(f"\n=== Drop {branch} ({lo}:{hi}) ===")
        mask = torch.ones(aux_dim, dtype=torch.float32)
        mask[lo:hi] = 0.0
        ha_tr_m = ha_tr * mask
        ha_te_m = ha_te * mask
        tl = make_loader(hm_tr[tr_idx], ha_tr_m[tr_idx], y_tr[tr_idx], True)
        vl = make_loader(hm_tr[val_idx], ha_tr_m[val_idx], y_tr[val_idx], False)
        el = make_loader(hm_te, ha_te_m, y_te, False)
        model = train_one(spine_dim, aux_dim, tl, vl, device)
        m = evaluate(model, el, device)
        results[f"drop_{branch}"] = m
        print(f"  Test: acc={m['acc']:.4f}  f1={m['f1']:.4f}")

    print("\n\n========== SUMMARY ==========")
    print(f"{'Configuration':<25s}  {'Acc':>8s}  {'F1':>8s}  {'ΔF1':>8s}")
    print("-" * 55)
    full_f1 = results["full"]["f1"]
    for name, m in results.items():
        delta = m["f1"] - full_f1
        print(f"{name:<25s}  {m['acc']:.4f}    {m['f1']:.4f}   {delta:+.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/ablation_results.json")


if __name__ == "__main__":
    main()
