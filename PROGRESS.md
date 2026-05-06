# MATC-Net Build Progress

## Environment

- **Platform:** Linux 6.6.87.2 (WSL2)
- **Python:** 3.10.12
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
- **PyTorch:** 2.2.2+cu121 (CUDA available)

---

## Phase 1: Project Structure — COMPLETE

Created the full project layout with 21 files:

```
MATC-Net/
├── config/
│   └── config.yaml                # All hyperparameters
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading & preprocessing
│   └── label_graph.py             # Label hierarchy graph construction
├── models/
│   ├── __init__.py
│   ├── matc_net.py                # Main MATC-Net model (orchestrator)
│   ├── mamba_encoder.py           # Mamba branch encoder
│   ├── transformer_encoder.py     # Transformer branch encoder
│   ├── gated_cross_fusion.py      # Gated Cross-Fusion module
│   ├── schema_aware_classifier.py # GAT-based label-aware classifier
│   └── contrastive_head.py        # SupCon projection head + loss
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # Training loop with both losses
│   └── evaluator.py               # Evaluation metrics + confusion matrix
├── utils/
│   ├── __init__.py
│   ├── helpers.py                 # Utility functions
│   └── visualization.py           # t-SNE, training curves, confusion matrix
├── train.py                       # Main entry point for training
├── evaluate.py                    # Standalone evaluation script
├── inference.py                   # Single text prediction
├── requirements.txt               # All dependencies
└── README.md                      # Project documentation
```

All 19 Python files pass `py_compile` syntax checks.

---

## Phase 2: Dependency Installation — COMPLETE

### Successfully Installed

| Package | Version | Status |
|---|---|---|
| torch | 2.2.2+cu121 | Installed |
| transformers | 4.57.6 | Installed |
| datasets | 4.5.0 | Installed |
| scikit-learn | 1.7.2 | Installed |
| torch-geometric | 2.7.0 | Installed |
| pytorch-metric-learning | 2.9.0 | Installed |
| pyyaml | Installed | Installed |
| matplotlib | Installed | Installed |
| seaborn | Installed | Installed |
| tqdm | Installed | Installed |
| numpy | Installed | Installed |
| pandas | Installed | Installed |

### Failed to Install

| Package | Reason |
|---|---|
| `mamba-ssm` | Requires CUDA toolkit (nvcc) at build time; `bare_metal_version` error during wheel build. **Fallback:** pure-PyTorch Mamba implementation is used instead. |

### Compatibility Fix Applied

- **Issue:** `torch-geometric` import failed due to `onnx` / `ml-dtypes` version mismatch (`float4_e2m1fn` not found in `ml_dtypes`).
- **Fix:** Upgraded `ml-dtypes` to 0.5.4 via `pip install --upgrade onnx ml-dtypes`. This resolved the import error (minor tensorflow incompatibility warning remains but does not affect functionality).

---

## Phase 3: Model Verification — COMPLETE

### Forward Pass Test

Built the model and ran a dummy forward pass successfully:

```
Loading RoBERTa embeddings...
MambaEncoder: mamba-ssm not available, using pure-PyTorch fallback (4 layers)
TransformerBranchEncoder: 4 layers, 12 heads, d_model=768
SchemaAwareClassifier: using torch-geometric GATConv
Initializing label embeddings from class names via RoBERTa...
  Initialized 4 label embeddings
MATC-Net Architecture: Encoders=[Mamba+Transformer] Fusion=gated Classifier=GAT Contrastive=SupCon

logits:  torch.Size([2, 4])
z:       torch.Size([2, 256])
h_fused: torch.Size([2, 768])

Total trainable parameters: 54,329,089
```

All tensor shapes are correct. Model builds and runs on GPU without errors.

---

## Phase 4: AG News Training — IN PROGRESS

### First Attempt

- **Command:** `python train.py --dataset ag_news --epochs 30 --batch_size 32 --max_seq_length 128`
- **Result:** Crashed immediately with `TypeError: '<=' not supported between instances of 'float' and 'str'`
- **Root Cause:** YAML `safe_load` parsed `2e-4` as a string, not a float.
- **Fix Applied:**
  1. Changed `config.yaml` value from `2e-4` to `0.0002`
  2. Added `_coerce_numeric()` helper in `utils/helpers.py` to auto-convert string numerics during config loading

### Second Attempt

- **Command:** Same as above
- **Result:** Training started successfully. Data loaded (Train: 102,000, Val: 18,000, Test: 7,600). Model initialized. Training loop began.
- **Issue:** Extremely slow — ~129 seconds per batch (3,187 batches per epoch). The pure-PyTorch Mamba fallback uses a sequential Python loop over `seq_len=128` tokens in `SelectiveSSM.forward()`, which does not parallelize on GPU.
- **Fix Applied:** Rewrote `models/mamba_encoder.py` with a **parallel associative scan** (Blelloch algorithm) to replace the sequential loop. For sequences ≤32 tokens, falls back to sequential scan. For longer sequences, uses `log2(L)` parallel sweep steps.

### Benchmarking & Optimization (Session 2)

**Problem 1: Blelloch scan was incorrect**
- Benchmarked old Blelloch parallel scan: max_diff of 0.47–0.58 vs sequential (completely wrong output)
- Root cause: Down-sweep used corrupted `aa` values from up-sweep; mishandled non-commutative associative operator

**Fix 1: Chunked parallel scan**
- Replaced Blelloch with correct chunked approach (sequential within 64-element chunks + carry propagation)
- Correctness verified: max_diff ~1e-7 (float32 precision)
- However, chunked scan was slightly slower than pure sequential for L≤512

**Problem 2: Autograd overhead**
- Even with correct scan, training was ~148s/batch because PyTorch builds a deep computation graph (128 sequential ops × 4 layers = 512 tracked operations)
- Backward pass traversal of this deep graph is extremely slow

**Fix 2: Custom `torch.autograd.Function`**
- Wrote `_ScanFn` with explicit forward/backward implementations
- Forward: sequential scan in O(L), no autograd graph built
- Backward: reverse-time scan for analytical gradient computation in O(L)
- Gradient correctness verified: max_diff 0.00e+00 vs autograd reference

**Benchmark Results (RTX 4050 Laptop, B=16, L=128, 4 layers):**

| Metric | Original | Blelloch | Custom Autograd |
|--------|----------|----------|-----------------|
| Speed (full model) | ~129s/batch | ~148s/batch | ~0.45s/batch |
| Correctness | Correct | Wrong (0.5 diff) | Correct (0.0 diff) |
| Speedup vs original | 1x | 0.87x | **~286x** |

### Third Attempt (Current — Running)

- **Command:** `python train.py --dataset ag_news --epochs 30 --batch_size 16 --max_seq_length 128`
- **Batch size:** Reduced from 32→16 to fit GPU memory (6GB RTX 4050)
- **Speed:** ~2.2 batches/sec (~0.45s/batch), ~48 min/epoch
- **Early learning:** CE loss dropping 1.52→0.35 in first 130 batches
- **Status:** Training in progress, estimated 12-16 hours total

---

## Bugs Found & Fixed

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `config/config.yaml` | `learning_rate: 2e-4` parsed as string by YAML | Changed to `0.0002`; added numeric coercion in config loader |
| 2 | `utils/helpers.py` | Config loader did not handle YAML string-to-float edge cases | Added `_coerce_numeric()` function in `_dict_to_namespace()` |
| 3 | `models/mamba_encoder.py` | Sequential scan loop (`for t in range(seq_len)`) made GPU training impractically slow (~129s/batch) | Replaced with parallel associative scan using Blelloch algorithm |
| 4 | `models/mamba_encoder.py` | Blelloch down-sweep produced wrong results (max_diff 0.5) | Replaced with correct chunked scan + raised sequential threshold to L≤512 |
| 5 | `models/mamba_encoder.py` | Autograd graph depth (128 ops × 4 layers) made backward pass ~148s/batch | Custom `torch.autograd.Function` with analytical gradients → 0.45s/batch |

---

## Next Steps

1. ~~Benchmark the optimized parallel scan Mamba encoder for speed~~ — DONE
2. **AG News training** — IN PROGRESS (~48min/epoch, ~12-16hrs total)
3. **Evaluate** on test set and generate visualizations (confusion matrix, t-SNE, training curves, gate distribution)
4. Train on remaining datasets: IMDB, 20 Newsgroups, BBC News
5. Run ablation studies
