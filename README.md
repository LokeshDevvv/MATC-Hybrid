# MATC-Net: Mamba-Augmented Transformer with Contrastive Schema-Aware Learning

A novel hybrid deep learning architecture for Natural Language Text Classification that combines:

- **Mamba (Selective State Space Model)** — O(n) linear-time global context modeling
- **Transformer Encoder (Multi-Head Self-Attention)** — fine-grained local semantic relationships
- **Gated Cross-Fusion (GCF)** — learned gating to dynamically fuse both branches
- **Schema-Aware Label Embeddings via GAT** — Graph Attention Network on label hierarchy
- **Supervised Contrastive Learning (SupCon)** — discriminative class boundary learning

## Architecture

```
Input Text
    │
    ▼
[RoBERTa Embeddings] (shared)
    │
    ├──────────────────┐
    ▼                  ▼
[Mamba Encoder]   [Transformer Encoder]
    │                  │
    ▼                  ▼
[Mean Pooling]    [CLS Pooling]
    │                  │
    └──────┬───────────┘
           ▼
    [Gated Cross-Fusion]
           │
           ▼
        H_fused
        /     \
       ▼       ▼
[Schema-Aware   [Contrastive
 GAT Classifier] Projection Head]
       │              │
    logits         SupCon Loss
       │
    CE Loss
       │
    Total Loss = α·CE + (1-α)·SupCon
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU support with Mamba:
```bash
pip install mamba-ssm
```

## Datasets

| Dataset | Classes | Train | Test | Key Test |
|---------|---------|-------|------|----------|
| AG News | 4 | 120,000 | 7,600 | Primary benchmark |
| IMDB | 2 | 25,000 | 25,000 | Long document capability |
| 20 Newsgroups | 20 | 11,314 | 7,532 | Hierarchical categories (GAT) |
| BBC News | 5 | ~1,558 | ~667 | Similar category separation (SupCon) |

## Training

```bash
# AG News (primary benchmark)
python train.py --dataset ag_news --epochs 30 --batch_size 32 --max_seq_length 128

# IMDB (long documents)
python train.py --dataset imdb --epochs 20 --batch_size 16 --max_seq_length 512

# 20 Newsgroups (hierarchical)
python train.py --dataset 20newsgroups --epochs 30 --batch_size 32 --max_seq_length 256

# BBC News
python train.py --dataset bbc --epochs 30 --batch_size 16 --max_seq_length 256
```

## Ablation Studies

```bash
python train.py --dataset ag_news --ablation no_mamba
python train.py --dataset ag_news --ablation no_transformer
python train.py --dataset ag_news --ablation no_supcon
python train.py --dataset ag_news --ablation no_gat
python train.py --dataset ag_news --ablation no_gating
```

## Evaluation

```bash
python evaluate.py --dataset ag_news --model_path results/best_model_ag_news.pt --visualize all
```

## Inference

```bash
python inference.py --dataset ag_news \
    --model_path results/best_model_ag_news.pt \
    --text "Apple announced a new iPhone today"
```

## Project Structure

```
MATC-Net/
├── config/config.yaml              # Hyperparameters
├── data/
│   ├── dataset.py                  # Dataset loading & preprocessing
│   └── label_graph.py              # Label hierarchy graph construction
├── models/
│   ├── matc_net.py                 # Main MATC-Net orchestrator
│   ├── mamba_encoder.py            # Mamba branch (with pure-PyTorch fallback)
│   ├── transformer_encoder.py      # Transformer branch
│   ├── gated_cross_fusion.py       # Gated Cross-Fusion module
│   ├── schema_aware_classifier.py  # GAT-based label-aware classifier
│   └── contrastive_head.py         # SupCon projection head + loss
├── training/
│   ├── trainer.py                  # Training loop (CE + SupCon)
│   └── evaluator.py               # Evaluation metrics
├── utils/
│   ├── helpers.py                  # Utilities (seed, config, device)
│   └── visualization.py           # t-SNE, confusion matrix, training curves
├── train.py                        # Main training entry point
├── evaluate.py                     # Standalone evaluation
├── inference.py                    # Single text prediction
└── requirements.txt
```

## Configuration

All hyperparameters are in `config/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.alpha` | 0.6 | CE loss weight (SupCon weight = 1-alpha) |
| `contrastive.temperature` | 0.07 | SupCon temperature τ |
| `mamba.num_layers` | 4 | Number of Mamba blocks |
| `transformer.num_layers` | 4 | Number of Transformer layers |
| `training.learning_rate` | 2e-4 | AdamW learning rate |
| `training.early_stopping_patience` | 5 | Epochs before early stopping |
