"""MATC-Net: Mamba-Augmented Transformer with Contrastive Schema-Aware Learning.

Main orchestrator module that wires together:
  - Shared RoBERTa embeddings
  - Dual-path encoders (Mamba + Transformer)
  - Gated Cross-Fusion
  - Schema-Aware GAT classifier
  - Supervised Contrastive projection head
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

from .mamba_encoder import MambaEncoder
from .transformer_encoder import TransformerBranchEncoder
from .gated_cross_fusion import GatedCrossFusion
from .schema_aware_classifier import SchemaAwareClassifier
from .contrastive_head import ContrastiveProjectionHead


class MATCNet(nn.Module):
    """MATC-Net: Dual-path hybrid model for text classification.

    Forward flow:
        input_ids -> Shared Embeddings -> Mamba Branch  ──┐
                                       -> Transformer Branch ──┤
                                                               ├──> Gated Cross-Fusion
                                                               │         |
                                                               │    H_fused
                                                               │    /     \\
                                                        Schema-Aware    Contrastive
                                                        Classifier      Head (SupCon)
                                                          logits          z
    """

    def __init__(self, config, num_classes, edge_index, class_names=None, ablation=None):
        """
        Args:
            config: configuration namespace
            num_classes: number of output classes
            edge_index: (2, E) label graph edges
            class_names: list of class name strings for label embedding init
            ablation: str or None — one of 'no_mamba', 'no_transformer',
                      'no_supcon', 'no_gat', 'no_gating'
        """
        super().__init__()
        self.config = config
        self.ablation = ablation
        d_model = config.model.embedding_dim

        # Register edge_index as buffer (moves with model to device)
        self.register_buffer("edge_index", edge_index)

        # Shared Embedding Layer (from RoBERTa)
        print("Loading RoBERTa embeddings...")
        roberta = RobertaModel.from_pretrained("roberta-base")
        self.embeddings = roberta.embeddings
        # Freeze embedding parameters initially (can unfreeze for fine-tuning)
        for param in self.embeddings.parameters():
            param.requires_grad = False
        del roberta

        self.embed_dropout = nn.Dropout(config.model.dropout)

        # Dual-Path Encoders
        self.use_mamba = ablation != "no_mamba"
        self.use_transformer = ablation != "no_transformer"

        if self.use_mamba:
            self.mamba_encoder = MambaEncoder(config.mamba)
        if self.use_transformer:
            self.transformer_encoder = TransformerBranchEncoder(config.transformer)

        # Gated Cross-Fusion
        if self.use_mamba and self.use_transformer and ablation != "no_gating":
            self.gated_fusion = GatedCrossFusion(d_model)
            self.fusion_mode = "gated"
        elif self.use_mamba and self.use_transformer:
            # no_gating ablation: simple concatenation + projection
            self.concat_proj = nn.Linear(2 * d_model, d_model)
            self.fusion_mode = "concat"
        else:
            self.fusion_mode = "single"

        # Schema-Aware Classifier (GAT) or flat Linear
        if ablation == "no_gat":
            self.classifier = nn.Linear(d_model, num_classes)
            self.use_gat = False
        else:
            self.schema_classifier = SchemaAwareClassifier(
                d_model=d_model,
                num_classes=num_classes,
                label_embed_dim=config.label_graph.label_embed_dim,
                gat_heads=config.label_graph.gat_heads,
                class_names=class_names,
            )
            self.use_gat = True

        # Contrastive Projection Head
        self.use_supcon = ablation != "no_supcon"
        if self.use_supcon:
            self.contrastive_head = ContrastiveProjectionHead(
                d_model=d_model,
                projection_dim=config.contrastive.projection_dim,
            )

        self._log_architecture()

    def _log_architecture(self):
        """Print model architecture summary."""
        parts = []
        if self.use_mamba:
            parts.append("Mamba")
        if self.use_transformer:
            parts.append("Transformer")
        fusion = self.fusion_mode
        classifier = "GAT" if self.use_gat else "Linear"
        supcon = "SupCon" if self.use_supcon else "None"
        print(f"MATC-Net Architecture: Encoders=[{'+'.join(parts)}] "
              f"Fusion={fusion} Classifier={classifier} Contrastive={supcon}")

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) — tokenized input
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            logits: (batch, num_classes)
            z: (batch, projection_dim) or None if no SupCon
            h_fused: (batch, d_model)
        """
        # Step 1: Shared Embeddings
        embeddings = self.embeddings(input_ids)  # (B, L, 768)
        embeddings = self.embed_dropout(embeddings)

        # Step 2: Dual-Path Encoding
        if self.use_mamba and self.use_transformer:
            h_mamba = self.mamba_encoder(embeddings, attention_mask)           # (B, 768)
            h_transformer = self.transformer_encoder(embeddings, attention_mask)  # (B, 768)
        elif self.use_mamba:
            h_mamba = self.mamba_encoder(embeddings, attention_mask)
            h_transformer = None
        else:
            h_mamba = None
            h_transformer = self.transformer_encoder(embeddings, attention_mask)

        # Step 3: Fusion
        if self.fusion_mode == "gated":
            h_fused = self.gated_fusion(h_mamba, h_transformer)
        elif self.fusion_mode == "concat":
            h_fused = self.concat_proj(torch.cat([h_mamba, h_transformer], dim=-1))
        else:
            h_fused = h_mamba if h_mamba is not None else h_transformer

        # Step 4: Classification
        if self.use_gat:
            logits, label_embeds = self.schema_classifier(h_fused, self.edge_index)
        else:
            logits = self.classifier(h_fused)

        # Step 5: Contrastive Projection
        z = self.contrastive_head(h_fused) if self.use_supcon else None

        return logits, z, h_fused

    def get_gate_values(self):
        """Return last gate values from GCF (for visualization)."""
        if self.fusion_mode == "gated":
            return self.gated_fusion.get_last_gate_values()
        return None
