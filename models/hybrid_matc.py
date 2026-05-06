"""HybridMATC: MATC-Net spine + pre-trained auxiliary branches.

Architecture:
    text
     |
     +--> MATC-Net (Mamba+Transformer+GAT+SupCon) -> h_matc (768-d)
     +--> Pre-trained Emotion model (frozen)      -> p_emo  (28-d)
     +--> Pre-trained Sentiment model (frozen)    -> p_sent (3-d)
     +--> Pre-trained Toxicity model (frozen)     -> p_tox  (6-d)
     +--> Pre-trained Topic zero-shot (frozen)    -> p_top  (10-d)
                                                       |
                                  concat (815-d) ------+
                                       |
                                  BatchNorm
                                       |
                          Fusion MLP (815 -> 512 -> 256 -> C)
                                       |
                                    logits
"""

import torch
import torch.nn as nn


# Auxiliary branch dimensions (must match feature extractor output)
EMOTION_DIM = 28
SENTIMENT_DIM = 3
TOXICITY_DIM = 6
TOPIC_DIM = 10
AUX_TOTAL_DIM = EMOTION_DIM + SENTIMENT_DIM + TOXICITY_DIM + TOPIC_DIM  # 47


class HybridMATC(nn.Module):
    """Hybrid model combining MATC-Net contextual embedding with pre-trained
    auxiliary signal branches via batch-normalised concatenation + fusion MLP.

    Inputs to forward() are pre-computed features (the upstream models are
    frozen during training, so features are extracted once and cached).
    """

    def __init__(
        self,
        spine_dim: int = 768,
        aux_dim: int = AUX_TOTAL_DIM,
        num_classes: int = 4,
        hidden_dims=(512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spine_dim = spine_dim
        self.aux_dim = aux_dim
        self.fused_dim = spine_dim + aux_dim

        self.fusion_bn = nn.BatchNorm1d(self.fused_dim)

        layers = []
        in_dim = self.fused_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, h_matc: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_matc: (B, 768) MATC-Net spine embedding
            h_aux:  (B, 47)  concatenated auxiliary probabilities
        Returns:
            logits: (B, num_classes)
        """
        h_fused = torch.cat([h_matc, h_aux], dim=-1)
        h_fused = self.fusion_bn(h_fused)
        return self.fusion_mlp(h_fused)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HybridMATC(num_classes=4)
    h_matc = torch.randn(8, 768)
    h_aux = torch.randn(8, AUX_TOTAL_DIM)
    logits = model(h_matc, h_aux)
    print(f"Input spine: {h_matc.shape}")
    print(f"Input aux:   {h_aux.shape}")
    print(f"Output:      {logits.shape}")
    print(f"Trainable params: {count_trainable_parameters(model):,}")
