"""Schema-Aware Label Classifier using Graph Attention Networks for MATC-Net.

Encodes label relationships through a GAT, then uses enriched label embeddings
as classifier weights (instead of a standard nn.Linear layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

try:
    from torch_geometric.nn import GATConv
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False


class SimpleGATLayer(nn.Module):
    """Fallback single-head GAT layer when torch-geometric is not available."""

    def __init__(self, in_features, out_features, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.randn(num_heads, out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)

    def forward(self, x, edge_index):
        """
        Args:
            x: (num_nodes, in_features)
            edge_index: (2, num_edges)
        Returns:
            out: (num_nodes, num_heads * head_dim)
        """
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)  # (N, heads, dim)

        src, dst = edge_index  # (E,), (E,)

        # Attention coefficients
        e_src = (h[src] * self.a_src.unsqueeze(0)).sum(dim=-1)  # (E, heads)
        e_dst = (h[dst] * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (E, heads)
        e = self.leaky_relu(e_src + e_dst)  # (E, heads)

        # Softmax over neighbors (sparse)
        e_max = torch.zeros(N, self.num_heads, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True)
        e = torch.exp(e - e_max[dst])

        # Sum of exp for normalization
        e_sum = torch.zeros(N, self.num_heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-8)  # (E, heads)

        # Weighted aggregation
        msg = alpha.unsqueeze(-1) * h[src]  # (E, heads, dim)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        return out.reshape(N, self.num_heads * self.head_dim)


class SchemaAwareClassifier(nn.Module):
    """GAT-based classifier that uses label graph structure.

    Instead of a flat nn.Linear, this module:
    1. Enriches label embeddings via a 2-layer GAT over the label hierarchy
    2. Classifies by computing dot product between projected features and label embeddings
    """

    def __init__(self, d_model: int, num_classes: int, label_embed_dim: int = 256,
                 gat_heads: int = 4, class_names: list = None):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim

        # Project fused features to label embedding space
        self.feature_proj = nn.Linear(d_model, label_embed_dim)

        # Label embeddings (initialized later with RoBERTa if class_names provided)
        self.label_embeddings = nn.Parameter(
            torch.randn(num_classes, label_embed_dim) * 0.02
        )

        # GAT layers
        gat_out_dim = label_embed_dim // gat_heads  # e.g. 256 // 4 = 64
        if GAT_AVAILABLE:
            print("SchemaAwareClassifier: using torch-geometric GATConv")
            self.gat1 = GATConv(label_embed_dim, gat_out_dim, heads=gat_heads)
            self.gat2 = GATConv(gat_out_dim * gat_heads, label_embed_dim, heads=1, concat=False)
        else:
            print("SchemaAwareClassifier: torch-geometric not available, using fallback GAT")
            self.gat1 = SimpleGATLayer(label_embed_dim, gat_out_dim, num_heads=gat_heads)
            self.gat2 = SimpleGATLayer(gat_out_dim * gat_heads, label_embed_dim, num_heads=1)

        self.gat_norm = nn.LayerNorm(label_embed_dim)

        # Temperature for logit scaling
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Initialize label embeddings from class names if provided
        if class_names is not None:
            self._initialize_label_embeddings(class_names)

    @torch.no_grad()
    def _initialize_label_embeddings(self, class_names):
        """Initialize label embeddings by encoding class names through RoBERTa."""
        print("Initializing label embeddings from class names via RoBERTa...")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta = RobertaModel.from_pretrained("roberta-base")
        roberta.eval()

        proj = nn.Linear(768, self.label_embed_dim)  # Temporary projection

        embeddings = []
        for name in class_names:
            # Clean up class name for better encoding
            clean_name = name.replace(".", " ").replace("_", " ")
            tokens = tokenizer(clean_name, return_tensors="pt", padding=True, truncation=True)
            output = roberta(**tokens).last_hidden_state.mean(dim=1)  # (1, 768)
            emb = proj(output).squeeze(0)  # (label_embed_dim,)
            embeddings.append(emb)

        init_embeds = torch.stack(embeddings, dim=0)  # (num_classes, label_embed_dim)
        self.label_embeddings.data.copy_(init_embeds)
        print(f"  Initialized {len(class_names)} label embeddings")

        # Free RoBERTa
        del roberta, tokenizer, proj

    def forward(self, h_fused, edge_index):
        """
        Args:
            h_fused: (batch, d_model) — fused representation
            edge_index: (2, num_edges) — label graph edges
        Returns:
            logits: (batch, num_classes)
            label_enriched: (num_classes, label_embed_dim) — for analysis
        """
        # Enrich label embeddings through GAT
        label_x = self.label_embeddings  # (num_classes, label_embed_dim)
        label_x = F.elu(self.gat1(label_x, edge_index))  # (num_classes, gat_heads * gat_out_dim)
        label_enriched = self.gat2(label_x, edge_index)   # (num_classes, label_embed_dim)
        label_enriched = self.gat_norm(label_enriched + self.label_embeddings)  # residual + norm

        # Project fused features to label space
        h_proj = self.feature_proj(h_fused)  # (batch, label_embed_dim)

        # Compute logits as scaled dot product
        logits = self.logit_scale * torch.matmul(h_proj, label_enriched.T)  # (batch, num_classes)

        return logits, label_enriched
