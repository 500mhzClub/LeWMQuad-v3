"""BeliefEncoder — history-conditioned place embedding over frozen LeWM latents.

Stage 2 of the topological-nav build (``docs/v3_topological_nav_plan.md`` §5;
decision in ``docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md``). Stage 1
showed naive mean/concat pooling of a frozen-latent history barely lifts
same-place recall (+0.014), yet a history *window* separates single-frame-aliased
different-cell pairs with AUC ~0.86 — the disambiguating information is present
but not trivially poolable. The BeliefEncoder is the *learned* temporal
aggregator meant to extract it: a small Transformer over the window + attention
pooling -> an L2-normalized place embedding, trained with **supervised
contrastive** loss (same masking as ``PlaceRetrievalHead``: same-cell positives,
BFS-distance >= 2 negatives).

No separate anti-collapse regularizer is used: the supervised-contrastive
*negatives* already make collapse maximally costly (a constant embedding cannot
separate positives from negatives), and the repo's existing ``PlaceRetrievalHead``
trains the same way with no regularizer. (The repo's JEPA anti-collapse mechanism
is **SIGReg** — ``lewm/models/sigreg.py`` — which exists for the *negative-free*
world-model objective and does not apply to a contrastive head with negatives.)

Input is a window of *frozen* LeWM latents ``(B, H, latent_dim)`` ordered
oldest->newest; the LeWM encoder stays frozen (recognition-not-metric base). The
body-motion auxiliary named in the spec is deferred to a later iteration; v1 is
visual-history only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = 128,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.max_len = int(max_len)

        self.input_proj = nn.Linear(latent_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attn_pool = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, z_history: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``z_history``: (B, H, latent_dim) oldest->newest. Returns (B, embedding_dim), L2-normalized.

        ``key_padding_mask``: optional (B, H) bool, True = padded/ignored step.
        """
        if z_history.dim() != 3:
            raise ValueError(f"expected (B, H, latent_dim), got {tuple(z_history.shape)}")
        b, h, _ = z_history.shape
        if h > self.max_len:
            raise ValueError(f"history length {h} exceeds max_len {self.max_len}")
        x = self.input_proj(z_history) + self.pos[:, :h]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        query = self.query.expand(b, -1, -1)
        pooled, _attn = self.attn_pool(query, x, x, key_padding_mask=key_padding_mask)
        emb = self.proj(self.norm(pooled.squeeze(1)))
        return F.normalize(emb, dim=-1)
