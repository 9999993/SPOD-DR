from __future__ import annotations

import torch
from torch import nn


class LesionQueryLocalBranch(nn.Module):
    def __init__(self, *, stage3_dim: int, local_dim: int, query_count: int, attention_heads: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(stage3_dim, local_dim, kernel_size=1, bias=False)
        self.queries = nn.Parameter(torch.randn(query_count, local_dim) * 0.02)
        self.attention = nn.MultiheadAttention(local_dim, attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(local_dim)
        self.pool_score = nn.Linear(local_dim, 1)

    def forward(self, stage3: torch.Tensor) -> dict[str, torch.Tensor]:
        local_feature = self.project(stage3)
        batch_size, _, height, width = local_feature.shape
        token_grid = local_feature.flatten(2).transpose(1, 2)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_tokens, attn = self.attention(
            queries,
            token_grid,
            token_grid,
            need_weights=True,
            average_attn_weights=False,
        )
        gate = torch.softmax(self.pool_score(query_tokens).squeeze(-1), dim=1)
        pooled = torch.sum(query_tokens * gate.unsqueeze(-1), dim=1)
        attention_maps = attn.mean(dim=1).reshape(batch_size, queries.shape[1], height, width)
        flat_attention = attention_maps.flatten(2).clamp_min(1e-8)
        attention_entropy = -(flat_attention * flat_attention.log()).sum(dim=2).mean(dim=1)
        gate_entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=1)
        return {
            "local_embedding_raw": self.norm(pooled),
            "local_query_tokens": query_tokens,
            "query_attention_maps": attention_maps,
            "query_gate": gate,
            "query_attention_entropy": attention_entropy,
            "query_gate_entropy": gate_entropy,
            "query_output_variance": query_tokens.var(dim=(1, 2), unbiased=False),
        }
