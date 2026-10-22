from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tasks.diabetic_retinopathy.models.cerl.local_branch import LesionQueryLocalBranch
from tasks.diabetic_retinopathy.models.retinal_carl import _build_feature_pyramid


class RCELClinicalEvidenceModel(nn.Module):
    """Minimal latent clinical evidence model for RCEL-v0 controlled ablations."""

    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        backbone_img_size: int | None = None,
        evidence_query_count: int = 8,
        evidence_dim: int = 256,
        attention_heads: int = 4,
        ordinal_thresholds: int = 4,
    ) -> None:
        super().__init__()
        self.feature_pyramid, _stage3_dim, stage4_dim = _build_feature_pyramid(
            backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
        )
        self.evidence_encoder = LesionQueryLocalBranch(
            stage3_dim=stage4_dim,
            local_dim=evidence_dim,
            query_count=evidence_query_count,
            attention_heads=attention_heads,
        )
        self.evidence_norm = nn.LayerNorm(evidence_dim)
        self.binary_head = nn.Linear(evidence_dim, 1)
        self.ordinal_head = nn.Linear(evidence_dim, ordinal_thresholds)
        self.progression_axis = nn.Linear(evidence_dim, 1, bias=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _stage3, stage4 = self.feature_pyramid(image)
        evidence = self.evidence_encoder(stage4)
        embedding = self.evidence_norm(evidence["local_embedding_raw"])
        binary_logits = self.binary_head(embedding).squeeze(1)
        ordinal_logits = self.ordinal_head(embedding)
        severity_score = self.progression_axis(embedding).squeeze(1)
        tokens = evidence["local_query_tokens"]
        token_norm = tokens.norm(dim=-1)

        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": embedding,
            "severity_score": severity_score,
            "clinical_evidence_tokens": tokens,
            "evidence_prob_pos": torch.sigmoid(binary_logits),
            "evidence_token_norm_mean": token_norm.mean(dim=1),
            "evidence_token_norm_std": token_norm.std(dim=1, unbiased=False),
            "evidence_token_variance": tokens.var(dim=(1, 2), unbiased=False),
            **evidence,
        }


def build_rcel_evidence_model(config: dict[str, Any]) -> nn.Module:
    return RCELClinicalEvidenceModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        evidence_query_count=int(config.get("evidence_query_count", 8)),
        evidence_dim=int(config.get("evidence_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
    )
