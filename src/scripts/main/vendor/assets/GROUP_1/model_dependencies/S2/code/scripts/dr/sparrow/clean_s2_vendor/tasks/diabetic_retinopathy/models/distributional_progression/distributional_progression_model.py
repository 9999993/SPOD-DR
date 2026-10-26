from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clean_s2_vendor.tasks.diabetic_retinopathy.models.cerl.global_branch import GlobalRetinaBranch
from clean_s2_vendor.tasks.diabetic_retinopathy.models.cerl.local_branch import LesionQueryLocalBranch
from clean_s2_vendor.tasks.diabetic_retinopathy.models.distributional_progression.disagreement_fusion import DisagreementFusion
from clean_s2_vendor.tasks.diabetic_retinopathy.models.distributional_progression.ordinal_gaussian_decoder import (
    compute_ordinal_gaussian_probs,
    grade_probs_to_ordinal_logits,
)
from clean_s2_vendor.tasks.diabetic_retinopathy.models.distributional_progression.severity_distribution_head import (
    SeverityDistributionHead,
)
from clean_s2_vendor.tasks.diabetic_retinopathy.models.retinal_carl import _build_feature_pyramid


class DisagreementAwareProgressionModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        backbone_img_size: int | None = None,
        pooling: str = "gem",
        local_query_count: int = 6,
        feature_dim: int = 256,
        attention_heads: int = 4,
        mode: str = "local_global",
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.feature_pyramid, stage3_dim, stage4_dim = _build_feature_pyramid(
            backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
        )
        self.local_branch = LesionQueryLocalBranch(
            stage3_dim=stage3_dim,
            local_dim=feature_dim,
            query_count=local_query_count,
            attention_heads=attention_heads,
        )
        self.local_project = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.global_branch = GlobalRetinaBranch(stage4_dim=stage4_dim, fusion_dim=feature_dim, pooling=pooling)
        self.local_head = SeverityDistributionHead(in_dim=feature_dim)
        self.global_head = SeverityDistributionHead(in_dim=feature_dim)
        self.fusion = DisagreementFusion(beta=beta)
        self.embedding_project = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def _probs(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            return compute_ordinal_gaussian_probs(mu.float(), sigma.float())

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stage3, stage4 = self.feature_pyramid(image)
        local_features = self.local_branch(stage3)
        z_local = self.local_project(local_features["local_embedding_raw"])
        z_global = self.global_branch(stage4)
        local_mu, local_sigma = self.local_head(z_local)
        global_mu, global_sigma = self.global_head(z_global)

        if self.mode == "global_only":
            fused_mu, fused_sigma = global_mu, global_sigma
            local_weight = torch.zeros_like(global_mu)
            global_weight = torch.ones_like(global_mu)
            within_variance = global_sigma.float().pow(2)
            disagreement_variance = torch.zeros_like(global_mu)
        else:
            fused = self.fusion(
                local_mu=local_mu,
                local_sigma=local_sigma,
                global_mu=global_mu,
                global_sigma=global_sigma,
            )
            fused_mu, fused_sigma = fused.mean, fused.std
            local_weight, global_weight = fused.local_weight, fused.global_weight
            within_variance, disagreement_variance = fused.within_variance, fused.disagreement_variance

        grade_probs = self._probs(fused_mu, fused_sigma)
        local_grade_probs = self._probs(local_mu, local_sigma)
        global_grade_probs = self._probs(global_mu, global_sigma)
        binary_prob = grade_probs[:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        ordinal_logits = grade_probs_to_ordinal_logits(grade_probs)
        embedding = self.embedding_project(
            torch.cat([z_local, z_global, torch.abs(z_local - z_global), z_local * z_global], dim=1)
        )
        return {
            "binary_logits": torch.logit(binary_prob),
            "ordinal_logits": ordinal_logits,
            "embedding": embedding,
            "local_embedding": z_local,
            "global_embedding": z_global,
            "grade_probs": grade_probs,
            "local_grade_probs": local_grade_probs,
            "global_grade_probs": global_grade_probs,
            "binary_prob": binary_prob,
            "severity_score": fused_mu,
            "fused_mu": fused_mu,
            "fused_sigma": fused_sigma,
            "local_mu": local_mu,
            "local_sigma": local_sigma,
            "global_mu": global_mu,
            "global_sigma": global_sigma,
            "local_weight": local_weight,
            "global_weight": global_weight,
            "within_variance": within_variance,
            "disagreement_variance": disagreement_variance,
            "total_variance": fused_sigma.float().pow(2),
            **local_features,
        }


def build_distributional_progression_model(config: dict[str, Any]) -> nn.Module:
    return DisagreementAwareProgressionModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        feature_dim=int(config.get("feature_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        mode=str(config.get("distribution_mode", "local_global")),
        beta=float(config.get("disagreement_beta", 1.0)),
    )
