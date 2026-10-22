from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tasks.diabetic_retinopathy.models.cerl.global_branch import GlobalRetinaBranch
from tasks.diabetic_retinopathy.models.cerl.local_branch import LesionQueryLocalBranch
from tasks.diabetic_retinopathy.models.retinal_carl import _build_feature_pyramid


class CERLLocalGlobalModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        backbone_img_size: int | None = None,
        pooling: str = "gem",
        local_query_count: int = 6,
        local_dim: int = 256,
        fusion_dim: int = 256,
        attention_heads: int = 4,
        ordinal_thresholds: int = 4,
    ) -> None:
        super().__init__()
        self.feature_pyramid, stage3_dim, stage4_dim = _build_feature_pyramid(
            backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
        )
        self.local_branch = LesionQueryLocalBranch(
            stage3_dim=stage3_dim,
            local_dim=local_dim,
            query_count=local_query_count,
            attention_heads=attention_heads,
        )
        self.local_project = nn.Sequential(nn.Linear(local_dim, fusion_dim), nn.GELU(), nn.LayerNorm(fusion_dim))
        self.global_branch = GlobalRetinaBranch(stage4_dim=stage4_dim, fusion_dim=fusion_dim, pooling=pooling)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.local_binary_head = nn.Linear(fusion_dim, 1)
        self.global_binary_head = nn.Linear(fusion_dim, 1)
        self.binary_head = nn.Linear(fusion_dim, 1)
        self.local_ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)
        self.global_ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)
        self.ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stage3, stage4 = self.feature_pyramid(image)
        local_features = self.local_branch(stage3)
        z_local = self.local_project(local_features["local_embedding_raw"])
        z_global = self.global_branch(stage4)
        fused = self.fusion(
            torch.cat([z_local, z_global, torch.abs(z_local - z_global), z_local * z_global], dim=-1)
        )

        local_binary = self.local_binary_head(z_local).squeeze(1)
        global_binary = self.global_binary_head(z_global).squeeze(1)
        fusion_binary = self.binary_head(fused).squeeze(1)
        local_prob = torch.sigmoid(local_binary)
        global_prob = torch.sigmoid(global_binary)
        fusion_prob = torch.sigmoid(fusion_binary)

        return {
            "binary_logits": fusion_binary,
            "ordinal_logits": self.ordinal_head(fused),
            "embedding": fused,
            "local_embedding": z_local,
            "global_embedding": z_global,
            "local_binary_logits": local_binary,
            "global_binary_logits": global_binary,
            "local_ordinal_logits": self.local_ordinal_head(z_local),
            "global_ordinal_logits": self.global_ordinal_head(z_global),
            "local_prob_pos": local_prob,
            "global_prob_pos": global_prob,
            "fusion_prob_pos": fusion_prob,
            "evidence_disagreement": torch.maximum(
                torch.abs(local_prob - fusion_prob),
                torch.abs(global_prob - fusion_prob),
            ),
            "local_global_abs_diff": torch.abs(local_prob - global_prob),
            "local_global_cosine": F.cosine_similarity(z_local, z_global, dim=-1),
            **local_features,
        }


def build_cerl_local_global_model(config: dict[str, Any]) -> nn.Module:
    return CERLLocalGlobalModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        local_dim=int(config.get("local_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
    )
