from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from disease_representation.adapter import DiseaseRepresentationAdapter
from disease_representation.prototype import DiseasePrototypeHead
from tasks.diabetic_retinopathy.models.distributional_progression import (
    build_cumulative_risk_alignment_model,
    build_distributional_progression_model,
)
from tasks.diabetic_retinopathy.models.clinical_risk import (
    build_clinical_risk_representation_learning_model,
)
from tasks.diabetic_retinopathy.models.clinical_risk_transport import (
    build_clinical_risk_transportability_learning_model,
)
from tasks.diabetic_retinopathy.models.common import GeMPooling, build_classifier_backbone
from tasks.diabetic_retinopathy.models.retinal_carl import (
    build_retinal_carl_cosine_model,
    build_retinal_carl_model,
    build_retinal_evidence_pathmix_model,
    build_retinal_graph_residual_model,
)
from tasks.diabetic_retinopathy.models.cerl import build_cerl_local_global_model
from tasks.diabetic_retinopathy.models.rcel import build_rcel_evidence_model


class DRGlobalOrdinalModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        pooling: str = "gem",
        backbone_root: str | None = None,
        backbone_img_size: int | None = None,
        ordinal_thresholds: int = 4,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_classifier_backbone(
            backbone_name,
            pretrained=pretrained,
            backbone_root=backbone_root,
            backbone_img_size=backbone_img_size,
        )
        self.feature_dim = int(feature_dim)
        self.pool = GeMPooling() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(feature_dim)
        self.binary_head = nn.Linear(feature_dim, 1)
        self.ordinal_head = nn.Linear(feature_dim, ordinal_thresholds)

    def _extract_backbone_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.ndim == 4 and features.shape[-1] == self.feature_dim and features.shape[1] != self.feature_dim:
            features = features.permute(0, 3, 1, 2).contiguous()
        if features.ndim == 3:
            token_count = int(features.shape[1])
            prefix_tokens = 1 if int(round(float(token_count - 1) ** 0.5)) ** 2 == token_count - 1 else 0
            patch_tokens = features[:, prefix_tokens:, :]
            spatial_size = int(round(float(patch_tokens.shape[1]) ** 0.5))
            if spatial_size * spatial_size != int(patch_tokens.shape[1]):
                raise RuntimeError(f"Backbone token count cannot form a square feature map: {features.shape}")
            features = patch_tokens.transpose(1, 2).reshape(
                features.shape[0], self.feature_dim, spatial_size, spatial_size
            )
        return features

    def _pool_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        features = feature_map
        if features.ndim == 4:
            pooled = self.pool(features)
            if pooled.ndim == 4:
                pooled = pooled.flatten(1)
        else:
            pooled = features
        return self.norm(pooled)

    def _extract_backbone_embedding(self, image: torch.Tensor) -> torch.Tensor:
        return self._pool_feature_map(self._extract_backbone_feature_map(image))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self._extract_backbone_embedding(image)
        binary_logits = self.binary_head(fused).squeeze(1)
        ordinal_logits = self.ordinal_head(fused)
        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": fused,
        }


class DRDiseaseAdapterOrdinalModel(DRGlobalOrdinalModel):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        pooling: str = "gem",
        backbone_root: str | None = None,
        backbone_img_size: int | None = None,
        disease_dim: int = 256,
        adapter_hidden_dim: int = 512,
        adapter_dropout: float = 0.1,
        ordinal_thresholds: int = 4,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=backbone_root,
            backbone_img_size=backbone_img_size,
            ordinal_thresholds=ordinal_thresholds,
        )
        self.adapter = DiseaseRepresentationAdapter(
            in_dim=self.feature_dim,
            hidden_dim=int(adapter_hidden_dim),
            out_dim=int(disease_dim),
            dropout=float(adapter_dropout),
        )
        self.binary_head = nn.Linear(int(disease_dim), 1)
        self.ordinal_head = nn.Linear(int(disease_dim), ordinal_thresholds)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_embedding = self._extract_backbone_embedding(image)
        disease_embedding = self.adapter(backbone_embedding)
        binary_logits = self.binary_head(disease_embedding).squeeze(1)
        ordinal_logits = self.ordinal_head(disease_embedding)
        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": disease_embedding,
            "disease_embedding": disease_embedding,
            "backbone_embedding": backbone_embedding,
        }


class DRDiseaseAdapterProgressionModel(DRDiseaseAdapterOrdinalModel):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        pooling: str = "gem",
        backbone_root: str | None = None,
        backbone_img_size: int | None = None,
        disease_dim: int = 256,
        adapter_hidden_dim: int = 512,
        adapter_dropout: float = 0.1,
        prototype_order_margin: float = 0.25,
        ordinal_thresholds: int = 4,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=backbone_root,
            backbone_img_size=backbone_img_size,
            disease_dim=disease_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout=adapter_dropout,
            ordinal_thresholds=ordinal_thresholds,
        )
        self.prototype_head = DiseasePrototypeHead(dim=int(disease_dim), num_grades=5)
        self.prototype_order_margin = float(prototype_order_margin)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(image)
        disease_embedding = output["disease_embedding"]
        prototype_distances = self.prototype_head.distances(disease_embedding)
        output.update(
            {
                "prototype_distances": prototype_distances,
                "severity_score": self.prototype_head.severity_score(disease_embedding),
                "prototype_order_margin_loss": self.prototype_head.order_margin_loss(self.prototype_order_margin),
            }
        )
        return output


@dataclass
class DRModelSpec:
    name: str
    backbone: str
    pretrained: bool = True
    pooling: str = "gem"
    backbone_root: str | None = None


def build_dr_model(config: dict[str, Any]) -> nn.Module:
    model_name = str(config.get("name", "global_ordinal"))
    backbone = str(config.get("backbone", "efficientnet_b3"))
    pretrained = bool(config.get("pretrained", True))
    pooling = str(config.get("pooling", "gem"))
    backbone_root = config.get("backbone_root")
    backbone_img_size = config.get("backbone_img_size")
    if model_name == "global_ordinal":
        return DRGlobalOrdinalModel(
            backbone_name=backbone,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=str(backbone_root) if backbone_root is not None else None,
            backbone_img_size=(int(backbone_img_size) if backbone_img_size is not None else None),
        )
    if model_name == "disease_adapter_ordinal":
        return DRDiseaseAdapterOrdinalModel(
            backbone_name=backbone,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=str(backbone_root) if backbone_root is not None else None,
            backbone_img_size=(int(backbone_img_size) if backbone_img_size is not None else None),
            disease_dim=int(config.get("disease_dim", 256)),
            adapter_hidden_dim=int(config.get("adapter_hidden_dim", 512)),
            adapter_dropout=float(config.get("adapter_dropout", 0.1)),
        )
    if model_name == "disease_adapter_progression":
        return DRDiseaseAdapterProgressionModel(
            backbone_name=backbone,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=str(backbone_root) if backbone_root is not None else None,
            backbone_img_size=(int(backbone_img_size) if backbone_img_size is not None else None),
            disease_dim=int(config.get("disease_dim", 256)),
            adapter_hidden_dim=int(config.get("adapter_hidden_dim", 512)),
            adapter_dropout=float(config.get("adapter_dropout", 0.1)),
            prototype_order_margin=float(config.get("prototype_order_margin", 0.25)),
        )
    if model_name == "odic_ordinal":
        from tasks.diabetic_retinopathy.models.odic import ODICGlobalOrdinalModel

        return ODICGlobalOrdinalModel(
            backbone_name=backbone,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=str(backbone_root) if backbone_root is not None else None,
            backbone_img_size=(int(backbone_img_size) if backbone_img_size is not None else None),
            ordinal_thresholds=int(config.get("ordinal_thresholds", 4)),
            odic_output_size=int(config.get("odic_output_size", 14)),
        )
    if model_name == "distributional_progression":
        return build_distributional_progression_model(config)
    if model_name == "distributional_progression_cra":
        return build_cumulative_risk_alignment_model(config)
    if model_name == "distributional_progression_crrl":
        return build_clinical_risk_representation_learning_model(config)
    if model_name == "distributional_progression_crtl":
        return build_clinical_risk_transportability_learning_model(config)
    if model_name == "retinal_carl_b":
        return build_retinal_carl_model(config)
    if model_name == "retinal_carl_cosine":
        return build_retinal_carl_cosine_model(config)
    if model_name == "retinal_evidence_pathmix":
        return build_retinal_evidence_pathmix_model(config)
    if model_name == "retinal_graph_residual":
        return build_retinal_graph_residual_model(config)
    if model_name == "cerl_local_global":
        return build_cerl_local_global_model(config)
    if model_name == "rcel_evidence":
        return build_rcel_evidence_model(config)
    if model_name == "meps_dr":
        from tasks.diabetic_retinopathy.models.meps_dr import build_meps_dr_model

        return build_meps_dr_model(config)
    raise NotImplementedError(
        f"Formal DR model '{model_name}' is not implemented yet. "
        "Currently supported: global_ordinal, disease_adapter_ordinal, disease_adapter_progression, "
        "distributional_progression, distributional_progression_cra, distributional_progression_crrl, "
        "distributional_progression_crtl, "
        "retinal_carl_b, retinal_carl_cosine, "
        "retinal_evidence_pathmix, retinal_graph_residual, cerl_local_global, rcel_evidence, odic_ordinal, meps_dr."
    )
