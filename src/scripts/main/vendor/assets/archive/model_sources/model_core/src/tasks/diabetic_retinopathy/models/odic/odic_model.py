from __future__ import annotations

import torch

from tasks.diabetic_retinopathy.models.global_ordinal import DRGlobalOrdinalModel
from tasks.diabetic_retinopathy.models.odic.decision_maps import build_ordinal_decision_maps


class ODICGlobalOrdinalModel(DRGlobalOrdinalModel):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        pooling: str = "gem",
        backbone_root: str | None = None,
        backbone_img_size: int | None = None,
        ordinal_thresholds: int = 4,
        odic_output_size: int = 14,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=backbone_root,
            backbone_img_size=backbone_img_size,
            ordinal_thresholds=ordinal_thresholds,
        )
        self.odic_output_size = int(odic_output_size)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self._extract_backbone_feature_map(image)
        fused = self._pool_feature_map(feature_map)
        binary_logits = self.binary_head(fused).squeeze(1)
        ordinal_logits = self.ordinal_head(fused)
        decision_maps = build_ordinal_decision_maps(
            feature_map=feature_map,
            ordinal_weights=self.ordinal_head.weight,
            output_size=self.odic_output_size,
        )
        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": fused,
            "feature_map": feature_map,
            "decision_maps": decision_maps,
        }
