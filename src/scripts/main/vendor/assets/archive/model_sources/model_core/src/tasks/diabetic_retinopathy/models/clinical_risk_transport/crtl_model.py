from __future__ import annotations

from typing import Any

from torch import nn

from tasks.diabetic_retinopathy.models.clinical_risk.crrl_model import (
    ClinicalRiskRepresentationLearningModel,
)


class ClinicalRiskTransportabilityLearningModel(ClinicalRiskRepresentationLearningModel):
    def __init__(
        self,
        *,
        transport_mode: str = "baseline",
        **kwargs: Any,
    ) -> None:
        resolved_transport_mode = str(transport_mode).strip().lower()
        if resolved_transport_mode not in {"baseline", "rda", "qrc", "rda_qrc"}:
            raise ValueError(f"Unsupported CRTL transport_mode: {transport_mode}")
        kwargs = dict(kwargs)
        kwargs["crrl_mode"] = "consistency"
        super().__init__(**kwargs)
        self.transport_mode = resolved_transport_mode

    def forward(self, image):
        output = super().forward(image)
        output["transport_mode"] = self.transport_mode
        return output


def build_clinical_risk_transportability_learning_model(config: dict[str, Any]) -> nn.Module:
    return ClinicalRiskTransportabilityLearningModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        feature_dim=int(config.get("feature_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        mode=str(config.get("distribution_mode", "local_global")),
        beta=float(config.get("disagreement_beta", 1.0)),
        risk_hidden_dim=int(config.get("risk_hidden_dim", 256)),
        risk_embedding_dim=int(config.get("risk_embedding_dim", 256)),
        transport_mode=str(config.get("transport_mode", "baseline")),
    )
