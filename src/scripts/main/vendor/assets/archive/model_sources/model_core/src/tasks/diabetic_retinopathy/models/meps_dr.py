from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from tasks.diabetic_retinopathy.models.global_ordinal import DRGlobalOrdinalModel


def _ordered_thresholds(raw_anchor: torch.Tensor, raw_deltas: torch.Tensor) -> torch.Tensor:
    thresholds = [raw_anchor]
    current = raw_anchor
    for delta in raw_deltas:
        current = current + F.softplus(delta) + 0.05
        thresholds.append(current)
    return torch.stack(thresholds, dim=0)


def _ordered_cumulative_weights(raw_base: torch.Tensor, raw_deltas: torch.Tensor) -> torch.Tensor:
    base = F.softplus(raw_base)
    weights = [base]
    current = base
    for delta in raw_deltas:
        current = current + F.softplus(delta)
        weights.append(current)
    return torch.stack(weights, dim=0)


def _descending_cumulative_probs(raw_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(raw_logits.float())
    reversed_probs = torch.flip(probs, dims=(1,))
    ordered_reversed = torch.cummax(reversed_probs, dim=1).values
    ordered = torch.flip(ordered_reversed, dims=(1,))
    return ordered.clamp(1e-6, 1.0 - 1e-6)


def _cumulative_probs_to_grade_probs(cumulative_probs: torch.Tensor) -> torch.Tensor:
    probs = torch.zeros(
        cumulative_probs.shape[0],
        cumulative_probs.shape[1] + 1,
        device=cumulative_probs.device,
        dtype=cumulative_probs.dtype,
    )
    probs[:, 0] = 1.0 - cumulative_probs[:, 0]
    for index in range(1, cumulative_probs.shape[1]):
        probs[:, index] = cumulative_probs[:, index - 1] - cumulative_probs[:, index]
    probs[:, -1] = cumulative_probs[:, -1]
    probs = probs.clamp_min(1e-8)
    return probs / probs.sum(dim=1, keepdim=True)


class DRMultiEvidenceProgressionStateModel(DRGlobalOrdinalModel):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        pooling: str = "gem",
        backbone_root: str | None = None,
        backbone_img_size: int | None = None,
        ordinal_thresholds: int = 4,
        evidence_query_count: int = 5,
        evidence_hidden_dim: int | None = None,
        attention_temperature: float = 1.0,
        progression_enabled: bool = False,
        method_off: bool = False,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pooling=pooling,
            backbone_root=backbone_root,
            backbone_img_size=backbone_img_size,
            ordinal_thresholds=ordinal_thresholds,
        )
        hidden_dim = int(evidence_hidden_dim or self.feature_dim)
        self.evidence_query_count = int(evidence_query_count)
        self.progression_enabled = bool(progression_enabled)
        self.method_off = bool(method_off)
        self.attention_temperature = max(float(attention_temperature), 1e-6)
        self._backbone_frozen = False
        if self.method_off:
            return
        self.token_norm = torch.nn.LayerNorm(self.feature_dim)
        self.query_embed = torch.nn.Embedding(self.evidence_query_count, self.feature_dim)
        self.query_proj = torch.nn.Linear(self.feature_dim, hidden_dim, bias=False)
        self.key_proj = torch.nn.Linear(self.feature_dim, hidden_dim, bias=False)
        self.value_proj = torch.nn.Linear(self.feature_dim, hidden_dim, bias=False)
        self.global_context_proj = torch.nn.Linear(self.feature_dim, hidden_dim)
        self.query_output_norm = torch.nn.LayerNorm(hidden_dim)
        self.evidence_head = torch.nn.Linear(hidden_dim, 1)
        self.raw_weight_base = torch.nn.Parameter(torch.zeros(self.evidence_query_count))
        self.raw_weight_deltas = torch.nn.Parameter(torch.zeros(max(ordinal_thresholds - 1, 0), self.evidence_query_count))
        self.raw_threshold_anchor = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.raw_threshold_deltas = torch.nn.Parameter(torch.zeros(max(ordinal_thresholds - 1, 0), dtype=torch.float32))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_frozen = not bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)

    def train(self, mode: bool = True) -> "DRMultiEvidenceProgressionStateModel":
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def _evidence_tokens(self, feature_map: torch.Tensor, global_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)
        query_tokens = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_proj(query_tokens)
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        attn_logits = torch.einsum("bqc,btc->bqt", query, keys) / math.sqrt(float(query.shape[-1]))
        attn_logits = attn_logits / self.attention_temperature
        attention = torch.softmax(attn_logits, dim=-1)
        pooled = torch.einsum("bqt,btc->bqc", attention, values)
        pooled = self.query_output_norm(pooled)
        pooled = pooled.clone()
        pooled[:, -1, :] = pooled[:, -1, :] + self.global_context_proj(global_embedding)
        return pooled, attention.reshape(batch_size, self.evidence_query_count, height, width)

    def _forward_method_off(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self._extract_backbone_embedding(image)
        binary_logits = self.binary_head(fused).squeeze(1)
        ordinal_logits = self.ordinal_head(fused)
        cumulative_probs = torch.sigmoid(ordinal_logits.float()).clamp(1e-6, 1.0 - 1e-6)
        grade_probs = _cumulative_probs_to_grade_probs(cumulative_probs)
        screening_prob = torch.sigmoid(binary_logits.float())
        expected_grade = torch.sum(
            grade_probs
            * torch.arange(grade_probs.shape[1], device=grade_probs.device, dtype=grade_probs.dtype).unsqueeze(0),
            dim=1,
        )
        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": fused,
            "grade_probs": grade_probs,
            "screening_prob": screening_prob,
            "screening_probability": screening_prob,
            "cumulative_risk": screening_prob,
            "expected_grade": expected_grade,
            "evidence_state_mean": fused.float().norm(dim=1),
            "evidence_state_std": fused.float().std(dim=1),
            "query_attention_entropy": torch.zeros_like(screening_prob),
            "query_attention_overlap_mean": torch.zeros_like(screening_prob),
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.method_off:
            return self._forward_method_off(image)

        feature_map = self._extract_backbone_feature_map(image)
        global_embedding = self._pool_feature_map(feature_map)
        pooled_queries, attention_maps = self._evidence_tokens(feature_map, global_embedding)
        evidence_state = F.softplus(self.evidence_head(pooled_queries).squeeze(-1))
        cumulative_weights = _ordered_cumulative_weights(self.raw_weight_base, self.raw_weight_deltas)
        thresholds = _ordered_thresholds(self.raw_threshold_anchor, self.raw_threshold_deltas)
        raw_ordinal_logits = evidence_state @ cumulative_weights.transpose(0, 1) - thresholds.unsqueeze(0)
        cumulative_probs = _descending_cumulative_probs(raw_ordinal_logits)
        ordinal_logits = torch.logit(cumulative_probs)
        grade_probs = _cumulative_probs_to_grade_probs(cumulative_probs)
        binary_logits = ordinal_logits[:, 1] if ordinal_logits.shape[1] > 1 else ordinal_logits[:, 0]
        screening_prob = torch.sigmoid(binary_logits.float())
        expected_grade = torch.sum(
            grade_probs
            * torch.arange(grade_probs.shape[1], device=grade_probs.device, dtype=grade_probs.dtype).unsqueeze(0),
            dim=1,
        )
        flattened_attention = attention_maps.flatten(2).clamp_min(1e-8)
        attention_similarity = torch.einsum("bqt,bkt->bqk", flattened_attention, flattened_attention)
        if self.evidence_query_count > 1:
            offdiag_mask = ~torch.eye(self.evidence_query_count, device=attention_similarity.device, dtype=torch.bool)
            diversity_loss = attention_similarity[:, offdiag_mask].mean()
            overlap_mean = attention_similarity[:, offdiag_mask].mean(dim=1)
        else:
            diversity_loss = binary_logits.sum() * 0.0
            overlap_mean = torch.zeros_like(screening_prob)
        attention_entropy = -(flattened_attention * flattened_attention.log()).sum(dim=-1).mean(dim=1)
        progression_score = torch.sum(evidence_state * cumulative_weights[-1].unsqueeze(0), dim=1)
        return {
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "embedding": global_embedding,
            "grade_probs": grade_probs,
            "screening_prob": screening_prob,
            "screening_probability": screening_prob,
            "cumulative_risk": screening_prob,
            "expected_grade": expected_grade,
            "evidence_state": evidence_state,
            "evidence_diversity_loss": diversity_loss,
            "progression_score": progression_score,
            "query_attention_entropy": attention_entropy,
            "query_attention_overlap_mean": overlap_mean,
            "evidence_state_mean": evidence_state.mean(dim=1),
            "evidence_state_std": evidence_state.std(dim=1),
        }


def build_meps_dr_model(config: dict[str, Any]) -> DRMultiEvidenceProgressionStateModel:
    return DRMultiEvidenceProgressionStateModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        pooling=str(config.get("pooling", "gem")),
        backbone_root=str(config["backbone_root"]) if config.get("backbone_root") is not None else None,
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        ordinal_thresholds=int(config.get("ordinal_thresholds", 4)),
        evidence_query_count=int(config.get("evidence_query_count", 5)),
        evidence_hidden_dim=(int(config["evidence_hidden_dim"]) if config.get("evidence_hidden_dim") is not None else None),
        attention_temperature=float(config.get("attention_temperature", 1.0)),
        progression_enabled=bool(config.get("progression_enabled", False)),
        method_off=bool(config.get("method_off", False)),
    )
