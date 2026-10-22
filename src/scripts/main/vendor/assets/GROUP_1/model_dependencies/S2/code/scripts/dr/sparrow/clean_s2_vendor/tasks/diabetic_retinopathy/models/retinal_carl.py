from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from clean_s2_vendor.tasks.diabetic_retinopathy.models.common import GeMPooling


class TinyConvFeaturePyramid(nn.Module):
    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (32, 64, 96, 128)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        current = in_channels
        for width in widths:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
            current = width
        self.stage3_dim = int(widths[-2])
        self.stage4_dim = int(widths[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        stage3 = None
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index == len(self.blocks) - 2:
                stage3 = hidden
        if stage3 is None:
            raise RuntimeError("TinyConvFeaturePyramid failed to materialize stage3 features.")
        return stage3, hidden


class TimmFeaturePyramid(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, backbone_img_size: int | None = None) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Backbone '{backbone_name}' requires timm. Install timm or use backbone=tiny_conv for smoke tests."
            ) from exc
        create_kwargs: dict[str, Any] = {
            "pretrained": pretrained,
            "features_only": True,
            "out_indices": (2, 3),
        }
        if backbone_img_size is not None:
            create_kwargs["img_size"] = int(backbone_img_size)
        self.backbone = _create_timm_feature_backbone(
            timm=timm,
            backbone_name=backbone_name,
            create_kwargs=create_kwargs,
        )
        channels = list(self.backbone.feature_info.channels())
        if len(channels) != 2:
            raise RuntimeError(f"Expected 2 feature maps from {backbone_name}, got {channels}")
        self.stage3_dim = int(channels[0])
        self.stage4_dim = int(channels[1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if len(features) != 2:
            raise RuntimeError(f"Expected 2 feature maps, got {len(features)}")
        normalized = []
        for feature, expected_channels in zip(features, (self.stage3_dim, self.stage4_dim), strict=True):
            if feature.ndim == 4 and feature.shape[-1] == expected_channels and feature.shape[1] != expected_channels:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            normalized.append(feature)
        return normalized[0], normalized[1]


def _is_missing_local_pretrained_weights(exc: Exception) -> bool:
    names = {item.__name__ for item in type(exc).__mro__}
    return any(
        "LocalEntryNotFoundError" in name or "OfflineModeIsEnabled" in name
        for name in names
    )


def _create_timm_feature_backbone(*, timm: Any, backbone_name: str, create_kwargs: dict[str, Any]) -> Any:
    attempt_kwargs = dict(create_kwargs)
    try:
        return timm.create_model(backbone_name, **attempt_kwargs)
    except TypeError:
        attempt_kwargs.pop("img_size", None)
    try:
        return timm.create_model(backbone_name, **attempt_kwargs)
    except Exception as exc:
        if bool(attempt_kwargs.get("pretrained")) and _is_missing_local_pretrained_weights(exc):
            fallback_kwargs = dict(attempt_kwargs)
            fallback_kwargs["pretrained"] = False
            return timm.create_model(backbone_name, **fallback_kwargs)
        raise


def _build_feature_pyramid(
    backbone_name: str,
    pretrained: bool,
    backbone_img_size: int | None = None,
) -> tuple[nn.Module, int, int]:
    lowered = backbone_name.lower()
    if lowered == "tiny_conv":
        backbone = TinyConvFeaturePyramid()
        return backbone, backbone.stage3_dim, backbone.stage4_dim
    if lowered == "retfound_mae_naturecfp":
        raise NotImplementedError(
            "RETFound is currently supported only for global-only DR experiments; "
            "use model.name=global_ordinal with backbone=retfound_mae_naturecfp."
        )
    backbone = TimmFeaturePyramid(backbone_name, pretrained=pretrained, backbone_img_size=backbone_img_size)
    return backbone, backbone.stage3_dim, backbone.stage4_dim


class QueryAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, local_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(local_tokens).squeeze(-1), dim=1)
        pooled = torch.sum(local_tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(tokens)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens


class ScaledCosineBinaryHead(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.empty(2, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = max(float(scale), 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = F.normalize(self.norm(x), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        logits = F.linear(feature, weight).clamp(-1.0 + 1e-6, 1.0 - 1e-6) * self.scale
        return logits[:, 1] - logits[:, 0]


class ScaledCosineOrdinalHead(nn.Module):
    def __init__(self, dim: int, num_thresholds: int = 4, scale: float = 12.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.empty(num_thresholds, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = max(float(scale), 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = F.normalize(self.norm(x), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(feature, weight).clamp(-1.0 + 1e-6, 1.0 - 1e-6) * self.scale


class RetinalFeatureCore(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        pretrained: bool,
        backbone_img_size: int | None,
        pooling: str,
        local_query_count: int,
        local_dim: int,
        fusion_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.feature_pyramid, stage3_dim, stage4_dim = _build_feature_pyramid(
            backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
        )
        self.global_pool = GeMPooling() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
        self.global_project = nn.Sequential(
            nn.Linear(stage4_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.local_project = nn.Conv2d(stage3_dim, local_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.local_queries = nn.Parameter(torch.randn(local_query_count, local_dim) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            embed_dim=local_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.local_pool = QueryAttentionPool(local_dim)
        self.local_fuse = nn.Sequential(
            nn.Linear(local_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.token_score = nn.Linear(local_dim, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stage3, stage4 = self.feature_pyramid(image)
        pooled = self.global_pool(stage4)
        if pooled.ndim == 4:
            pooled = pooled.flatten(1)
        z_global = self.global_project(pooled)

        local_feature = self.local_project(stage3)
        batch_size, _, height, width = local_feature.shape
        token_grid = local_feature.flatten(2).transpose(1, 2)
        queries = self.local_queries.unsqueeze(0).expand(batch_size, -1, -1)
        local_tokens, attention_weights = self.query_attention(
            queries,
            token_grid,
            token_grid,
            need_weights=True,
            average_attn_weights=False,
        )
        pooled_query, query_gate = self.local_pool(local_tokens)
        z_local_query = self.local_fuse(pooled_query)

        context_token = token_grid.mean(dim=1)
        z_local_context = self.local_fuse(context_token)

        saliency_weights = torch.softmax(self.token_score(token_grid).squeeze(-1), dim=1)
        pooled_saliency = torch.sum(token_grid * saliency_weights.unsqueeze(-1), dim=1)
        z_local_saliency = self.local_fuse(pooled_saliency)

        fused_local_tokens = self.local_fuse(local_tokens)
        attention_map = attention_weights.mean(dim=1).reshape(batch_size, queries.shape[1], height, width)
        return {
            "global_embedding": z_global,
            "local_embedding": z_local_query,
            "context_embedding": z_local_context,
            "saliency_embedding": z_local_saliency,
            "local_query_tokens": local_tokens,
            "local_query_tokens_fused": fused_local_tokens,
            "query_attention_maps": attention_map,
            "query_gate": query_gate,
            "token_grid": token_grid,
            "saliency_weights": saliency_weights,
        }


class RetinalCARLModel(nn.Module):
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
        self.core = RetinalFeatureCore(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
            pooling=pooling,
            local_query_count=local_query_count,
            local_dim=local_dim,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.binary_head = nn.Linear(fusion_dim, 1)
        self.ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.core(image)
        z_global = features["global_embedding"]
        z_local = features["local_embedding"]
        fused = self.fuse(
            torch.cat(
                [
                    z_global,
                    z_local,
                    torch.abs(z_global - z_local),
                    z_global * z_local,
                ],
                dim=-1,
            )
        )
        return {
            "binary_logits": self.binary_head(fused).squeeze(1),
            "ordinal_logits": self.ordinal_head(fused),
            "embedding": fused,
            **features,
        }


class RetinalCARLCosineModel(nn.Module):
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
        cosine_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.core = RetinalFeatureCore(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
            pooling=pooling,
            local_query_count=local_query_count,
            local_dim=local_dim,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim * 4 + 1, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.binary_head = ScaledCosineBinaryHead(fusion_dim, scale=cosine_scale)
        self.ordinal_head = ScaledCosineOrdinalHead(fusion_dim, num_thresholds=ordinal_thresholds, scale=cosine_scale)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.core(image)
        z_global = features["global_embedding"]
        z_local = features["local_embedding"]
        cosine_agreement = F.cosine_similarity(z_global, z_local, dim=-1).unsqueeze(-1)
        fused = self.fuse(
            torch.cat(
                [
                    z_global,
                    z_local,
                    torch.abs(z_global - z_local),
                    z_global * z_local,
                    cosine_agreement,
                ],
                dim=-1,
            )
        )
        return {
            "binary_logits": self.binary_head(fused),
            "ordinal_logits": self.ordinal_head(fused),
            "embedding": fused,
            "cosine_agreement": cosine_agreement.squeeze(-1),
            **features,
        }


class RetinalEvidencePathMixModel(nn.Module):
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
        self.core = RetinalFeatureCore(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
            pooling=pooling,
            local_query_count=local_query_count,
            local_dim=local_dim,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.path_gate = nn.Sequential(
            nn.Linear(fusion_dim * 4 + 3, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 3),
        )
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim * 5, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.binary_head = nn.Linear(fusion_dim, 1)
        self.ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.core(image)
        z_global = features["global_embedding"]
        path_query = features["local_embedding"]
        path_saliency = features["saliency_embedding"]
        path_context = features["context_embedding"]
        agreements = torch.stack(
            [
                F.cosine_similarity(path_query, path_saliency, dim=-1),
                F.cosine_similarity(path_query, path_context, dim=-1),
                F.cosine_similarity(path_saliency, path_context, dim=-1),
            ],
            dim=-1,
        )
        gate_input = torch.cat(
            [z_global, path_query, path_saliency, path_context, agreements],
            dim=-1,
        )
        path_logits = self.path_gate(gate_input)
        path_weights = torch.softmax(path_logits, dim=-1)
        path_stack = torch.stack([path_query, path_saliency, path_context], dim=1)
        mixed_local = torch.sum(path_stack * path_weights.unsqueeze(-1), dim=1)
        path_dispersion = torch.std(path_stack, dim=1, unbiased=False)
        fused = self.fuse(
            torch.cat(
                [
                    z_global,
                    mixed_local,
                    torch.abs(z_global - mixed_local),
                    z_global * mixed_local,
                    path_dispersion,
                ],
                dim=-1,
            )
        )
        return {
            "binary_logits": self.binary_head(fused).squeeze(1),
            "ordinal_logits": self.ordinal_head(fused),
            "embedding": fused,
            "path_weights": path_weights,
            "path_dispersion": path_dispersion,
            **features,
        }


class RetinalGraphResidualModel(nn.Module):
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
        graph_layers: int = 2,
        graph_dropout: float = 0.1,
        residual_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.core = RetinalFeatureCore(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
            pooling=pooling,
            local_query_count=local_query_count,
            local_dim=local_dim,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.context_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.graph_layers = nn.ModuleList(
            [ResidualSelfAttentionBlock(fusion_dim, num_heads=attention_heads, dropout=graph_dropout) for _ in range(graph_layers)]
        )
        self.residual_alpha = float(residual_alpha)
        self.readout = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.binary_head = nn.Linear(fusion_dim, 1)
        self.ordinal_head = nn.Linear(fusion_dim, ordinal_thresholds)
        nn.init.trunc_normal_(self.context_token, std=0.02)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.core(image)
        z_global = features["global_embedding"]
        z_query = features["local_embedding"]
        z_context = features["context_embedding"]
        local_query_tokens = features["local_query_tokens_fused"]
        batch_size = z_global.shape[0]

        seed_tokens = torch.cat(
            [
                z_global.unsqueeze(1),
                z_query.unsqueeze(1),
                z_context.unsqueeze(1),
                local_query_tokens,
                self.context_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        hidden = seed_tokens
        for layer in self.graph_layers:
            hidden = layer(hidden)

        graph_global = hidden[:, 0]
        graph_query = hidden[:, 1]
        graph_context = hidden[:, 2]
        graph_local_mean = hidden[:, 3:-1].mean(dim=1)
        residual_feature = graph_global + self.residual_alpha * (graph_local_mean - z_global)
        fused = self.readout(
            torch.cat(
                [
                    residual_feature,
                    graph_query,
                    graph_context,
                    torch.abs(graph_query - graph_context),
                ],
                dim=-1,
            )
        )
        return {
            "binary_logits": self.binary_head(fused).squeeze(1),
            "ordinal_logits": self.ordinal_head(fused),
            "embedding": fused,
            "graph_tokens": hidden,
            "graph_local_mean": graph_local_mean,
            **features,
        }


def build_retinal_carl_model(config: dict[str, Any]) -> nn.Module:
    return RetinalCARLModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        local_dim=int(config.get("local_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
    )


def build_retinal_carl_cosine_model(config: dict[str, Any]) -> nn.Module:
    return RetinalCARLCosineModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        local_dim=int(config.get("local_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        cosine_scale=float(config.get("cosine_scale", 16.0)),
    )


def build_retinal_evidence_pathmix_model(config: dict[str, Any]) -> nn.Module:
    return RetinalEvidencePathMixModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        local_dim=int(config.get("local_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
    )


def build_retinal_graph_residual_model(config: dict[str, Any]) -> nn.Module:
    return RetinalGraphResidualModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        local_dim=int(config.get("local_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        graph_layers=int(config.get("graph_layers", 2)),
        graph_dropout=float(config.get("graph_dropout", 0.1)),
        residual_alpha=float(config.get("residual_alpha", 0.5)),
    )
