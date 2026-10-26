from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=(-2, -1)).pow(1.0 / self.p)
        return x


class TinyConvBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class RETFoundBackbone(nn.Module):
    def __init__(self, model_root: str | Path) -> None:
        super().__init__()
        root = Path(model_root)
        if not root.exists():
            raise FileNotFoundError(f"RETFound model root does not exist: {root}")
        self.backend = "transformers"
        transformers_error: Exception | None = None
        try:
            from transformers import AutoModel  # type: ignore
            self.encoder = AutoModel.from_pretrained(root.as_posix(), local_files_only=True)
            config = getattr(self.encoder, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(config, "embed_dim", None)
            if hidden_size is None:
                hidden_size = getattr(self.encoder, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Could not infer RETFound hidden size from encoder config.")
            self.out_dim = int(hidden_size)
            return
        except Exception as exc:
            transformers_error = exc

        checkpoint_paths = [
            path
            for pattern in ("*.pth", "*.pt", "*.ckpt", "*.bin")
            for path in root.rglob(pattern)
            if path.is_file()
        ]
        if not checkpoint_paths:
            raise RuntimeError(
                "RETFound could not be loaded as a local Transformers model and no official checkpoint was found "
                f"under {root}. Transformers error: {transformers_error!r}"
            ) from transformers_error
        try:
            import timm  # type: ignore
        except Exception as exc:
            raise RuntimeError("Official RETFound checkpoints require timm for the ViT-Large encoder.") from exc

        self.backend = "timm_checkpoint"
        try:
            self.encoder = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,
                global_pool="",
                dynamic_img_size=True,
            )
        except TypeError:
            self.encoder = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
        checkpoint_path = max(checkpoint_paths, key=lambda path: path.stat().st_size)
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = raw
        if isinstance(raw, dict):
            for container_key in ("model", "state_dict", "teacher", "student"):
                candidate = raw.get(container_key)
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported RETFound checkpoint payload at {checkpoint_path}")
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_key = str(key)
            for prefix in ("module.", "model.", "encoder."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            normalized[normalized_key] = value
        encoder_state = self.encoder.state_dict()
        compatible = {
            key: value
            for key, value in normalized.items()
            if key in encoder_state and tuple(value.shape) == tuple(encoder_state[key].shape)
        }
        if not compatible:
            raise RuntimeError(
                f"RETFound checkpoint {checkpoint_path} has no parameters compatible with timm vit_large_patch16_224."
            )
        self.encoder.load_state_dict(compatible, strict=False)
        self.out_dim = int(getattr(self.encoder, "num_features", 1024))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "transformers":
            try:
                output = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
            except TypeError:
                output = self.encoder(pixel_values=x)
            hidden = getattr(output, "last_hidden_state", None)
        else:
            hidden = self.encoder.forward_features(x)
            if isinstance(hidden, dict):
                hidden_map = hidden
                hidden = hidden_map.get("x")
                if hidden is None:
                    hidden = hidden_map.get("last_hidden_state")
        if hidden is None:
            raise RuntimeError("RETFound encoder did not return spatial features.")
        if hidden.ndim == 4:
            if hidden.shape[1] == self.out_dim:
                return hidden
            if hidden.shape[-1] == self.out_dim:
                return hidden.permute(0, 3, 1, 2).contiguous()
            raise RuntimeError(f"Unsupported RETFound spatial feature shape: {tuple(hidden.shape)}")
        if hidden.ndim != 3:
            raise RuntimeError("RETFound encoder must return token features shaped [B, tokens, dim].")
        if hidden.shape[1] <= 1:
            raise RuntimeError("RETFound encoder returned no patch tokens.")
        prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1))
        patch_tokens = hidden[:, prefix_tokens:, :]
        spatial_size = int(round(float(patch_tokens.shape[1]) ** 0.5))
        if spatial_size * spatial_size != int(patch_tokens.shape[1]):
            raise RuntimeError(
                f"RETFound patch token count {patch_tokens.shape[1]} cannot be reshaped into a square grid."
            )
        return patch_tokens.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], spatial_size, spatial_size)


def build_classifier_backbone(
    name: str,
    pretrained: bool,
    *,
    backbone_root: str | Path | None = None,
    backbone_img_size: int | None = None,
) -> tuple[nn.Module, int]:
    lowered = name.lower()
    if lowered == "tiny_conv":
        backbone = TinyConvBackbone()
        return backbone, backbone.out_dim
    if lowered == "retfound_mae_naturecfp":
        if backbone_root is None:
            raise ValueError("RETFound backbone requires model.backbone_root to point at the downloaded local weights.")
        backbone = RETFoundBackbone(backbone_root)
        return backbone, backbone.out_dim

    try:
        import timm  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Backbone '{name}' requires timm. Install timm or use backbone=tiny_conv for smoke tests."
        ) from exc

    create_kwargs = {
        "pretrained": pretrained,
        "num_classes": 0,
        "global_pool": "",
    }
    if backbone_img_size is not None:
        create_kwargs["img_size"] = int(backbone_img_size)
    try:
        model = timm.create_model(name, **create_kwargs)
    except TypeError:
        create_kwargs.pop("img_size", None)
        model = timm.create_model(name, **create_kwargs)
    feature_info = getattr(model, "feature_info", None)
    out_dim = None
    if feature_info is not None:
        try:
            out_dim = int(feature_info.channels()[-1])
        except Exception:
            out_dim = None
    if out_dim is None:
        out_dim = int(getattr(model, "num_features", 1536))
    return model, out_dim
