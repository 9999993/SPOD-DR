from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import torch
from torch.nn import functional as F

ORBIT_IDS = (
    "identity",
    "gamma",
    "contrast",
    "white_balance_channel_gain",
    "low_frequency_illumination_field",
    "sensor_noise",
    "jpeg_quality",
)


@dataclass(frozen=True)
class AppearanceOrbitConfig:
    seed: int = 20260906
    gamma_min: float = 0.86
    gamma_max: float = 1.16
    contrast_min: float = 0.86
    contrast_max: float = 1.14
    white_balance_min: float = 0.94
    white_balance_max: float = 1.06
    illumination_amplitude: float = 0.08
    sensor_noise_std: float = 0.012
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrbitResult:
    image: torch.Tensor
    fov_mask: torch.Tensor
    parameters: dict[str, Any]


def calibrate_source_quantiles(rows: Iterable[Any] | None = None, seed: int = 20260906) -> AppearanceOrbitConfig:
    """Return the pre-registered source-only transform envelope.

    The envelope is deliberately fixed and does not inspect held-domain
    outcomes.  ``rows`` is accepted for a stable public interface; callers may
    pass source rows, but no target statistic is ever consumed here.
    """

    del rows
    return AppearanceOrbitConfig(seed=int(seed))


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    return generator


def _sample_uniform(generator: torch.Generator, low: float, high: float, *, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(low, high, generator=generator).item())


def apply_orbit(
    image: torch.Tensor,
    fov_mask: torch.Tensor,
    orbit_id: str,
    sample_seed: int,
    config: AppearanceOrbitConfig | None = None,
) -> OrbitResult:
    """Apply a deterministic photometric, grade-preserving orbit.

    ``image`` is ``[C,H,W]`` or ``[B,C,H,W]`` in ``[0,1]``.  No operation
    changes spatial geometry or the supplied FOV mask.  JPEG is represented by
    deterministic scalar quantisation, which avoids an implementation-specific
    codec while retaining a registered compression perturbation.
    """

    cfg = config or AppearanceOrbitConfig()
    name = str(orbit_id).strip().lower()
    if name not in ORBIT_IDS:
        raise ValueError(f"unknown SPARROW orbit: {orbit_id}")
    if image.ndim not in (3, 4):
        raise ValueError(f"expected [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    if fov_mask.shape[-2:] != image.shape[-2:]:
        raise ValueError("FOV mask and image geometry differ")
    if not torch.is_floating_point(image):
        raise TypeError("orbit input must be floating point")
    generator = _generator(int(sample_seed) + int(cfg.seed), image.device)
    output = image.float().clamp(0.0, 1.0)
    parameters: dict[str, Any] = {"orbit_id": name, "sample_seed": int(sample_seed)}
    if name == "identity":
        pass
    elif name == "gamma":
        value = _sample_uniform(generator, cfg.gamma_min, cfg.gamma_max, device=image.device)
        output = output.clamp_min(1e-5).pow(value)
        parameters["gamma"] = value
    elif name == "contrast":
        value = _sample_uniform(generator, cfg.contrast_min, cfg.contrast_max, device=image.device)
        mean = output.mean(dim=(-2, -1), keepdim=True)
        output = (output - mean) * value + mean
        parameters["contrast"] = value
    elif name == "white_balance_channel_gain":
        gains = torch.empty((output.shape[-3],), device=image.device).uniform_(cfg.white_balance_min, cfg.white_balance_max, generator=generator)
        shape = (1, -1, 1, 1) if output.ndim == 4 else (-1, 1, 1)
        output = output * gains.view(*shape)
        parameters["channel_gains"] = [float(v) for v in gains.detach().cpu()]
    elif name == "low_frequency_illumination_field":
        h, w = output.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, h, device=image.device)
        xx = torch.linspace(-1.0, 1.0, w, device=image.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        phase = _sample_uniform(generator, -3.14159265, 3.14159265, device=image.device)
        field = 1.0 + cfg.illumination_amplitude * torch.sin(grid_x * 1.7 + phase) * torch.cos(grid_y * 1.3 - phase)
        output = output * field.view((1, 1, h, w) if output.ndim == 4 else (1, h, w))
        parameters["phase"] = phase
    elif name == "sensor_noise":
        noise = torch.randn(output.shape, generator=generator, device=image.device, dtype=output.dtype) * cfg.sensor_noise_std
        output = output + noise
        parameters["noise_std"] = cfg.sensor_noise_std
    elif name == "jpeg_quality":
        quality = int(torch.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max + 1, (), generator=generator, device=image.device).item())
        # Quantisation step increases as quality decreases. This is a stable
        # codec-free proxy for a JPEG round trip, preserving all geometry.
        step = (101.0 - float(quality)) / 500.0 + 1e-4
        output = torch.round(output / step) * step
        parameters["quality"] = quality
        parameters["quantization_step"] = step
    output = output.clamp(0.0, 1.0)
    return OrbitResult(output.to(dtype=image.dtype), fov_mask.clone(), parameters)


def orbit_registry_hash(config: AppearanceOrbitConfig) -> str:
    serialized = repr(config.to_dict()).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = ["ORBIT_IDS", "AppearanceOrbitConfig", "OrbitResult", "apply_orbit", "calibrate_source_quantiles", "orbit_registry_hash"]
