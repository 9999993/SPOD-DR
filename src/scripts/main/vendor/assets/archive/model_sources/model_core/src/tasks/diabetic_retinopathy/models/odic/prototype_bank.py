from __future__ import annotations

import torch
import torch.nn.functional as F


class RunningGradeDecisionPrototypeBank:
    def __init__(
        self,
        *,
        num_grades: int,
        channels: int,
        height: int,
        width: int,
        momentum: float = 0.9,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_grades = int(num_grades)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.momentum = float(momentum)
        self.prototypes = torch.zeros(
            self.num_grades,
            self.channels,
            self.height,
            self.width,
            device=device,
            dtype=dtype,
        )
        self.counts = torch.zeros(self.num_grades, device=device, dtype=torch.float32)

    def _ensure_device(self, reference: torch.Tensor) -> None:
        if self.prototypes.device != reference.device or self.prototypes.dtype != reference.dtype:
            self.prototypes = self.prototypes.to(device=reference.device, dtype=reference.dtype)
            self.counts = self.counts.to(device=reference.device)

    def update(self, decision_maps: torch.Tensor, grades: torch.Tensor) -> None:
        self._ensure_device(decision_maps)
        for grade_index in range(self.num_grades):
            mask = grades == int(grade_index)
            if not bool(mask.any()):
                continue
            batch_mean = decision_maps[mask].mean(dim=0)
            if float(self.counts[grade_index].item()) <= 0.0:
                self.prototypes[grade_index] = batch_mean.detach()
            else:
                self.prototypes[grade_index] = (
                    float(self.momentum) * self.prototypes[grade_index]
                    + (1.0 - float(self.momentum)) * batch_mean.detach()
                )
            self.counts[grade_index] += float(mask.sum().item())

    def lookup(self, grades: torch.Tensor) -> torch.Tensor:
        self._ensure_device(grades.float())
        return self.prototypes[grades.long()]

    def adjacent_relation_loss(self) -> torch.Tensor:
        if self.prototypes.shape[0] <= 1:
            return self.prototypes.sum() * 0.0
        diffs = self.prototypes[1:] - self.prototypes[:-1]
        magnitudes = diffs.flatten(1).norm(dim=1)
        if magnitudes.numel() <= 1:
            return magnitudes.mean() * 0.0
        return F.smooth_l1_loss(magnitudes[1:], magnitudes[:-1])
