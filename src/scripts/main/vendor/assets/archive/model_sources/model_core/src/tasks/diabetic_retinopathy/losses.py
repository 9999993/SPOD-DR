from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tasks.diabetic_retinopathy.models.clinical_risk_transport.quantile_risk_calibration import (
    quantile_risk_calibration_loss,
)
from tasks.diabetic_retinopathy.cargo_dr.group_loss import compute_grade_conditioned_soft_cvar_loss
from tasks.diabetic_retinopathy.models.clinical_risk_transport.risk_distribution_alignment import (
    risk_distribution_alignment_loss,
)
from tasks.diabetic_retinopathy.models.cerl.evidence_agreement import binary_kl_agreement
from tasks.diabetic_retinopathy.models.cerl.ordinal_constraint import ordinal_ranking_loss
from tasks.diabetic_retinopathy.models.odic.prototype_bank import RunningGradeDecisionPrototypeBank


def build_coral_targets(grades: torch.Tensor, num_thresholds: int = 4) -> torch.Tensor:
    thresholds = torch.arange(num_thresholds, device=grades.device, dtype=grades.dtype)
    return (grades.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)


@dataclass
class BinaryOrdinalLoss:
    binary_weight: float = 1.0
    ordinal_weight: float = 0.25
    agreement_weight: float = 0.0
    rank_weight: float = 0.0
    stability_weight: float = 0.0
    prototype_weight: float = 0.0
    prototype_order_weight: float = 0.0
    distribution_grade_weight: float = 0.0
    branch_weight: float = 0.0
    rps_weight: float = 0.0
    risk_consistency_weight: float = 0.0
    risk_ranking_weight: float = 0.0
    risk_invariance_weight: float = 0.0
    risk_distribution_alignment_weight: float = 0.0
    quantile_risk_calibration_weight: float = 0.0
    boundary_aware_weight: float = 0.0
    cargo_group_weight: float = 0.0
    cargo_cvar_alpha: float = 0.5
    evidence_diversity_weight: float = 0.0
    progression_direction_weight: float = 0.0
    odic_consistency_weight: float = 0.0
    odic_relation_weight: float = 0.0
    odic_num_grades: int = 5
    odic_momentum: float = 0.9
    transport_pair_sample_limit: int = 128
    boundary_pair_limit: int = 64
    rank_margin: float = 0.25
    boundary_margin: float = 0.05
    boundary_min_confidence: float = 0.0
    _odic_bank: RunningGradeDecisionPrototypeBank | None = None

    def __call__(
        self,
        *,
        binary_logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        binary_target: torch.Tensor,
        grade_target: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        ordinal_pos_weight: torch.Tensor | None = None,
        local_binary_logits: torch.Tensor | None = None,
        global_binary_logits: torch.Tensor | None = None,
        severity_score: torch.Tensor | None = None,
        stability_target_score: torch.Tensor | None = None,
        prototype_distances: torch.Tensor | None = None,
        prototype_order_margin_loss: torch.Tensor | None = None,
        grade_probs: torch.Tensor | None = None,
        local_grade_probs: torch.Tensor | None = None,
        global_grade_probs: torch.Tensor | None = None,
        screening_prob: torch.Tensor | None = None,
        cumulative_risk: torch.Tensor | None = None,
        risk_order_score: torch.Tensor | None = None,
        clinical_risk_embedding: torch.Tensor | None = None,
        stability_target_embedding: torch.Tensor | None = None,
        transport_risk_embedding: torch.Tensor | None = None,
        transport_cumulative_risk: torch.Tensor | None = None,
        cargo_group_probs: torch.Tensor | None = None,
        cargo_sample_weight: torch.Tensor | None = None,
        evidence_diversity_loss: torch.Tensor | None = None,
        progression_score: torch.Tensor | None = None,
        decision_maps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        binary_target = binary_target.to(dtype=torch.float32)
        grade_target = grade_target.to(dtype=torch.long)
        binary_loss_per_sample = F.binary_cross_entropy_with_logits(
            binary_logits.float(),
            binary_target.float(),
            pos_weight=pos_weight.float() if pos_weight is not None else None,
            reduction="none",
        )
        binary_loss = binary_loss_per_sample.mean()
        coral_targets = build_coral_targets(grade_target, num_thresholds=ordinal_logits.shape[1])
        ordinal_loss_per_threshold = F.binary_cross_entropy_with_logits(
            ordinal_logits.float(),
            coral_targets.float(),
            pos_weight=ordinal_pos_weight.float() if ordinal_pos_weight is not None else None,
            reduction="none",
        )
        ordinal_loss_per_sample = ordinal_loss_per_threshold.mean(dim=1)
        ordinal_loss = ordinal_loss_per_threshold.mean()
        supervised_per_sample_loss = (
            float(self.binary_weight) * binary_loss_per_sample
            + float(self.ordinal_weight) * ordinal_loss_per_sample
        )
        if cargo_sample_weight is not None:
            sample_weight = cargo_sample_weight.float().clamp_min(1e-6)
            normalized_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
            binary_loss = (binary_loss_per_sample * normalized_weight).mean()
            ordinal_loss = (ordinal_loss_per_sample * normalized_weight).mean()
            supervised_per_sample_loss = supervised_per_sample_loss * normalized_weight
        agreement_loss = binary_logits.sum() * 0.0
        if (
            float(self.agreement_weight) > 0
            and local_binary_logits is not None
            and global_binary_logits is not None
        ):
            agreement_loss = binary_kl_agreement(
                local_logits=local_binary_logits.float(),
                global_logits=global_binary_logits.float(),
                fusion_logits=binary_logits.float(),
            )
        if severity_score is None:
            severity_score = ordinal_logits.float().sum(dim=1)
        rank_loss = binary_logits.sum() * 0.0
        if float(self.rank_weight) > 0:
            rank_loss = ordinal_ranking_loss(
                severity_score=severity_score.float(),
                grades=grade_target,
                margin=float(self.rank_margin),
            )
        stability_loss = binary_logits.sum() * 0.0
        if float(self.stability_weight) > 0 and stability_target_score is not None:
            stability_loss = F.smooth_l1_loss(severity_score.float(), stability_target_score.float())
        prototype_loss = binary_logits.sum() * 0.0
        if float(self.prototype_weight) > 0 and prototype_distances is not None:
            prototype_loss = F.cross_entropy(-prototype_distances.float(), grade_target)
        prototype_order_loss = binary_logits.sum() * 0.0
        if float(self.prototype_order_weight) > 0 and prototype_order_margin_loss is not None:
            prototype_order_loss = prototype_order_margin_loss.float()
        distribution_grade_loss = binary_logits.sum() * 0.0
        branch_loss = binary_logits.sum() * 0.0
        rps_loss = binary_logits.sum() * 0.0
        if grade_probs is not None:
            safe_probs = grade_probs.float().clamp_min(1e-7)
            distribution_grade_loss = -safe_probs.gather(1, grade_target.view(-1, 1)).log().mean()
            cumulative = safe_probs.cumsum(dim=1)[:, :-1]
            targets = torch.stack([(grade_target <= threshold).float() for threshold in range(4)], dim=1)
            rps_loss = ((cumulative - targets) ** 2).mean()
        if local_grade_probs is not None and global_grade_probs is not None:
            local_safe = local_grade_probs.float().clamp_min(1e-7)
            global_safe = global_grade_probs.float().clamp_min(1e-7)
            branch_loss = (
                -local_safe.gather(1, grade_target.view(-1, 1)).log().mean()
                - global_safe.gather(1, grade_target.view(-1, 1)).log().mean()
            )
        risk_consistency_loss = binary_logits.sum() * 0.0
        if float(self.risk_consistency_weight) > 0 and cumulative_risk is not None:
            aligned_prob = screening_prob.float() if screening_prob is not None else torch.sigmoid(binary_logits.float())
            risk_consistency_loss = F.mse_loss(aligned_prob, cumulative_risk.float())
        risk_ranking_loss = binary_logits.sum() * 0.0
        if float(self.risk_ranking_weight) > 0:
            score = risk_order_score.float() if risk_order_score is not None else (
                screening_prob.float() if screening_prob is not None else torch.sigmoid(binary_logits.float())
            )
            risk_ranking_loss = ordinal_ranking_loss(
                severity_score=score,
                grades=grade_target,
                margin=float(self.rank_margin),
            )
        risk_invariance_loss = binary_logits.sum() * 0.0
        if float(self.risk_invariance_weight) > 0 and clinical_risk_embedding is not None and stability_target_embedding is not None:
            risk_invariance_loss = F.mse_loss(clinical_risk_embedding.float(), stability_target_embedding.float())
        risk_distribution_alignment = binary_logits.sum() * 0.0
        if (
            float(self.risk_distribution_alignment_weight) > 0
            and clinical_risk_embedding is not None
            and transport_risk_embedding is not None
            and cumulative_risk is not None
            and transport_cumulative_risk is not None
        ):
            risk_distribution_alignment = risk_distribution_alignment_loss(
                risk_embedding=clinical_risk_embedding,
                transport_risk_embedding=transport_risk_embedding,
                cumulative_risk=cumulative_risk,
                transport_cumulative_risk=transport_cumulative_risk,
            )
        quantile_risk_calibration = binary_logits.sum() * 0.0
        transport_pair_count = torch.tensor(0.0, device=binary_logits.device)
        if (
            float(self.quantile_risk_calibration_weight) > 0
            and cumulative_risk is not None
            and transport_cumulative_risk is not None
        ):
            quantile_risk_calibration, transport_pair_count = quantile_risk_calibration_loss(
                cumulative_risk=cumulative_risk,
                transport_cumulative_risk=transport_cumulative_risk,
                max_pairs=int(self.transport_pair_sample_limit),
            )
        boundary_aware_loss = binary_logits.sum() * 0.0
        boundary_pair_count = torch.tensor(0.0, device=binary_logits.device)
        if float(self.boundary_aware_weight) > 0 and cumulative_risk is not None:
            from tasks.diabetic_retinopathy.disease_preserving_distillation.grade_boundary_loss import (
                compute_grade_boundary_loss,
            )

            student_risk = screening_prob.float() if screening_prob is not None else torch.sigmoid(binary_logits.float())
            teacher_confidence = (
                grade_probs.float().amax(dim=1)
                if grade_probs is not None
                else torch.ones_like(student_risk, dtype=torch.float32)
            )
            boundary_stats = compute_grade_boundary_loss(
                teacher_risk=cumulative_risk.float(),
                student_risk=student_risk,
                true_grades=grade_target,
                teacher_confidence=teacher_confidence,
                max_pairs=int(self.boundary_pair_limit),
                min_teacher_gap=float(self.boundary_margin),
                min_teacher_confidence=float(self.boundary_min_confidence),
            )
            boundary_aware_loss = boundary_stats.loss
            boundary_pair_count = torch.tensor(float(boundary_stats.pair_count), device=binary_logits.device)
        cargo_group_loss = binary_logits.sum() * 0.0
        if float(self.cargo_group_weight) > 0 and cargo_group_probs is not None:
            cargo_group_loss = compute_grade_conditioned_soft_cvar_loss(
                per_sample_loss=supervised_per_sample_loss,
                grades=grade_target,
                group_probs=cargo_group_probs.float(),
                num_grades=int(self.odic_num_grades),
                alpha=float(self.cargo_cvar_alpha),
            )
        resolved_evidence_diversity_loss = binary_logits.sum() * 0.0
        if float(self.evidence_diversity_weight) > 0 and evidence_diversity_loss is not None:
            resolved_evidence_diversity_loss = evidence_diversity_loss.float()
        progression_direction_loss = binary_logits.sum() * 0.0
        if float(self.progression_direction_weight) > 0 and progression_score is not None:
            progression_direction_loss = ordinal_ranking_loss(
                severity_score=progression_score.float(),
                grades=grade_target,
                margin=float(self.rank_margin),
            )
        odic_consistency_loss = binary_logits.sum() * 0.0
        odic_relation_loss = binary_logits.sum() * 0.0
        if (
            (float(self.odic_consistency_weight) > 0 or float(self.odic_relation_weight) > 0)
            and decision_maps is not None
        ):
            if (
                self._odic_bank is None
                or self._odic_bank.num_grades != int(self.odic_num_grades)
                or self._odic_bank.channels != int(decision_maps.shape[1])
                or self._odic_bank.height != int(decision_maps.shape[2])
                or self._odic_bank.width != int(decision_maps.shape[3])
            ):
                self._odic_bank = RunningGradeDecisionPrototypeBank(
                    num_grades=int(self.odic_num_grades),
                    channels=int(decision_maps.shape[1]),
                    height=int(decision_maps.shape[2]),
                    width=int(decision_maps.shape[3]),
                    momentum=float(self.odic_momentum),
                    device=decision_maps.device,
                    dtype=decision_maps.dtype,
                )
            self._odic_bank.update(decision_maps.detach(), grade_target.detach())
            prototype_targets = self._odic_bank.lookup(grade_target)
            odic_consistency_loss = F.mse_loss(decision_maps.float(), prototype_targets.float())
            odic_relation_loss = self._odic_bank.adjacent_relation_loss()
        loss = (
            float(self.binary_weight) * binary_loss
            + float(self.ordinal_weight) * ordinal_loss
            + float(self.agreement_weight) * agreement_loss
            + float(self.rank_weight) * rank_loss
            + float(self.stability_weight) * stability_loss
            + float(self.prototype_weight) * prototype_loss
            + float(self.prototype_order_weight) * prototype_order_loss
            + float(self.distribution_grade_weight) * distribution_grade_loss
            + float(self.branch_weight) * branch_loss
            + float(self.rps_weight) * rps_loss
            + float(self.risk_consistency_weight) * risk_consistency_loss
            + float(self.risk_ranking_weight) * risk_ranking_loss
            + float(self.risk_invariance_weight) * risk_invariance_loss
            + float(self.risk_distribution_alignment_weight) * risk_distribution_alignment
            + float(self.quantile_risk_calibration_weight) * quantile_risk_calibration
            + float(self.boundary_aware_weight) * boundary_aware_loss
            + float(self.cargo_group_weight) * cargo_group_loss
            + float(self.evidence_diversity_weight) * resolved_evidence_diversity_loss
            + float(self.progression_direction_weight) * progression_direction_loss
            + float(self.odic_consistency_weight) * odic_consistency_loss
            + float(self.odic_relation_weight) * odic_relation_loss
        )
        return {
            "loss": loss,
            "loss_binary": binary_loss.detach(),
            "loss_ordinal": ordinal_loss.detach(),
            "loss_agreement": agreement_loss.detach(),
            "loss_rank": rank_loss.detach(),
            "loss_stability": stability_loss.detach(),
            "loss_prototype": prototype_loss.detach(),
            "loss_prototype_order": prototype_order_loss.detach(),
            "loss_distribution_grade": distribution_grade_loss.detach(),
            "loss_branch": branch_loss.detach(),
            "loss_rps": rps_loss.detach(),
            "loss_risk_consistency": risk_consistency_loss.detach(),
            "loss_risk_ranking": risk_ranking_loss.detach(),
            "loss_risk_invariance": risk_invariance_loss.detach(),
            "loss_risk_distribution_alignment": risk_distribution_alignment.detach(),
            "loss_quantile_risk_calibration": quantile_risk_calibration.detach(),
            "loss_boundary_aware": boundary_aware_loss.detach(),
            "loss_cargo_group": cargo_group_loss.detach(),
            "loss_evidence_diversity": resolved_evidence_diversity_loss.detach(),
            "loss_progression_direction": progression_direction_loss.detach(),
            "loss_odic_consistency": odic_consistency_loss.detach(),
            "loss_odic_relation": odic_relation_loss.detach(),
            "transport_pair_count": transport_pair_count.detach(),
            "boundary_pair_count": boundary_pair_count.detach(),
        }
