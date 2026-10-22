# Frozen P4 loss from evidence commit 6109168c; no scientific changes.
from __future__ import annotations
from typing import Any, Sequence
P4_LOSS_WEIGHTS = {"ord": .20, "style": .10, "orbit": .20, "severity": .10}
C2_LOO_LOSS_WEIGHTS = {}
def p4_loss(
    arm: str,
    output: dict[str, Any],
    grade: Any,
    source: Sequence[str],
    view_output: dict[str, Any] | None,
    *,
    loss_weights_override: dict[str, float] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch
    from torch.nn import functional as F
    for field in (
        "grade_probs", "severity_score", "aux_cumulative_stage3", "aux_cumulative_stage4",
    ):
        value = output.get(field)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite P4 output {field} for {arm}")
    if view_output is not None:
        for field in ("grade_probs", "severity_score"):
            value = view_output.get(field)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite P4 second-view output {field} for {arm}")
    prob = output["grade_probs"].float().clamp_min(1e-7)
    target = grade.long()
    main = -prob[torch.arange(prob.shape[0], device=prob.device), target].log().mean()
    components: dict[str, Any] = {"grade": main}
    counts: dict[str, Any] = {"style_anchors": 0, "style_anchor_by_grade": [0] * 5, "anchors_by_grade": [0] * 5}
    if arm == "C0":
        return main, components, counts
    if arm == "SNR-B3-adapted":
        aux = output["snr_aux"]
        dual = main.new_zeros(())
        for position in ("48", "136"):
            p0 = torch.softmax(aux[f"{position}_logits0"].float(), dim=1)
            pplus = torch.softmax(aux[f"{position}_logits_plus"].float(), dim=1)
            pminus = torch.softmax(aux[f"{position}_logits_minus"].float(), dim=1)
            e0, ep, em = _entropy_from_prob(p0), _entropy_from_prob(pplus), _entropy_from_prob(pminus)
            part = F.softplus(ep - e0) + F.softplus(e0 - em)
            components[f"snr_dual_{position}"] = part
            dual = dual + part
        snr_ce = F.cross_entropy(aux["136_logits_plus"].float(), target)
        components["snr_ce_136_plus"] = snr_ce
        total = main + 0.01 * dual + 0.01 * snr_ce
        return total, components, counts

    ordinal_targets = torch.stack([(target >= k).float() for k in range(1, 5)], dim=1)
    ord_loss = 0.5 * (
        F.binary_cross_entropy(output["aux_cumulative_stage3"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
        + F.binary_cross_entropy(output["aux_cumulative_stage4"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
    )
    parts = []
    for key in ("contrastive_stage3", "contrastive_stage4"):
        pooled = output[key].mean(dim=(-2, -1))
        parts.append(F.normalize(pooled.float(), dim=1))
    embedding = F.normalize(torch.cat(parts, dim=1), dim=1)
    batch = int(grade.shape[0])
    same = target.view(-1, 1).eq(target.view(1, -1))
    other_source = torch.tensor([[source[i] != source[j] for j in range(batch)] for i in range(batch)], device=target.device, dtype=torch.bool)
    diagonal = torch.eye(batch, device=target.device, dtype=torch.bool)
    positive_mask = same & other_source & ~diagonal
    similarity = embedding @ embedding.T
    valid = positive_mask.any(dim=1)
    style = main.new_zeros(())
    if bool(valid.any()):
        positives = similarity[positive_mask]
        negatives = similarity[~diagonal]
        style = (1.0 - positives).mean() + 0.05 * F.relu(negatives.mean() - positives.mean() + 0.1)
    valid_cpu = valid.detach().cpu().numpy()
    grades_cpu = target.detach().cpu().numpy()
    counts["style_anchors"] = int(valid_cpu.sum())
    for k in range(5):
        counts["anchors_by_grade"][k] = int((grades_cpu == k).sum())
        counts["style_anchor_by_grade"][k] = int(((grades_cpu == k) & valid_cpu).sum())
    orbit = main.new_zeros(())
    severity = main.new_zeros(())
    if view_output is not None:
        p = output["grade_probs"].float().clamp_min(1e-7)
        q = view_output["grade_probs"].float().clamp_min(1e-7)
        orbit = 0.5 * (F.kl_div(p.log(), q, reduction="batchmean") + F.kl_div(q.log(), p, reduction="batchmean"))
        severity = (output["severity_score"].float() - view_output["severity_score"].float()).abs().mean()
    components.update({"ord": ord_loss, "style": style, "orbit": orbit, "severity": severity})
    weights = loss_weights_override or C2_LOO_LOSS_WEIGHTS.get(arm, P4_LOSS_WEIGHTS)
    for name, value in components.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite P4 loss component {name} for {arm}")
    total = main + weights["ord"] * ord_loss + weights["style"] * style + weights["orbit"] * orbit + weights["severity"] * severity
    return total, components, counts
