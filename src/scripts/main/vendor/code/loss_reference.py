"""Forward-loss specification only. No models, data loading, optimizer or training loop.
Not imported/executed against project data in this design review.
The project runner must supply stable log probabilities and original bound raw losses.
"""
from typing import Optional, Tuple, Dict
import torch


def _validated_logp(logp: torch.Tensor, name: str) -> torch.Tensor:
    if logp.ndim != 2 or logp.shape[1] != 5:
        raise ValueError(f"{name} must have shape [batch, 5]")
    z = logp.to(torch.float64)
    if not torch.isfinite(z).all():
        raise ValueError(f"{name}: nonfinite log probabilities; do not floor or silently repair")
    if not torch.allclose(torch.logsumexp(z, dim=1), torch.zeros(z.shape[0], device=z.device, dtype=z.dtype), atol=1e-9, rtol=0):
        raise ValueError(f"{name}: not a normalized log distribution")
    return z


def loss_parts(
    arm: str,
    clean_logp: torch.Tensor,
    gamma_logp: torch.Tensor,
    grade: torch.Tensor,
    clean_mu: torch.Tensor,
    gamma_mu: torch.Tensor,
    raw_ordinal_loss: torch.Tensor,
    raw_style_loss: torch.Tensor,
    parent_clean_logp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return (main_loss, auxiliary_loss, log_terms).

    phase1: backward(main + auxiliary).
    phase2: two autograd.grad calls on the SAME student graph; sum, no projection.
    Parent inference is done in eval/no_grad upstream. This function detaches again.
    V_OFF/VSUP still receive and log the auxiliary view; BN exposure is runner-owned.
    """
    if arm not in {"C2_SUM", "V_OFF", "VSUP", "LS20", "PREF"}:
        raise ValueError(f"Unknown registered arm: {arm}")
    lc = _validated_logp(clean_logp, "clean")
    la = _validated_logp(gamma_logp, "gamma")
    if lc.shape != la.shape or grade.shape != (lc.shape[0],):
        raise ValueError("Batch/grade shape mismatch")
    if grade.dtype != torch.long or not bool(((grade >= 0) & (grade < 5)).all()):
        raise ValueError("grade must be torch.long labels 0..4")
    nc = -lc.gather(1, grade[:, None]).mean()
    na = -la.gather(1, grade[:, None]).mean()
    sym_kl = 0.5 * ((lc.exp() * (lc-la)).sum(1).mean() + (la.exp() * (la-lc)).sum(1).mean())
    mu_cons = (clean_mu.to(torch.float64)-gamma_mu.to(torch.float64)).abs().mean()
    os_loss = 0.2*raw_ordinal_loss.to(torch.float64) + 0.1*raw_style_loss.to(torch.float64)
    view_loss = 0.2*sym_kl + 0.1*mu_cons
    parent_ce = lc.new_zeros(())
    uniform_ce = -lc.mean(1).mean()
    if arm == "VSUP":
        main, aux = 0.5*(nc+na), os_loss + 0.0*view_loss
    elif arm == "V_OFF":
        main, aux = nc, os_loss + 0.0*view_loss
    elif arm == "LS20":
        main, aux = 0.8*nc+0.2*uniform_ce, os_loss+view_loss
    elif arm == "PREF":
        if parent_clean_logp is None:
            raise ValueError("PREF requires same-seed frozen parent output on this exact clean tensor")
        lt = _validated_logp(parent_clean_logp, "parent").detach()
        if lt.shape != lc.shape:
            raise ValueError("Parent/student batch mismatch")
        parent_ce = -(lt.exp()*lc).sum(1).mean()
        main, aux = 0.8*nc+0.2*parent_ce, os_loss+view_loss
    else:
        main, aux = nc, os_loss+view_loss
    return main, aux, {"nll_clean":nc, "nll_gamma":na, "raw_os":os_loss,
                       "view_consistency":view_loss, "sym_kl":sym_kl,
                       "mu_consistency":mu_cons, "parent_ce":parent_ce,
                       "uniform_ce":uniform_ce}
