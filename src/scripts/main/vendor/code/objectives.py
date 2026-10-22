"""C22 changes only the registered clean main consumer; common auxiliaries fixed."""
import torch
from loss_reference import loss_parts

def raw_nll(model, mu, sigma, y):
    p = model.inner.base._probs(mu.float(), sigma.float())
    return -p.float().clamp_min(1e-7)[torch.arange(len(y), device=y.device), y].log().mean()

def losses(model, a, b, y, sources, p4):
    aa = {**a, 'grade_probs': a['logp'].exp(), 'severity_score': a['mu']}
    bb = {**b, 'grade_probs': b['logp'].exp(), 'severity_score': b['mu']}
    _, raw, detail = p4.p4_loss('C2', aa, y, sources, bb)
    main, aux, parts = loss_parts('C2_SUM', a['logp'], b['logp'], y,
                                a['mu'], b['mu'], raw['ord'], raw['style'])
    if model.arm in ('RAW_N', 'PREFIX_RAW_N'):
        main = raw_nll(model, a['mu'], a['sigma'], y)
    model.last_exposure=detail
    parts.update(ord=raw['ord'], style=raw['style'], main=main)
    return main, aux, parts
