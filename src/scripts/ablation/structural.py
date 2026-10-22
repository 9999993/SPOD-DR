"""A-owned structural ablations of the frozen bf47a967 SPOD-DR graph.

API: apply(public_initialized_model, 'NO_DESTYLE' | 'NO_LOCAL').
Call before configure/optimizer construction, in parent and O reconstruction.
No trained initialization, substitute branch or change to the other variant.
"""
import contextlib
import inspect
import torch
from torch import nn
import parent_models as pm
import models as cm
from numerics import gaussian_logp

API_VERSION = 'SPOD_ABLATION_STRUCTURAL_R1'


def configure_global(self, phase):
    """Original configure, retaining exactly the surviving global parameters."""
    self.phase = phase
    self.eval()
    for p in self.parameters():
        p.requires_grad_(False)
    if self.parent_new:
        self.train()
        for name, p in self.named_parameters():
            p.requires_grad_(not any(t in name for t in ('risk_', 'screening_', 'severity_embedding')))
        groups = [{'params': [p for p in self.parameters() if p.requires_grad], 'lr': 3e-5, 'weight_decay': 1e-4}]
    else:
        active = [self.inner.base.global_head, self.inner.axis_stage4]
        if phase == 2:
            active += cm.suffix_modules(self.b, self.backbone, True)
            active += [self.inner.destylize_stage4, self.inner.base.global_branch]
        for module in active:
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)
        backbone = {id(p) for p in self.backbone.parameters()}
        grouped = {}
        for _, p in self.named_parameters():
            if p.requires_grad:
                lr = 1e-4 if phase == 1 else 1e-5 if id(p) in backbone else 3e-5
                grouped.setdefault((lr, 1e-4), []).append(p)
        groups = [{'params': ps, 'lr': lr, 'weight_decay': wd} for (lr, wd), ps in grouped.items()]
    self.aux_only = {n for n, p in self.named_parameters() if p.requires_grad and 'axis_stage4.' in n}
    self.legal_aux_none = set()
    flat = [p for g in groups for p in g['params']]
    assert len(flat) == len({id(p) for p in flat}) == sum(p.requires_grad for p in self.parameters())
    self.lock = {'backbone': self.b, 'arm': self.arm, 'phase': phase,
                 'groups': [{'lr': g['lr'], 'weight_decay': g['weight_decay'],
                             'parameters': [n for n, p in self.named_parameters() if any(p is q for q in g['params'])]} for g in groups],
                 'BN': {n: m.training for n, m in self.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)},
                 'aux_only': sorted(self.aux_only), 'legal_aux_none': [],
                 'parent_legal_none': sorted(self.parent_legal_none)}
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def forward_global(self, x, aux=False):
    """Same feature boundaries, global destyle/GeM/Gaussian, direct decode."""
    bb = self.backbone
    with contextlib.nullcontext() if self.parent_new else torch.no_grad():
        if self.b == 'B0':
            for i in range(4):
                x = bb[i](x)
        else:
            x = pm.split_prefix(self.b, bb, x)
        prefix = x
    with torch.no_grad() if self.phase == 1 and not self.parent_new else contextlib.nullcontext():
        if self.b == 'B0':
            s4 = bb[5](bb[4](prefix))
        else:
            middle = pm.split_middle(self.b, bb, prefix)
            s4 = middle if self.b == 'B3' else pm.split_tail(self.b, bb, middle)
            if self.b == 'SwinTiny':
                s4 = s4.permute(0, 3, 1, 2).contiguous()
        base = self.inner.base
        zg = base.global_branch(self.inner.destylize_stage4(s4))
    mu, sigma = base.global_head(zg)
    stable = gaussian_logp(mu, sigma)
    out = {'logp': stable, 'mu': mu, 'sigma': sigma,
           'aux_cumulative_stage4': self.inner.axis_stage4.cumulative(s4),
           'contrastive_stage4': s4, '_spod_no_local': True}
    if self.parent_new:
        raw = base._probs(mu, sigma)
        out.update(logp=raw.double().log(), historical_grade_probs=raw, stable_logp=stable)
    return out


def install_routes():
    if getattr(pm.Model, '_spod_structural_installed', False):
        return
    # Keep all historical loss arithmetic except the deleted local consumers.
    # Ordinal was a branch mean; the remaining mean is the global term.
    # Style was concatenation of normalized branch pools; use its global pool.
    p4 = pm.p4
    original_loss = p4.p4_loss
    source = inspect.getsource(original_loss)
    old_ord = '''ord_loss = 0.5 * (
        F.binary_cross_entropy(output["aux_cumulative_stage3"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
        + F.binary_cross_entropy(output["aux_cumulative_stage4"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
    )'''
    assert old_ord in source, 'frozen loss changed'
    source = source.replace(old_ord, 'ord_loss = F.binary_cross_entropy(output["aux_cumulative_stage4"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)')
    source = source.replace('("contrastive_stage3", "contrastive_stage4")', '("contrastive_stage4",)')
    namespace = dict(original_loss.__globals__)
    exec(compile(source, __file__ + ':global_only_p4_loss', 'exec'), namespace)
    global_loss = namespace['p4_loss']
    def routed_loss(arm, output, grade, source, view_output, **kwargs):
        fn = global_loss if output.get('_spod_no_local', False) else original_loss
        return fn(arm, output, grade, source, view_output, **kwargs)
    p4.p4_loss = routed_loss
    for cls in (pm.Model, cm.ReferenceModel):
        original_forward, original_configure = cls.forward, cls.configure
        def forward(self, x, aux=False, original=original_forward):
            return forward_global(self, x, aux) if getattr(self, 'spod_variant', None) == 'NO_LOCAL' else original(self, x, aux)
        def configure(self, phase, original=original_configure):
            return configure_global(self, phase) if getattr(self, 'spod_variant', None) == 'NO_LOCAL' else original(self, phase)
        cls.forward, cls.configure = forward, configure
    import b0
    original_b0 = b0.forward_b0
    def routed_b0(self, x, aux=False):
        return forward_global(self, x, aux) if getattr(self, 'spod_variant', None) == 'NO_LOCAL' else original_b0(self, x, aux)
    b0.forward_b0 = routed_b0
    pm.Model._spod_structural_installed = True


def apply(model, variant):
    assert model.parent_new and model.full and variant in ('NO_DESTYLE', 'NO_LOCAL')
    install_routes()
    model.spod_variant = variant
    # The historical B3 full graph stops at blocks[4]. Its public factory
    # retains the unused later ImageNet tail; AdamW already skips None grads.
    # Declare that existing disconnection for the parent gradient receipt only.
    if model.b == 'B3':
        prefix = 'inner.base.feature_pyramid.backbone.'
        tail = ('blocks.5.', 'blocks.6.', 'conv_head.', 'bn2.', 'classifier.')
        model.parent_legal_none |= {n for n, _ in model.named_parameters()
                                    if n.startswith(tuple(prefix+t for t in tail))}
    if variant == 'NO_DESTYLE':
        model.inner.destylize_stage3 = nn.Identity()
        model.inner.destylize_stage4 = nn.Identity()
    else:
        for name in ('local_branch', 'local_project', 'local_head', 'fusion'):
            delattr(model.inner.base, name)
        for name in ('destylize_stage3', 'axis_stage3'):
            delattr(model.inner, name)
        assert not any(any(t in n for t in ('local_branch.', 'local_project.', 'local_head.', 'fusion.', 'axis_stage3.', 'destylize_stage3.')) for n, _ in model.named_parameters())
    return model
