from runtime import *
from torch import nn
from numerics import gaussian_logp
from training_resources import initialization_for
def public_b0():
    from torchvision.models import efficientnet_b0
    entry=initialization_for('B0');weight=Path(entry['path'])
    if sha(weight)!=entry['sha256']:raise ValueError('B0 public initialization SHA256 mismatch')
    net=efficientnet_b0(weights=None);net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);return net
class B0Pyramid(nn.Module):
    def __init__(self):
        super().__init__();net=public_b0();self.backbone=nn.Sequential(*list(net.features.children())[:6])
    def forward(self,x):
        s3=None
        for i,m in enumerate(self.backbone):
            x=m(x)
            if i==3:s3=x
        assert s3.shape[1]==40 and x.shape[1]==112
        return s3,x
def build_full(seed):
    from unittest.mock import patch
    old=pm.s2_backend();old.seed_all(seed);pyramid=B0Pyramid()
    with patch.object(old.dpm,'_build_feature_pyramid',return_value=(pyramid,40,112)):
        base=old.ClinicalRiskRepresentationLearningModel(pretrained=False,backbone_name='efficientnet_b0',backbone_img_size=512,pooling='gem',local_query_count=6,feature_dim=256,attention_heads=4,mode='local_global',beta=1.,crrl_mode='consistency',risk_hidden_dim=256,risk_embedding_dim=256)
    return old.Arm(base,'C2',(40,112))
def native_b0():
    m=public_b0();m.classifier[1]=nn.Linear(m.classifier[1].in_features,5);return m
_native=pm.native;_build=pm.build_full;_suffix=cm.suffix_modules
pm.native=lambda b:native_b0() if b=='B0' else _native(b)
pm.build_full=lambda b,s:build_full(s) if b=='B0' else _build(b,s)
cm.suffix_modules=lambda b,m,full:[m[4],m[5]] if b=='B0' and full else _suffix(b,m,full)
def forward_b0(m,x,aux=False):
    assert x.shape[-2:]==(512,512)
    if not m.full:
        logits=m.native(x);return {'logp':torch.log_softmax(logits.double(),1),'logits':logits}
    import contextlib
    bb=m.backbone
    with contextlib.nullcontext() if m.parent_new else torch.no_grad():
        for i in range(4):x=bb[i](x)
        s3=x
    with torch.no_grad() if m.phase==1 and not m.parent_new else contextlib.nullcontext():
        s4=bb[5](bb[4](s3));base=m.inner.base
        local=base.local_branch(m.inner.destylize_stage3(s3));zl=base.local_project(local['local_embedding_raw']);zg=base.global_branch(m.inner.destylize_stage4(s4))
    lm,ls=base.local_head(zl);gm,gs=base.global_head(zg);fusion=base.fusion(local_mu=lm,local_sigma=ls,global_mu=gm,global_sigma=gs)
    stable=gaussian_logp(fusion.mean,fusion.std)
    out={'logp':stable,'mu':fusion.mean,'sigma':fusion.std,'aux_cumulative_stage3':m.inner.axis_stage3.cumulative(s3),'aux_cumulative_stage4':m.inner.axis_stage4.cumulative(s4),'contrastive_stage3':s3,'contrastive_stage4':s4}
    if m.parent_new:
        raw=base._probs(fusion.mean,fusion.std);out.update(logp=raw.double().log(),historical_grade_probs=raw,stable_logp=stable)
    return out
