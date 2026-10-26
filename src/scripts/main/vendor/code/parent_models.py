"""Six explicit native feature boundaries and intact historical C2 consumers."""
from common import *
from assets import parents,public
p4=backend()
import torch,timm,contextlib
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from numerics import gaussian_logp
WIDTH={'B3':136,'ResNet50':1024,'DenseNet121':1024,'SwinTiny':384,'ConvNeXtTiny':384,'InceptionV3':768}
FACTORIES={'B3':'efficientnet_b3','ResNet50':'resnet50','DenseNet121':'densenet121','SwinTiny':'swin_tiny_patch4_window7_224','ConvNeXtTiny':'convnext_tiny.fb_in1k','InceptionV3':'inception_v3.tf_in1k'}
def canonical_c2_state(state):
 """Remove only the owned serialization wrapper, never backbone/model names."""
 if not state:raise RuntimeError('empty C2 parent state')
 wrapped=[k.startswith('inner.') for k in state]
 if any(wrapped) and not all(wrapped):raise RuntimeError('mixed C2 serialization namespace')
 out={k.removeprefix('inner.'):v for k,v in state.items()}
 if not any(k.startswith('base.feature_pyramid.backbone.') for k in out):raise RuntimeError('missing consumed C2 backbone namespace')
 return out
def native(b):
 kw={'img_size':512} if b=='SwinTiny' else {'aux_logits':False} if b=='InceptionV3' else {}
 m=timm.create_model(FACTORIES[b],pretrained=False,num_classes=1000,**kw)
 sd=load_file(public()[b]['path'])
 if b=='SwinTiny':sd={k:v for k,v in sd.items() if 'relative_position_index' not in k and 'attn_mask' not in k}
 m.load_state_dict(sd,strict=True);m.reset_classifier(5)
 return m
def split_prefix(b,m,x):
 if b=='B3':
  x=m.bn1(m.conv_stem(x))
  for block in list(m.blocks)[:3]:x=block(x)
 elif b=='ResNet50':x=m.layer2(m.layer1(m.maxpool(m.act1(m.bn1(m.conv1(x))))))
 elif b=='DenseNet121':
  for name,m0 in m.features.named_children():
   x=m0(x)
   if name=='transition2':break
 elif b=='SwinTiny':x=m.layers[1](m.layers[0](m.patch_embed(x)))
 elif b=='ConvNeXtTiny':x=m.stages[1](m.stages[0](m.stem(x)))
 else:
  for name in ('Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Pool1','Conv2d_3b_1x1','Conv2d_4a_3x3','Pool2','Mixed_5b','Mixed_5c','Mixed_5d'):x=getattr(m,name)(x)
 return x
def split_middle(b,m,x):
 if b=='B3':return m.blocks[4](m.blocks[3](x))
 if b=='ResNet50':return m.layer3(x)
 if b=='DenseNet121':return m.features.denseblock3(x)
 if b=='SwinTiny':return m.layers[2](x)
 if b=='ConvNeXtTiny':return m.stages[2](x)
 for name in ('Mixed_6a','Mixed_6b','Mixed_6c','Mixed_6d','Mixed_6e'):x=getattr(m,name)(x)
 return x
def split_tail(b,m,x):
 if b=='B3':
  for block in list(m.blocks)[5:]:x=block(x)
  return m.bn2(m.conv_head(x))
 if b=='ResNet50':return m.layer4(x)
 if b=='DenseNet121':return m.features.norm5(m.features.denseblock4(m.features.transition3(x)))
 if b=='SwinTiny':return m.norm(m.layers[3](x))
 if b=='ConvNeXtTiny':return m.norm_pre(m.stages[3](x))
 for name in ('Mixed_7a','Mixed_7b','Mixed_7c'):x=getattr(m,name)(x)
 return x
def suffix_modules(b,m,full):
 if b=='B3':return list(m.blocks)[3:5] if full else list(m.blocks)[3:]+[m.conv_head,m.bn2,m.global_pool]
 if b=='ResNet50':return [m.layer3,m.layer4]
 if b=='DenseNet121':return [m.features.denseblock3,m.features.transition3,m.features.denseblock4,m.features.norm5]
 if b=='SwinTiny':return [m.layers[2],m.layers[3],m.norm]
 if b=='ConvNeXtTiny':return [m.stages[2],m.stages[3],m.norm_pre] + ([] if full else [m.head])
 return [getattr(m,n) for n in ('Mixed_6a','Mixed_6b','Mixed_6c','Mixed_6d','Mixed_6e','Mixed_7a','Mixed_7b','Mixed_7c')]
def build_full(b,s):
 if b=='B3':m,_=p4.build_p4_model('C2',s,weight_path=Path(public()['B3']['path']));return m.inner.cpu()
 old=s2_backend()
 if b in ('ResNet50','DenseNet121','SwinTiny'):m,_=old.build(NAMES[b],'C2',s);return m
 old.seed_all(s)
 class Pyramid(nn.Module):
  def __init__(self):
   super().__init__();self.backbone=native(b);self.backbone.reset_classifier(0)
  def forward(self,x):
   bb=self.backbone;x=split_prefix(b,bb,x);s3=split_middle(b,bb,x);s4=split_tail(b,bb,s3);return s3,s4
 pyramid=Pyramid();c3,c4=(384,768) if b=='ConvNeXtTiny' else (768,2048)
 from unittest.mock import patch
 with patch.object(old.dpm,'_build_feature_pyramid',return_value=(pyramid,c3,c4)):
  base=old.ClinicalRiskRepresentationLearningModel(pretrained=False,backbone_name=NAMES[b],backbone_img_size=512,pooling='gem',local_query_count=6,feature_dim=256,attention_heads=4,mode='local_global',beta=1.,crrl_mode='consistency',risk_hidden_dim=256,risk_embedding_dim=256)
 return old.Arm(base,'C2',(c3,c4))
class Model(nn.Module):
 def __init__(self,b,arm,s,parent_new=False):
  super().__init__();self.b=b;self.arm=arm;self.seed=s;self.phase=0;self.parent_new=parent_new
  self.full=arm in ('P','C2_SUM','GAUSS_G','C2');self.inner=None;self.native=None;self.initialization=[]
  state=None
  if not parent_new:
   pp,rr=parents(b,s,'normal' if arm=='Nplus' else 'C2');ck=torch.load(pp,map_location='cpu',weights_only=False);state=ck.get('model',ck)
   if arm!='Nplus':state=canonical_c2_state(state)
   recovery=torch.load(rr,map_location='cpu',weights_only=False)
   pos=recovery.get('next_update',recovery.get('local_update'))
   if pos!=15001:raise RuntimeError('nonterminal parent '+str(pp))
   self.parent_rng=recovery['rng']
  else:p4._set_seed(s);self.parent_rng=None
  if self.full:
   self.inner=build_full(b,s)
   if state is not None:self.inner.load_state_dict(state,strict=True)
  else:
   self.native=native(b)
   if state is not None:
    if arm=='Nplus':self.native.load_state_dict({k.removeprefix('native.'):v for k,v in state.items()},strict=True)
    else:
     sd=self.native.state_dict();prefix='base.feature_pyramid.backbone.'
     for k,v in sd.items():
      consumed=(k.startswith(('conv_stem.','bn1.')) or any(k.startswith(f'blocks.{i}.') for i in range(5))) if b=='B3' else (prefix+k in state and not (b=='ConvNeXtTiny' and k.startswith('head.')))
      if consumed:sd[k]=state[prefix+k]
      self.initialization.append({'name':'native.'+k,'origin':'same-seed consumed C2' if consumed else 'public unconsumed tail','sha256':p4._tensor_sha256(sd[k])})
     self.native.load_state_dict(sd,strict=True)
     g=torch.Generator().manual_seed(s+170000);head=self.native.get_classifier()
     with torch.no_grad():head.weight.copy_(torch.randn(head.weight.shape,generator=g)*.01);head.bias.zero_()
   if arm.startswith(('T','Z')):
    from tasks.diabetic_retinopathy.sparrow_dr.destyle import ResidualDestylizer
    self.d=ResidualDestylizer(WIDTH[b]);key='destylize_stage4.' if b=='B3' else 'destylize_stage3.'
    self.d.load_state_dict({k.removeprefix(key):v for k,v in state.items() if k.startswith(key)},strict=True)
    if arm=='T_IN':
     with torch.no_grad():self.d.gamma.fill_(1);self.d.beta.zero_()
    if arm.startswith('Z'):
     self.gate=nn.Parameter(torch.zeros(1 if arm=='Z_SCALAR' else WIDTH[b]),requires_grad=False)
     if arm=='Z_HALF':self.gate.data.fill_(.5)
     self.register_buffer('gamma0',self.d.gamma.detach().clone());self.register_buffer('beta0',self.d.beta.detach().clone())
  self.mix_cpu=torch.Generator().manual_seed(s+970000)
  self.aux_only=set();self.legal_aux_none=set()
  # Historical full-C2 bypasses the native classifier normalization and the
  # embedding output. Parent AdamW skips their None grads exactly as before.
  self.parent_legal_none={n for n,p in self.named_parameters() if self.parent_new and self.full and (n.startswith('inner.base.embedding_project.') or n.startswith('inner.base.feature_pyramid.backbone.head.'))}
 @property
 def backbone(self):return self.inner.base.feature_pyramid.backbone if self.full else self.native
 def configure(self,phase):
  self.phase=phase;self.eval()
  for p in self.parameters():p.requires_grad_(False)
  if self.parent_new:
   self.train()
   for p in self.parameters():p.requires_grad_(True)
   if self.full:
    for name,p in self.named_parameters():
     if any(t in name for t in ('risk_','screening_','severity_embedding')):p.requires_grad_(False)
   groups=[{'params':[p for p in self.parameters() if p.requires_grad],'lr':3e-5,'weight_decay':1e-4}]
  else:
   active=([self.inner.base.local_head,self.inner.base.global_head,self.inner.base.fusion,self.inner.axis_stage3,self.inner.axis_stage4] if self.full else [self.native.get_classifier()])
   if phase==2:
    active+=suffix_modules(self.b,self.backbone,self.full)
    if self.full:active += [self.inner.destylize_stage3,self.inner.destylize_stage4,self.inner.base.local_branch,self.inner.base.local_project,self.inner.base.global_branch]
   for m in active:
    m.train()
    for p in m.parameters():p.requires_grad_(True)
   if phase==2 and self.arm.startswith('Z'):
    self.gate.requires_grad_(self.arm!='Z_HALF')
    if self.arm=='Z_JOINT':self.d.gamma.requires_grad_(True);self.d.beta.requires_grad_(True)
   classifier=set() if self.full else {id(p) for p in self.native.get_classifier().parameters()}
   backbone={id(p) for p in self.backbone.parameters()}-classifier
   grouped={}
   for name,p in self.named_parameters():
    if not p.requires_grad:continue
    lr=1e-4 if phase==1 else 1e-5 if id(p) in backbone else 3e-5;wd=1e-4
    if name=='gate':lr=3e-4;wd=0
    if self.arm=='Z_JOINT' and name in ('d.gamma','d.beta'):lr=1e-5;wd=0
    grouped.setdefault((lr,wd),[]).append(p)
   groups=[{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in grouped.items()]
  self.aux_only={n for n,p in self.named_parameters() if p.requires_grad and ('axis_stage3.' in n or 'axis_stage4.' in n)}
  flat=[p for g in groups for p in g['params']]
  assert len(flat)==len({id(p) for p in flat})==sum(p.requires_grad for p in self.parameters())
  self.lock={'backbone':self.b,'arm':self.arm,'phase':phase,'groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':[{'name':n,'shape':list(p.shape),'object_id':id(p),'consumer':'declared_unused_parent' if n in self.parent_legal_none else 'aux_only' if n in self.aux_only else 'primary_shared'} for n,p in self.named_parameters() if any(p is q for q in g['params'])]} for g in groups],'BN':{n:m.training for n,m in self.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)},'aux_only':sorted(self.aux_only),'legal_aux_none':sorted(self.legal_aux_none),'parent_legal_none':sorted(self.parent_legal_none)}
  return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
 def transform(self,x,aux):
  nhwc=self.b=='SwinTiny';f=x.permute(0,3,1,2) if nhwc else x
  if self.arm=='T_MS' and aux:
   # Isolated CPU stream: Bernoulli, Beta via two Gamma draws, random donor.
   if float(torch.rand((),generator=self.mix_cpu))<.5:
    mean=f.mean((-2,-1),keepdim=True).detach();std=(f.var((-2,-1),keepdim=True,correction=1)+1e-6).sqrt().detach()
    shape=torch.full((len(f),1,1,1),.1);u=torch._standard_gamma(shape,generator=self.mix_cpu);v=torch._standard_gamma(shape,generator=self.mix_cpu);lam=(u/(u+v)).to(f.device)
    donor=torch.randperm(len(f),generator=self.mix_cpu).to(f.device);f=(f-mean)/std*(lam*std+(1-lam)*std[donor])+lam*mean+(1-lam)*mean[donor]
  elif self.arm in ('T','T_IN') and aux:f=self.d(f)
  elif self.arm.startswith('Z'):f=f+self.gate.view(1,-1,1,1)*(self.d(f)-f)
  return f.permute(0,2,3,1).contiguous() if nhwc else f
 def forward(self,x,aux=False):
  if self.b=='InceptionV3':x=(p4._denormalize(x)-.5)/.5
  b=self.backbone
  ctx=contextlib.nullcontext() if self.parent_new else torch.no_grad()
  with ctx:x=split_prefix(self.b,b,x);prefix=x
  ctx=torch.no_grad() if self.phase==1 and not self.parent_new else contextlib.nullcontext()
  with ctx:
   middle=split_middle(self.b,b,x)
   if self.full:
    s3=prefix if self.b=='B3' else middle
    s4=middle if self.b=='B3' else split_tail(self.b,b,middle)
    if self.b=='SwinTiny':s3=s3.permute(0,3,1,2).contiguous();s4=s4.permute(0,3,1,2).contiguous()
    base=self.inner.base;local=base.local_branch(self.inner.destylize_stage3(s3));zl=base.local_project(local['local_embedding_raw']);zg=base.global_branch(self.inner.destylize_stage4(s4))
   else:
    if self.phase==2 or self.phase==0:middle=self.transform(middle,aux)
    value=split_tail(self.b,b,middle)
    if self.b=='B3':value=b.global_pool(value)
  if not self.full:
   if self.b=='B3':
    if b.drop_rate>0:value=F.dropout(value,p=b.drop_rate,training=b.classifier.training)
    logits=b.classifier(value)
   elif self.b=='DenseNet121':logits=b.forward_head(F.relu(value,inplace=False))
   else:logits=b.forward_head(value)
   return {'logp':torch.log_softmax(logits.double(),1)}
  lm,ls=base.local_head(zl);gm,gs=base.global_head(zg);fusion=base.fusion(local_mu=lm,local_sigma=ls,global_mu=gm,global_sigma=gs)
  stable=gaussian_logp(fusion.mean,fusion.std)
  out={'logp':stable,'mu':fusion.mean,'sigma':fusion.std,'aux_cumulative_stage3':self.inner.axis_stage3.cumulative(s3),'aux_cumulative_stage4':self.inner.axis_stage4.cumulative(s4),'contrastive_stage3':s3,'contrastive_stage4':s4}
  if self.parent_new:
   raw=base._probs(fusion.mean,fusion.std)
   out.update(logp=raw.double().log(),historical_grade_probs=raw,stable_logp=stable)
  return out
 def project_gate(self):
  if hasattr(self,'gate') and self.gate.requires_grad:
   with torch.no_grad():self.gate.clamp_(0,1)
 def restore_parent_rng(self):
  if self.parent_rng is None:p4._set_seed(self.seed);return
  r=self.parent_rng
  if 'cpu' in r:r={'python':r['python'],'numpy':r['numpy'],'torch_cpu':r['cpu'],'torch_cuda':r['cuda']}
  p4._restore_rng_state(r)
