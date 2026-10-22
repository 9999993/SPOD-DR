"""Fixed FP64 block Shampoo direction, per-tensor Adam-length grafting."""
import time,torch

def root4(S,t):
 S=S/(1-.999**t);S=(S+S.transpose(-2,-1))*.5
 d=S.shape[-1];delta=(1e-4*S.diagonal(dim1=-2,dim2=-1).sum(-1)/d).clamp_min(1e-12)
 lam,Q=torch.linalg.eigh(S)
 if (lam[...,0]<-1e-8*lam[...,-1].clamp_min(1e-12)).any():raise FloatingPointError('material negative Shampoo spectrum')
 shifted=lam.clamp_min(0)+delta[...,None]
 root=(Q*shifted.pow(-.25).unsqueeze(-2))@Q.transpose(-2,-1)
 if not torch.isfinite(root).all():raise FloatingPointError('nonfinite Shampoo root')
 # Spectral residual on the clamped+damped matrix, independent of its scale.
 root2=root@root;root4value=root2@root2
 regularized=(Q*shifted.unsqueeze(-2))@Q.transpose(-2,-1)
 identity=torch.eye(d,device=S.device,dtype=torch.float64)
 residual=(regularized@root4value-identity).norm(dim=(-2,-1))/d**.5
 residual=residual.max()
 return root,dict(delta_min=float(delta.min()),delta_max=float(delta.max()),inverse_residual=float(residual))

def graft(z,adam):
 an=adam.double().norm();zn=z.double().norm()
 if zn==0:
  if an!=0:raise FloatingPointError('zero Shampoo direction with nonzero Adam')
  return torch.zeros_like(z),0.
 ratio=an/zn
 return z*ratio,float(ratio)

class AdamGraftedShampoo(torch.optim.Optimizer):
 def __init__(self,groups):
  super().__init__(groups,dict(lr=3e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8))
  self.last_audit=[]
 @torch.no_grad()
 def step(self,closure=None):
  if closure is not None:raise ValueError('closure unsupported')
  self.last_audit=[]
  for gi,group in enumerate(self.param_groups):
   for pi,p in enumerate(group['params']):
    if p.grad is None:continue
    G=p.grad;st=self.state[p]
    if not st:
     st.update(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
     if p.ndim>=2:
      shape=(p.shape[0],p.numel()//p.shape[0]);layout={}
      for i in range(0,shape[0],128):
       for j in range(0,shape[1],128):layout.setdefault((min(128,shape[0]-i),min(128,shape[1]-j)),[]).append((i,j))
      st['blocks']=[]
      for (a,b),ij in layout.items():st['blocks'].append(dict(ij=ij,L=G.new_zeros((len(ij),a,a),dtype=torch.float64),R=G.new_zeros((len(ij),b,b),dtype=torch.float64),rootL=None,rootR=None,last_refresh=0))
    st['step']+=1;t=st['step'];st['m'].mul_(.9).add_(G,alpha=.1);st['v'].mul_(.999).addcmul_(G,G,value=.001)
    mhat=st['m']/(1-.9**t);vhat=st['v']/(1-.999**t);adam=mhat/(vhat.sqrt()+1e-8)
    audit=dict(group=gi,parameter_index=pi,step=t,refreshes=0,root_seconds=0.,inverse_residual=0.)
    if p.ndim<2:D=adam;ratio=1.
    else:
     gm=G.reshape(p.shape[0],-1);mm=mhat.reshape_as(gm);Z=torch.empty_like(gm,dtype=torch.float64)
     for block in st['blocks']:
      a,b=block['L'].shape[-1],block['R'].shape[-1];ij=block['ij']
      gb=torch.stack([gm[i:i+a,j:j+b] for i,j in ij]).double();mb=torch.stack([mm[i:i+a,j:j+b] for i,j in ij]).double()
      block['L'].mul_(.999).add_(gb@gb.transpose(-2,-1),alpha=.001)
      block['R'].mul_(.999).add_(gb.transpose(-2,-1)@gb,alpha=.001)
      if t==1 or t%10==0:
       start=time.monotonic();block['rootL'],left=root4(block['L'],t);block['rootR'],right=root4(block['R'],t);block['last_refresh']=t
       audit['root_seconds']+=time.monotonic()-start;audit['refreshes']+=2*len(ij);audit['inverse_residual']=max(audit['inverse_residual'],left['inverse_residual'],right['inverse_residual'])
       audit['delta_min']=min(audit.get('delta_min',float('inf')),left['delta_min'],right['delta_min']);audit['delta_max']=max(audit.get('delta_max',0),left['delta_max'],right['delta_max'])
      zb=block['rootL']@mb@block['rootR']
      for k,(i,j) in enumerate(ij):Z[i:i+a,j:j+b]=zb[k]
     D,ratio=graft(Z.reshape_as(p),adam);D=D.to(p.dtype)
     if not torch.allclose(D.double().norm(),adam.double().norm(),rtol=1e-6,atol=1e-8):raise FloatingPointError('graft norm mismatch')
     if t==1 or t%100==0:
      audit.update(Z_norm=float(Z.norm()),graft_ratio=ratio,Adam_norm=float(adam.double().norm()),D_norm=float(D.double().norm()),direction_cos=float((D.double()*adam.double()).sum()/(D.double().norm()*adam.double().norm())) if D.norm()>0 else None)
    p.mul_(1-group['lr']*group['weight_decay']);p.add_(D,alpha=-group['lr'])
    self.last_audit.append(audit)
 def load_state_dict(self,state_dict):
  # PyTorch's default loader casts nested FP64 block matrices to parameter FP32.
  saved=state_dict['state'];super().load_state_dict(state_dict)
  for group,ids in zip(self.param_groups,state_dict['param_groups']):
   for p,key in zip(group['params'],ids['params']):
    old=saved.get(key,{})
    if 'blocks' in old:
     self.state[p]['blocks']=[{k:(v.to(p.device,dtype=torch.float64) if torch.is_tensor(v) else v) for k,v in block.items()} for block in old['blocks']]
