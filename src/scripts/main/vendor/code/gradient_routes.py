"""Final-output VJP removal on F; unchanged clean forward and auxiliaries."""
import torch
from numerics import gaussian_logp

def partials(a,y):
 u=a['mu'].detach().requires_grad_(True);v=a['sigma'].detach().requires_grad_(True)
 n=-gaussian_logp(u,v).gather(1,y[:,None]).mean()
 du,dv=torch.autograd.grad(n,(u,v))
 return n.detach(),du.detach(),dv.detach()

def backward_route(main,aux,m,a,y,diagnose=False):
 named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];ps=[p for _,p in named]
 gn=torch.autograd.grad(main,ps,retain_graph=True,allow_unused=True)
 ga=torch.autograd.grad(aux,ps,retain_graph=True,allow_unused=True)
 val,du,dv=partials(a,y)
 if not torch.equal(val,main.detach()):raise RuntimeError('main consumer/output partial mismatch')
 indices=[i for i,(n,_) in enumerate(named) if n in m.extra_names];fp=[ps[i] for i in indices]
 removed={};routes={}
 keys=('mu','sigma') if diagnose else (('sigma',) if m.arm=='SGF' else ('mu',))
 for key in keys:
  routes[key]=torch.autograd.grad(a[key],fp,grad_outputs=du if key=='mu' else dv,retain_graph=True,allow_unused=True)
 route=routes['sigma' if m.arm=='SGF' else 'mu'];removed=dict(zip(indices,route))
 none=[];err2=norm2=0.;r2=mu2=sigma2=dot=0.
 for i,((n,p),x,z) in enumerate(zip(named,gn,ga)):
  if (x is None)!=(n in m.aux_only):raise RuntimeError('unexpected main disconnection '+n)
  if z is None and n not in m.legal_aux_none:raise RuntimeError('unexpected aux disconnection '+n)
  if x is None or z is None:none.append(dict(name=n,main_none=x is None,aux_none=z is None))
  g=torch.zeros_like(p) if x is None else x.detach();h=torch.zeros_like(p) if z is None else z.detach()
  q=removed.get(i);q=torch.zeros_like(p) if q is None else q.detach()
  # Match the original FP64 composition before casting to parameter dtype.
  p.grad=(g.double()-q.double()+h.double()).to(p.dtype)
  if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite routed gradient '+n)
  r2+=float(q.double().square().sum())
  if diagnose and i in indices:
   j=indices.index(i);u=routes['mu'][j];v=routes['sigma'][j]
   u=torch.zeros_like(p) if u is None else u;v=torch.zeros_like(p) if v is None else v
   err2+=float((g.double()-u.double()-v.double()).square().sum());norm2+=float(g.double().square().sum())
   mu2+=float(u.double().square().sum());sigma2+=float(v.double().square().sum());dot+=float((u.double()*v.double()).sum())
 rel=(err2/max(norm2,1e-300))**.5
 if diagnose and rel>1e-5:raise RuntimeError('clean gradient chain reconstruction '+str(rel))
 log=dict(route=m.arm,removed_norm=r2**.5,chain_relative_L2=rel if diagnose else None,legal_none=none,H_preclip_unchanged=True)
 if diagnose:
  log.update(mu_F_norm=mu2**.5,sigma_F_norm=sigma2**.5,mu_sigma_cos=dot/(mu2*sigma2)**.5 if mu2*sigma2 else None,
   by_grade=[dict(grade=k,count=int((y==k).sum()),mu_mean=float(a['mu'][y==k].detach().mean()) if (y==k).any() else None,sigma_mean=float(a['sigma'][y==k].detach().mean()) if (y==k).any() else None,mu_partial_norm=float(du[y==k].double().norm()),sigma_partial_norm=float(dv[y==k].double().norm())) for k in range(5)])
 return log
