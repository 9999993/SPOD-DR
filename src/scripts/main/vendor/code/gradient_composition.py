"""Two autograd traversals of one graph; fixed S/A, FP64 projection reduction."""
import torch
def combine(g,h,shared,project):
 q=sum((x.double().square().sum() for x,s in zip(g,shared) if s),torch.zeros((),device=g[0].device,dtype=torch.float64))
 d=sum(((x.double()*y.double()).sum() for x,y,s in zip(g,h,shared) if s),torch.zeros_like(q))
 hn=sum((y.double().square().sum() for y,s in zip(h,shared) if s),torch.zeros_like(q))
 flag=bool(project and q>1e-24 and d<0)
 ratio=d/q if flag else torch.zeros_like(q)
 grads=[(x.double()+y.double()-ratio*x.double()).to(x.dtype) if s else y for x,y,s in zip(g,h,shared)]
 return grads,{'q':float(q),'d':float(d),'cos':float(d/(q*hn).sqrt()) if q>0 and hn>0 else None,'projected':flag}
def backward(main,aux,model,project,probe=False):
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];ps=[p for n,p in named]
 gg=torch.autograd.grad(main,ps,retain_graph=True,allow_unused=True)
 hh=torch.autograd.grad(aux,ps,allow_unused=True)
 g=[];h=[];shared=[];none=[]
 for (n,p),x,y in zip(named,gg,hh):
  s=n not in model.aux_only
  if x is None and s:raise RuntimeError('unexpected primary disconnection '+n)
  if x is not None and not s:raise RuntimeError('aux-only consumed by primary '+n)
  if y is None and n not in model.legal_aux_none:raise RuntimeError('unexpected auxiliary disconnection '+n)
  if x is None or y is None:none.append({'name':n,'main_none':x is None,'aux_none':y is None})
  g.append(torch.zeros_like(p) if x is None else x.detach());h.append(torch.zeros_like(p) if y is None else y.detach());shared.append(s)
  if not torch.isfinite(g[-1]).all() or not torch.isfinite(h[-1]).all():raise FloatingPointError('nonfinite gradient '+n)
 vals,log=combine(g,h,shared,project)
 for p,v in zip(ps,vals):p.grad=v
 if probe:log['legal_none']=none
 return log
