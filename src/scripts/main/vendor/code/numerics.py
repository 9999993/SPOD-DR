"""FP64 differentiable fixed-bin Gaussian; no class floor or detached gradient."""
import torch

def logdiff(a,b):
 delta=b-a
 if not bool((delta<0).all()):raise FloatingPointError('nonpositive Gaussian interval')
 return a+torch.log(-torch.expm1(delta))

def gaussian_logp(mu,sigma):
 mu=mu.double();sigma=sigma.double().clamp_min(1e-6)
 z=(mu.new_tensor([.5,1.5,2.5,3.5])[None,:]-mu[:,None])/sigma[:,None]
 out=[torch.special.log_ndtr(z[:,0])]
 for k in range(3):
  lo,hi=z[:,k],z[:,k+1];positive=lo>=0
  v=torch.empty_like(lo)
  if bool(positive.any()):v[positive]=logdiff(torch.special.log_ndtr(-lo[positive]),torch.special.log_ndtr(-hi[positive]))
  other=~positive
  if bool(other.any()):v[other]=logdiff(torch.special.log_ndtr(hi[other]),torch.special.log_ndtr(lo[other]))
  out.append(v)
 out.append(torch.special.log_ndtr(-z[:,3]))
 logp=torch.stack(out,1)
 if not bool(torch.isfinite(logp).all()):raise FloatingPointError('nonfinite Gaussian logp')
 return logp

def score(logp):return torch.logsumexp(logp[:,2:],1)-torch.logsumexp(logp[:,:2],1)
def cumulative(logp,y):
 parts=[]
 for k in range(1,5):
  pos=torch.logsumexp(logp[:,k:],1);neg=torch.logsumexp(logp[:,:k],1)
  parts.append(-torch.where(y>=k,pos,neg))
 return torch.stack(parts,1).mean()
def symmetric_kl(a,b):return .5*((a.exp()*(a-b)).sum(1)+(b.exp()*(b-a)).sum(1)).mean()
def objective(row,o,v,y,sources,p4):
 a,b=o['logp'],v['logp'];ce=-a[torch.arange(len(y),device=y.device),y].mean()
 if row=='N-budget':return .5*(ce-b[torch.arange(len(y),device=y.device),y].mean()),{'CE':ce}
 j=symmetric_kl(a,b)
 if row in ('G','R','A'):cons=(o['mu']-v['mu']).abs().mean()
 else:
  grades=a.new_tensor(range(5));cons=((a.exp()*grades).sum(1)-(b.exp()*grades).sum(1)).abs().mean()
 total=ce+.2*j+.1*cons;parts={'CE':ce,'J':j,'consistency':cons}
 if row in ('A','B0','B'):
  cum=cumulative(a,y);total=total+.2*cum;parts['cumulative']=cum
 if row=='G':
  # Use the bound original definitions for raw axes/style, retaining their clamps.
  original={**o,'grade_probs':a.exp(),'severity_score':o['mu']}
  vv={**v,'grade_probs':b.exp(),'severity_score':v['mu']}
  _,aux,_=p4.p4_loss('C2',original,y,sources,vv)
  total=total+.2*aux['ord']+.1*aux['style'];parts.update(ord=aux['ord'],style=aux['style'])
 if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite objective')
 return total,parts
