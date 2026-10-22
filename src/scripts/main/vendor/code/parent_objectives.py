"""Registered objectives; auxiliary raw inputs use the executed p4 definition."""
import torch
from numerics import cumulative,symmetric_kl
def losses(model,a,b,y,sources,p4):
 main=-a['logp'][torch.arange(len(y),device=y.device),y].mean()
 if model.parent_new and not model.full:return main,main*0,{'CE':main}
 if model.arm=='Nplus':return .5*(main-b['logp'][torch.arange(len(y),device=y.device),y].mean()),main*0,{'CE':main}
 j=symmetric_kl(a['logp'],b['logp'])
 if model.full:
  aa={**a,'grade_probs':a['historical_grade_probs'] if model.parent_new else a['logp'].exp(),'severity_score':a['mu']};bb={**b,'grade_probs':b['historical_grade_probs'] if model.parent_new else b['logp'].exp(),'severity_score':b['mu']}
  original,parts,_=p4.p4_loss('C2',aa,y,sources,bb)
  if model.parent_new:return original,original*0,parts
  cons=(a['mu']-b['mu']).abs().mean();aux=.2*parts['ord']+.1*parts['style']+.2*j+.1*cons
  return main,aux,{'CE':main,'ord':parts['ord'],'style':parts['style'],'J':j,'consistency':cons}
 grades=a['logp'].new_tensor(range(5));cons=((a['logp'].exp()*grades).sum(1)-(b['logp'].exp()*grades).sum(1)).abs().mean();cum=cumulative(a['logp'],y)
 total=main+.2*cum+.2*j+.1*cons;parts={'CE':main,'cumulative':cum,'J':j,'consistency':cons}
 if model.arm=='Z_JOINT':
  anchor=.01/(2*model.d.channels)*((model.d.gamma-model.gamma0).square().sum()+(model.d.beta-model.beta0).square().sum());total=total+anchor;parts['anchor']=anchor
 return total,total*0,parts
