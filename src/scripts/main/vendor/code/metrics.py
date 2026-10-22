"""C10 numeric risk, source threshold, and whole-tie retrospective frontier."""
import numpy as np
from scipy.special import logsumexp, expit

def scores(logits):
 x=np.asarray(logits,np.float64)
 if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all():raise ValueError('finite five-class logits required')
 return logsumexp(x[:,2:],axis=1)-logsumexp(x[:,:2],axis=1)

def wilson_lower(k,n):
 z=1.2815515655;p=np.asarray(k,np.float64)/n
 return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def select_threshold(score,grade):
 s=np.asarray(score,np.float64);y=np.asarray(grade)>=2
 if not np.isfinite(s).all():raise ValueError('finite log odds required')
 if y.sum()<15 or (~y).sum()<1:return {'status':'NOT_ESTIMABLE','threshold':None}
 candidates=np.r_[-np.inf,np.unique(s)];positives=np.sort(s[y])
 tp=len(positives)-np.searchsorted(positives,candidates,side='left')
 lower=wilson_lower(tp,len(positives));feasible=lower>=.90
 if not feasible.any():return {'status':'NOT_ESTIMABLE','threshold':None}
 index=np.flatnonzero(feasible)[-1];tau=float(candidates[index])
 # JSON has no -infinity numeric literal. Encode only the all-referral sentinel.
 return {'status':'PASS','threshold':tau if np.isfinite(tau) else '-inf','scale':'float64 log odds',
 'comparison':'>=','candidate_count':len(candidates),'risk_probability':float(expit(tau)),
 'CAL_positive':int(y.sum()),'CAL_negative':int((~y).sum()),'CAL_TP':int(tp[index]),
 'wilson_lower_90':float(lower[index])}

def decode_threshold(value):return -np.inf if value=='-inf' else value

def rates(grade,predicted,score,tau,probabilities=None):
 from sklearn.metrics import cohen_kappa_score,roc_auc_score,average_precision_score,confusion_matrix
 y=np.asarray(grade);s=np.asarray(score,np.float64);p=np.asarray(predicted);label=y>=2
 r=s>=decode_threshold(tau) if tau is not None else None
 out={'images':len(y),'positive_labels':int(label.sum()),'negative_labels':int((~label).sum()),
  'qwk':float(cohen_kappa_score(y,p,labels=list(range(5)),weights='quadratic')),
  'auroc':float(roc_auc_score(label,s)),'auprc':float(average_precision_score(label,s)),
  'confusion_matrix':confusion_matrix(y,p,labels=list(range(5))).tolist(),
  'predicted_grade_support':np.bincount(p,minlength=5).tolist()}
 masks={'total_referral_rate':np.ones(len(y),bool),'false_referral_grade0':y==0,'false_referral_grade1':y==1,'false_referral_grade01':y<=1,'miss_grade2':y==2,'miss_grade3':y==3,'miss_grade4':y==4,'miss_grade234':y>=2}
 for k,mask in masks.items():
  denom=int(mask.sum());event=(~r if k.startswith('miss_') else r) if r is not None else None
  num=int((mask&event).sum()) if event is not None else None
  out[k]=num/denom if num is not None and denom else None;out[k+'_numerator']=num;out[k+'_denominator']=denom
 if r is not None:
  for k,m in {'TP':r&label,'FP':r&~label,'TN':~r&~label,'FN':~r&label}.items():out[k]=int(m.sum())
  out['sensitivity']=1-out['miss_grade234'];out['specificity']=1-out['false_referral_grade01']
  out['ppv']=out['TP']/(out['TP']+out['FP']) if out['TP']+out['FP'] else None
  out['npv']=out['TN']/(out['TN']+out['FN']) if out['TN']+out['FN'] else None
 if probabilities is not None:
  probs=np.asarray(probabilities,np.float64);q=expit(s)
  out['nll']=float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1)).mean())
  out['brier']=float(((q-label)**2).mean());out['ece']=0.
  for i in range(10):
   mask=(q>=i/10)&((q<(i+1)/10) if i<9 else(q<=1))
   if mask.any():out['ece']+=float(mask.mean()*abs(q[mask].mean()-label[mask].mean()))
 return out

def frontier(score,grade):
 s=np.asarray(score,np.float64);y=np.asarray(grade)>=2
 order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
 ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
 tp=np.cumsum(yy)[ends];fp=ends+1-tp
 sens=tp/y.sum();spec=1-fp/(~y).sum()
 out={}
 for target in (.90,.95):
  ix=np.flatnonzero(sens>=target);j=int(ix[0])
  out[str(target)]={'specificity':float(spec[j]),'sensitivity':float(sens[j]),'threshold':float(ss[ends[j]]),'TP':int(tp[j]),'FP':int(fp[j]),'whole_tie':True,'retrospective_only':True}
 # Omit unrequested full frontier; only two fixed summaries.
 # out['high_sensitivity_steps']=[{'sensitivity':float(a),'specificity':float(b),'threshold':float(ss[e]),'TP':int(t),'FP':int(f)} for a,b,e,t,f in zip(sens,spec,ends,tp,fp) if a>=.85]
 return out

def referral_volume_bracket(score,grade,target):
 """Both attainable whole-tie sides of a frozen reference referral count."""
 s=np.asarray(score,np.float64);y=np.asarray(grade);order=np.argsort(-s,kind='stable');ss=s[order]
 ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1];counts=np.r_[0,ends+1]
 ix=sorted(set([int(np.flatnonzero(counts<=target)[-1]),int(np.flatnonzero(counts>=target)[0])]))
 output=[]
 for j in ix:
  tau=float(ss[ends[j-1]]) if j else None;r=s>=tau if j else np.zeros(len(s),bool)
  entry={'reference_referral_count':int(target),'achieved_referral_count':int(r.sum()),'exact_volume_match':int(r.sum())==target,'threshold':tau if j else '+inf','whole_tie':True,'retrospective_only':True,'new_CAL_policy':False}
  for name,mask,miss in [('false_referral_grade01',y<=1,False),('miss_grade234',y>=2,True),*[(f'miss_grade{g}',y==g,True) for g in (2,3,4)]]:
   n=int(mask.sum());k=int((mask&(~r if miss else r)).sum());entry.update({name:k/n if n else None,name+'_numerator':k,name+'_denominator':n})
  output.append(entry)
 return output
