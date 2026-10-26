"""One CPU reconstruction of all role splits, policies and paired clinical costs."""
from runtime import *
from collections import defaultdict
import shutil, math
from datetime import datetime, timedelta
from sklearn.metrics import cohen_kappa_score,roc_auc_score

DEST=WORK/'reports/ablation'/OWNER
CORE=('qwk','auroc','false_referral_grade01','miss_grade234')
EVENTS=('false_referral_grade0','false_referral_grade1','false_referral_grade01','miss_grade2','miss_grade3','miss_grade4','miss_grade234','undergrade_ge2_overall',*[f'undergrade_ge2_grade{g}' for g in range(5)])

def independent_threshold(s,y):
    positive=np.sort(s[y>=2]);c=np.r_[-np.inf,np.unique(s)];tp=len(positive)-np.searchsorted(positive,c,side='left');z=1.2815515655;n=len(positive);p=tp/n
    lower=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    return c[np.flatnonzero(lower>=.9)[-1]]
def metric(z,tau):
    y=z['grade'];pred=z['pred'];s=z['score'];n=len(y);ref=s>=decode_threshold(tau);cmat=np.zeros((5,5),dtype=np.int64);np.add.at(cmat,(y,pred),1)
    weights=(np.arange(5)[:,None]-np.arange(5)[None,:])**2;expected=np.outer(cmat.sum(1),cmat.sum(0))/max(n,1);den=float((weights*expected).sum())
    q=1-float((weights*cmat).sum())/den if den else None
    neg=np.sort(s[y<2]);pos=s[y>=2];auc=float((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right')).sum()*.5/(len(neg)*len(pos))) if len(neg) and len(pos) else None
    if q is not None:assert abs(q-cohen_kappa_score(y,pred,labels=list(range(5)),weights='quadratic'))<1e-12
    if auc is not None:assert abs(auc-roc_auc_score(y>=2,s))<1e-12
    result={'status':'VALID' if q is not None and auc is not None else 'NE_CORE_DENOMINATOR','images':n,'components':len(set(z['component_hash'])),'positive':int((y>=2).sum()),'negative':int((y<2).sum()),'qwk':q,'auroc':auc,'accuracy':float((y==pred).mean()) if n else None,'mae':float(np.abs(y-pred).mean()) if n else None,'confusion_matrix':cmat.tolist()}
    event_arrays={}
    for e in EVENTS:
        mask=np.ones(n,bool) if e=='undergrade_ge2_overall' else y<2 if e=='false_referral_grade01' else y>=2 if e=='miss_grade234' else y==int(e[-1])
        error=y-pred>=2 if e.startswith('undergrade') else ~ref if e.startswith('miss') else ref
        total=int(mask.sum());num=int((error&mask).sum());result.update({e:num/total if total else None,e+'_numerator':num,e+'_denominator':total});event_arrays[e]=(mask,error&mask)
    return result,event_arrays

def subset(z,mask): return {k:v[mask] for k,v in z.items() if isinstance(v,np.ndarray) and v.shape and len(v)==len(mask)}
