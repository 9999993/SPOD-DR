"""Compute DEV metrics and LT1_J checkpoint rankings."""
from runtime import *
SHARED=ROOT
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score,roc_auc_score
from metrics import select_threshold,decode_threshold
METRICS=('qwk','auroc','FR','MISS')
def dev_metric(z):
    y=z['grade'];s=z['score'];fold=np.array([int(r['fold']) for r in csvread(SHARED/'DEV_FOLDS.csv')]);assert len(fold)==len(y)
    if z['image_key'].tolist() != [r['image_id'] for r in rows('DEV')]:
        raise ValueError('DEV prediction row order differs from the frozen DEV manifest')
    if not np.isfinite(s).all():return {'status':'NE'}
    referral=np.empty(len(y),dtype=bool);thresholds=[]
    for k in range(5):
        p=select_threshold(s[fold!=k],y[fold!=k]);thresholds.append(p)
        if p['status']!='PASS':return {'status':'NE','fold_policies':thresholds}
        referral[fold==k]=s[fold==k]>=decode_threshold(p['threshold'])
    neg=y<=1;pos=y>=2;FP=int((referral&neg).sum());FN=int((~referral&pos).sum())
    m={'status':'VALID','qwk':float(cohen_kappa_score(y,z['pred'],labels=list(range(5)),weights='quadratic')),'auroc':float(roc_auc_score(pos,s)),'FR':FP/int(neg.sum()),'MISS':FN/int(pos.sum()),'FP':FP,'FN':FN,'negative':int(neg.sum()),'positive':int(pos.sum()),'fold_policies':thresholds}
    for g in range(5):
        mask=y==g;m[f'grade{g}_n']=int(mask.sum());m[f'grade{g}_errors']=int((mask&(referral if g<2 else ~referral)).sum())
    if not all(np.isfinite(m[k]) for k in METRICS):m['status']='NE'
    return m
def ranks(metrics):
    n=len(metrics)
    if n==1:return np.zeros(1)
    values=[rankdata([-m[k] if k in ('qwk','auroc') else m[k] for m in metrics],method='average') for k in METRICS]
    q,a,f,s=[(v-1)/(n-1) for v in values]
    return (q+a+f+2*s)/5
def tie(m,ident):return (m['MISS'],m['FR'],-m['qwk'],-m['auroc'],ident)
