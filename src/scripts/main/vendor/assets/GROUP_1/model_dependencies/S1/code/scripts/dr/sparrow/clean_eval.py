"""Frozen shared evaluator API for clean R0. No model/old-result auto-discovery."""
from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
from clean_contract import DATA,ROOT,MANIFESTS,read_clean,sha,digest,immutable,canonical
import p4_analysis as old
import p4_execution as p4

SCHEMA='cleanr0_prediction_v1'
COHORTS={'DEV':2984,'Retino':1392,'DDR':12422,'SetB':400,'IDRiD':103}
CODES={'Retino':11,'DDR':17,'SetB':29,'IDRiD':37}
CLINICAL=old.CLINICAL_METRICS
SOURCES={
 'Retino':(p4.EXTERNAL_MANIFESTS['Retino'],'55a8c24c85f36d60a00bc6d3fa5d2e9c1faf9f1ccf6a60c77cced8f3f8f46fea'),
 'SetB':(p4.EXTERNAL_MANIFESTS['SetB'],'622237a3f07df2fdd7de2e313e78c2744c97e5cb183451f53c29fd2799a48c77'),
 'DDR':(p4.ARCHIVE/'reference_assets/GROUP_1/evaluation_manifests/ddr_gradable_unique_manifest.csv','8c8f4781eb6e4437b814be332b44f99e299891acd99ec0bd13c62826a0821ee5'),
 'IDRiD':(p4.MANIFEST_DIR/'idrid_external_test.csv','06fc2a7a2b8e105eaa56c9a9d2574842ea6f1e97ba2ca98b04d1daa7cc36fd1e'),
}

def rows_for(cohort):
 if cohort in ('CAL','DEV'):return read_clean(cohort)
 if cohort not in SOURCES:raise ValueError('cohort closed or unknown')
 path,expected=SOURCES[cohort]
 if sha(path)!=expected:raise ValueError(f'fixed {cohort} input manifest differs')
 with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 if len(rows)!=COHORTS[cohort]:raise ValueError('fixed cohort count mismatch')
 for i,r in enumerate(rows):
  g=int(r['grade'])
  if g not in range(5) or int(r['label'])!=int(g>=2):raise ValueError('external grade/q2 mismatch')
  cache=str(r.get('cache_path','')).strip()
  if cohort=='DDR':
   prefix='analysis/public_medcv_3direction_program_v1/preprocessing/diabetic_retinopathy_external_risk_discovery/ddr_rgb512/cache/'
   relative=Path(cache)
   if relative.is_absolute() or '..' in relative.parts or not cache.startswith(prefix):raise ValueError('DDR cache must use the registered source-relative prefix')
   mapped=p4.ARCHIVE/'model_sources/evaluation_sources'/relative
   r['cache_path']=str(mapped)
  else:r['cache_path']=str(p4._mapped_path(cache)) if cache else ''
  r['image_path']=str(p4._mapped_path(r['image_path']))
  if not Path(r['cache_path'] or r['image_path']).is_file():raise FileNotFoundError('registered evaluation input missing')
  # Preserve Q2's published duplicate component keys for DDR; SetB uses patient.
  component=r['component_hash'] if cohort=='DDR' else digest([cohort,r['patient_id'] if cohort=='SetB' else r['image_id']])
  r['component_hash']=component;r['row_index']=str(i)
 return rows

def manifest_identity(cohort):
 rows=rows_for(cohort)
 path=DATA/MANIFESTS[cohort][0] if cohort in ('CAL','DEV') else SOURCES[cohort][0]
 return {'path':str(path),'sha256':sha(path),'images':len(rows),'ordered_keys_sha256':digest([[r['row_index'],r['component_hash'],int(r['grade'])] for r in rows]),'components':len({r['component_hash'] for r in rows}),'grade_support':{str(g):sum(int(r['grade'])==g for r in rows) for g in range(5)},'component_unit':'official patient key' if cohort=='SetB' else 'accepted clean component' if cohort in ('CAL','DEV') else 'registered image/duplicate proxy; not patient'}

def evaluator_contract():
 return {'schema':'cleanr0_shared_evaluator_v1','prediction_schema':SCHEMA,'score':'q2 = float64 sum of stored float32 p2:p4 cast to float32; never screening head','grade_pred':'argmax(p0:p4), first maximum','cohorts':{c:manifest_identity(c) for c in ('CAL',*COHORTS)},'threshold':{'algorithm':'p4_analysis.select_calibration_threshold (corrected)','candidates':'unique q2 union {0}','positive':'q2>=tau','criterion':'maximum feasible one-sided 90% Wilson lower sensitivity >= .90','z':1.2815515655,'minimum_true_positive':15,'minimum_true_negative':1,'no_feasible':'explicit NA, no 0.5 fallback','new_model_new_CAL_and_threshold':True},'metrics':{'qwk':'sklearn quadratic kappa labels [0,1,2,3,4]','auroc_q2':'sklearn roc_auc_score, ties handled','auprc_q2':'sklearn average_precision_score (P4 definition)','nll':'five grade NLL clip[1e-12,1]','brier':'binary q2 Brier','ece':'10 equal-width [0,1] bins; left-inclusive, right-exclusive except last inclusive; weighted absolute gap; original clinical_metrics binning','clinical':'true grade conditioned counts/rates; denominators preserved when threshold NA','units':'rates in [0,1], differences same units; display clinical differences in percentage points'},'input_path_mapping':{'DDR_cache_relative_root':str(p4.ARCHIVE/'model_sources/evaluation_sources'),'other_external': 'explicit historical CRRL/data migration mapping; no fallback'},'preprocessing':{'cached':'accepted registered float32 CHW 3x512x512 ImageNet normalized; no new preprocessing','Retino':'RGB centered square, BICUBIC512, [0,1], mean(.485,.456,.406) std(.229,.224,.225)'},'bootstrap':{'rng':'PCG64 SeedSequence([20261004,cohort_code])','codes':CODES,'candidate_cap':20000,'valid_per_estimand_group':10000,'percentile':'linear 2.5/97.5','shared_draws':'all compared new models within cohort','seed_resampling':False,'condition':'fixed paired training seeds and CAL thresholds; uncorrected multiple comparisons'},'publishing_guard':'all S1/S2/S3 core scientific configs frozen before new external metrics are published','SetC':'CLOSED','legacy_models':'HISTORY_ONLY'}

def make_prediction(prob,rows):
 prob=np.asarray(prob,dtype=np.float32)
 return {'grade_probs':prob,'grade_pred':prob.argmax(1).astype(np.int8),'q2':prob[:,2:].sum(1,dtype=np.float64).astype(np.float32),'grade':np.array([int(r['grade']) for r in rows],np.int8),'row_index':np.arange(len(rows),dtype=np.int32),'component_hash':np.array([r['component_hash'] for r in rows],dtype='U64')}

def validate_prediction(pred,rows=None):
 required=('grade_probs','grade_pred','q2','grade','row_index','component_hash')
 if set(pred)!=set(required):raise ValueError('prediction fields mismatch')
 p=np.asarray(pred['grade_probs']);n=len(p)
 if p.dtype!=np.float32 or p.shape!=(n,5) or n==0:raise ValueError('probability format mismatch')
 if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=1e-5,atol=1e-6):raise ValueError('invalid five-level distribution')
 for k in required[1:]:
  if np.asarray(pred[k]).shape!=(n,):raise ValueError('prediction row alignment mismatch')
 if not np.isin(pred['grade'],range(5)).all():raise ValueError('grade outside [0,4]')
 if not np.array_equal(pred['row_index'],np.arange(n)):raise ValueError('manifest order required')
 if any(len(str(c))!=64 for c in pred['component_hash']):raise ValueError('component identity malformed')
 if not np.array_equal(pred['grade_pred'],p.argmax(1)):raise ValueError('grade prediction mismatch')
 if not np.array_equal(pred['q2'],p[:,2:].sum(1,dtype=np.float64).astype(np.float32)):raise ValueError('q2 must be regenerated from stored grade probabilities')
 if rows is not None:
  if n!=len(rows) or not np.array_equal(pred['grade'],[int(r['grade']) for r in rows]) or not np.array_equal(pred['component_hash'],[r['component_hash'] for r in rows]):raise ValueError('prediction differs from frozen cohort keys')
 return pred

def save_prediction(path,pred,metadata,rows):
 validate_prediction(pred,rows);path=Path(path)
 if path.exists():raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('xb') as f:np.savez_compressed(f,**pred);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path)
 finally:tmp.unlink()
 immutable(path.with_suffix('.receipt.json'),{**metadata,'schema':SCHEMA,'prediction_sha256':sha(path),'images':len(rows)})

def load_prediction(path,cohort=None,bindings=None):
 path=Path(path);r=json.loads(path.with_suffix('.receipt.json').read_text())
 if r['schema']!=SCHEMA or r['prediction_sha256']!=sha(path):raise ValueError('prediction receipt mismatch')
 if bindings:
  for k,v in bindings.items():
   if r.get(k)!=v:raise ValueError(f'prediction binding mismatch: {k}')
 with np.load(path,allow_pickle=False) as a:pred={k:a[k] for k in a.files}
 return validate_prediction(pred,rows_for(cohort) if cohort else None),r

def select_threshold(pred):
 validate_prediction(pred)
 return old.select_calibration_threshold(pred['q2'],(pred['grade']>=2).astype(np.int8))

def metrics(pred,tau):
 validate_prediction(pred);p={**pred,'screening_prob':pred['q2']}
 out=old._metrics_for_predictions(p,tau);out.pop('auroc_screening_probability')
 y=pred['grade'];q=pred['q2'].astype(np.float64);ref=q>=tau if tau is not None else None
 for name,(mask,event) in {'false_referral_grade0':(y==0,True),'false_referral_grade1':(y==1,True),'false_referral_grade01':(y<=1,True),'miss_grade2':(y==2,False),'miss_grade3':(y==3,False),'miss_grade4':(y==4,False),'miss_grade234':(y>=2,False),'sensitivity':(y>=2,True),'specificity':(y<=1,False)}.items():
  if tau is None:out['clinical_counts'][name]={'numerator':None,'denominator':int(mask.sum()),'rate':None,'reason':'model CAL threshold not estimable'}
 bins=np.linspace(0,1,11);ece=0.;cal=[]
 for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
  mask=(q>=lo)&((q<hi) if i<9 else(q<=hi));n=int(mask.sum());score=float(q[mask].mean()) if n else None;truth=float((y[mask]>=2).mean()) if n else None
  if n:ece+=n/len(y)*abs(score-truth)
  cal.append({'left':float(lo),'right':float(hi),'n':n,'mean_q2':score,'referable_fraction':truth})
 out.update({'ece_q2':ece,'reliability_bins_q2':cal,'positive_fraction':float(ref.mean()) if ref is not None else None,'distant_undergrade_ge2':{'numerator':int(((y>=2)&(pred['grade_pred']<=y-2)).sum()),'denominator':int((y>=2).sum())}})
 if ref is not None:
  for name,mask,event in [('ppv',ref,y>=2),('npv',~ref,y<=1)]:
   n=int(mask.sum());k=int((mask&event).sum());out[name]={'numerator':k,'denominator':n,'rate':k/n if n else None}
 else:out['ppv']=out['npv']={'numerator':None,'denominator':None,'rate':None}
 return out

# Efficient inherited estimands, with this new random stream and per-estimand validity.
def bootstrap(cohort,predictions,thresholds,pairs,*,required=10000,attempt_cap=20000):
 if cohort not in CODES:raise ValueError('bootstrap only for fixed four tests')
 keys=sorted(predictions);first=predictions[keys[0]];y=first['grade']
 for k in keys:
  validate_prediction(predictions[k])
  if not np.array_equal(y,predictions[k]['grade']) or not np.array_equal(first['component_hash'],predictions[k]['component_hash']):raise ValueError('paired cases differ')
 _,clusters=np.unique(first['component_hash'],return_inverse=True);ncluster=int(clusters.max()+1)
 rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261004,CODES[cohort]])))
 # pairs maps comparison name to list of same-seed (left,right) model keys.
 groups={name:{m:[] for m in ('qwk','auroc_q2',*CLINICAL)} for name in pairs};attempted=0
 for w,_ in old._cluster_weight_batches(ncluster,clusters,rng,attempt_cap=attempt_cap,batch_size=64):
  attempted+=len(w);values={}
  for k in keys:
   p=predictions[k];values[k]={'qwk':old._weighted_qwk_many(y,p['grade_pred'],w),'auroc_q2':old._weighted_auc_many((y>=2).astype(np.int8),p['q2'],w)}
   clinical=old._clinical_rates_many(y,(p['q2']>=thresholds[k])[:,None],w)[:,0,:] if thresholds.get(k) is not None else np.full((len(w),len(CLINICAL)),np.nan)
   values[k].update({m:clinical[:,i] for i,m in enumerate(CLINICAL)})
  for name,pp in pairs.items():
   for metric in groups[name]:
    v=np.mean([values[a][metric]-values[b][metric] for a,b in pp],axis=0)
    groups[name][metric].extend(v[np.isfinite(v)].tolist())
  if all(len(v)>=required for g in groups.values() for v in g.values()):break
 out={}
 for name,g in groups.items():
  out[name]={}
  for m,v in g.items():
   a=np.array(v[:required]);valid=len(a)
   out[name][m]={'status':'PASS' if valid==required else 'NOT_ESTIMABLE','valid':valid,'required':required,'attempted':attempted,'ci95':np.quantile(a,[.025,.975],method='linear').tolist() if valid==required else None,'bootstrap_mean':float(a.mean()) if valid else None,'reason':None if valid==required else 'fewer than required finite paired draws; fixed candidates exhausted'}
 return out

def predict_model(model,rows,device,batch_size=16,workers=2):
 """Shared inference adapter. Caller binds its own new checkpoint and phase locks."""
 import torch
 from torch.utils.data import DataLoader
 model.eval();loader=DataLoader(old._EvalDataset(rows),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
 probs=[]
 try:
  with torch.inference_mode():
   for batch in loader:
    output=model(batch['image'].to(device,non_blocking=True))
    if hasattr(model,'feature_pyramid') and hasattr(model.feature_pyramid,'pop_aux'):model.feature_pyramid.pop_aux()
    probs.append(output['grade_probs'].float().cpu().numpy())
 finally:
  if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
 return validate_prediction(make_prediction(np.concatenate(probs),rows),rows)
