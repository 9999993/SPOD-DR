"""Fixed C22 training, shared prefixes, SAM replay and complete atomic recovery."""
from common import *
from models import Model,p4
from objectives import losses,raw_nll
from gradient_composition import backward
import torch,numpy as np,resource,traceback,signal,pickle
TRAINING_SOURCE_FILES=('common.py','assets.py','models.py','numerics.py','loss_reference.py','gradient_composition.py','objectives.py','training.py','parent_models.py','parent_objectives.py','parent_training.py','gradient_routes.py','optimizer_sham.py')
SOURCE_HASHES_AT_IMPORT={n:sha(REPORT/'code'/n) for n in TRAINING_SOURCE_FILES}
STOP_REQUESTED=False
def request_stop(*args):
 global STOP_REQUESTED
 STOP_REQUESTED=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
def precision():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
def cpu_seconds():return sum(resource.getrusage(w).ru_utime+resource.getrusage(w).ru_stime for w in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))
def counter():return dict(updates=0,training_forward=0,teacher_forward=0,extra_forward=0,reverse_sweeps=0)
def loader(seed,start,stop,rs=None):
 sampler=p4.P4BatchSampler(len(rs) if rs is not None else len(manifest('TRAIN')),seed,15001+start,15001+stop)
 return torch.utils.data.DataLoader(p4.P4ManifestDataset(rs or manifest('TRAIN')),batch_sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed+710000))
def initialize(b,arm,s):
 m=Model(b,arm,s).cuda();m.restore_parent_rng();return m,m.configure(1)
def optimizer_digest(o):
 state={}
 def visit(v,path):
  if torch.is_tensor(v):state[path]=v
  elif isinstance(v,dict):
   for k,z in sorted(v.items(),key=lambda x:str(x[0])):visit(z,path+':'+str(k))
  elif isinstance(v,(tuple,list)):
   for i,z in enumerate(v):visit(z,path+':'+str(i))
  else:state[path]=torch.tensor(list(str(v).encode()),dtype=torch.uint8)
 for i,p in enumerate([p for g in o.param_groups for p in g['params']]):visit(o.state.get(p,{}),str(i))
 return p4._state_digest(state)
def state_digest(m,o):
 r=p4._rng_state()
 return {'model':p4._state_digest(m.state_dict()),'optimizer':optimizer_digest(o),'rng':hashlib.sha256(pickle.dumps({'python':r['python'],'numpy':r['numpy']})).hexdigest(),'torch_rng':p4._state_digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'mix':m.mix_cpu.get_state()})}
def snapshot(m,o,local,counts,prior=None):
 return {'model':m.state_dict(),'optimizer':o.state_dict(),'rng':p4._rng_state(),'mix_rng':m.mix_cpu.get_state(),'local_update':local,'next_update':local,'global_next_zero_index':15001+local,'phase':m.phase,'backbone':m.b,'arm':m.arm,'seed':m.seed,'sampler':{'size':len(manifest('TRAIN')),'next_zero_index':15001+local},'parameter_lock':m.lock,'counts':{k:counts[k]+(prior or {}).get(k,0) for k in counts},'code_hashes':SOURCE_HASHES_AT_IMPORT,'exposure':getattr(m,'exposure',{}),'sp_alpha':m.sp_alpha,'sam_rho':m.sam_rho,'parent_seed':m.parent_seed,'continuation_seed':m.continuation_seed,'replicate':m.replicate}
def save(path,m,o,local,counts,prior=None):p4._atomic_torch_save(snapshot(m,o,local,counts,prior),Path(path))
def restore(path,b,arm,seed,fork=False):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['backbone']==b and ck['seed']==seed
 assert fork or ck['arm']==arm
 if not fork:assert ck['code_hashes']==SOURCE_HASHES_AT_IMPORT,'training source changed'
 else:
  binding=read(RUN/'contracts/REUSE_BINDINGS.json')
  allowed={x['sha256'] for x in binding['prefixes'].values()}
  assert ck['code_hashes']==SOURCE_HASHES_AT_IMPORT or sha(path) in allowed,'unregistered foreign prefix'
 m=Model(b,arm,seed).cuda();o=m.configure(ck['phase']);m.load_state_dict(ck['model'],strict=True)
 o.load_state_dict(ck['optimizer']);m.sp_alpha=ck.get('sp_alpha',.01);m.sam_rho=ck.get('sam_rho',.05)
 m.parent_seed=seed;m.continuation_seed=ck.get('continuation_seed',seed);m.replicate=ck.get('replicate',0)
 m.exposure=ck.get('exposure',{});m.mix_cpu.set_state(ck['mix_rng']);p4._restore_rng_state(ck['rng'])
 return m,o,ck['local_update'],ck['counts']

def initialize_stream(b,arm,parent_seed,replicate):
 m=Model(b,arm,parent_seed).cuda();m.restore_parent_rng();m.parent_seed=parent_seed;m.replicate=replicate;m.continuation_seed=parent_seed+10000*replicate
 if replicate==1:
  p4._set_seed(m.continuation_seed);m.mix_cpu.manual_seed(m.continuation_seed+970000)
 return m,m.configure(1)
def frozen_digest(m):
 active={n for n,p in m.named_parameters() if p.requires_grad};mutable={n for n,x in m.named_modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm) and x.training}
 assert {n:x.training for n,x in m.named_modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)}==m.lock['BN']
 return p4._state_digest({n:v for n,v in m.state_dict().items() if n not in active and not any(n.startswith(k+'.') for k in mutable)})
def gradient_norm(ps):return torch.stack([p.grad.double().square().sum() for p in ps if p.grad is not None]).sum().sqrt()
def displacement(m):
 a=m.anchors();p=dict(m.named_parameters());d=sum((p[n].detach().double()-v.double()).square().sum() for n,v in a.items());z=sum(v.double().square().sum() for v in a.values())
 return {'displacement_L2':float(d.sqrt()),'relative_parent_norm':float((d/z).sqrt()) if z>0 else None,'scope':'descriptive parameter displacement only; no anchoring term in C23'}
def numeric_observation(m,a,y,sources):
 from numerics import gaussian_logp
 mu=a['mu'].detach().clone().requires_grad_(True);sigma=a['sigma'].detach().clone().requires_grad_(True)
 stable=-gaussian_logp(mu,sigma).gather(1,y[:,None]).mean();raw=raw_nll(m,mu,sigma,y)
 original_output={**a,'grade_probs':m.inner.base._probs(mu.float(),sigma.float()),'severity_score':mu}
 _,original_parts,_=p4.p4_loss('C2',original_output,y,sources,None)
 reference=original_parts['grade'];assert torch.equal(raw,reference),'RAW_N value differs from actual p4 grade consumer'
 reference_grad=torch.autograd.grad(reference,(mu,sigma),retain_graph=True)
 gs=torch.autograd.grad(stable,(mu,sigma),retain_graph=True);gr=torch.autograd.grad(raw,(mu,sigma));p=m.inner.base._probs(mu.detach().float(),sigma.detach().float())
 assert all(torch.equal(x,z) for x,z in zip(reference_grad,gr)),'RAW_N derivative differs from actual p4 grade consumer'
 result={'actual_p4_grade_value_bitwise':True,'actual_p4_mu_sigma_gradients_bitwise':True,'N_stable':float(stable),'N_raw':float(raw),'raw_floor_or_zero_entries':int((p<=1e-7).sum()),'by_grade':[]}
 for k in range(5):
  mask=y==k
  result['by_grade'].append({'grade':k,'support':int(mask.sum()),'stable_derivative_norm':float(sum(g[mask].double().square().sum() for g in gs).sqrt()),'raw_derivative_norm':float(sum(g[mask].double().square().sum() for g in gr).sqrt()),'derivative_dot':float(sum((x[mask].double()*z[mask].double()).sum() for x,z in zip(gs,gr)))})
 return result
def step(m,o,batch,local,counts,probe=False,rows=None):
 x=batch['image'].cuda(non_blocking=True);y=batch['grade'].cuda(non_blocking=True)
 _,_,orbit,calibrate=p4._bootstrap_old_sources()
 with torch.no_grad():aux=p4._normalize_from_01(orbit(p4._denormalize(x),torch.ones((len(x),1,512,512),device=x.device,dtype=x.dtype),'gamma',m.continuation_seed*100000+15001+local,calibrate(seed=m.continuation_seed)).image)
 ps=[p for p in m.parameters() if p.requires_grad];o.zero_grad(set_to_none=True)
 do_sam=m.phase==2 and m.arm in ('SAM','SAM_ON','F_SAM','H_SAM','L_SAM');do_sp=m.phase==2 and m.arm in ('SP','SP_SAM')
 rng_before=p4._rng_state();mix_before=m.mix_cpu.get_state().clone();saved=None;numeric=None;first_loss=None;perturb_norm=0.;restored=True
 def data_gradient(observe):
  nonlocal numeric
  counts['training_forward']+=len(x);a=m(x);counts['training_forward']+=len(x);b=m(aux,aux=True)
  main,other,parts=losses(m,a,b,y,batch['source'],p4);total=main+other
  if not torch.isfinite(total):raise FloatingPointError('nonfinite data objective')
  if observe and probe:numeric=numeric_observation(m,a,y,batch['source'])
  if m.phase==2 and m.arm in ('SGF','MGF'):
   from gradient_routes import backward_route
   log=backward_route(main,other,m,a,y,probe or local in (1501,3000,6000))
  else:log=backward(main,other,m,False,probe) if m.phase==2 else {}
  if m.phase==1:total.backward()
  counts['reverse_sweeps']+=((4 if probe or local in (1501,3000,6000) else 3) if m.arm in ('SGF','MGF') else 2) if m.phase==2 else 1
  return float(total.detach()),{k:float(v.detach()) for k,v in parts.items()},log
 data,parts,log=data_gradient(True);rng_after=p4._rng_state();first_loss=data;data_norm=float(gradient_norm(ps))
 if do_sam:
  selected=m.__dict__.setdefault('_selected_cache',m.selected_names()) if '_selected_cache' not in m.__dict__ else m.__dict__['_selected_cache'];selected_ps=[p for n,p in m.named_parameters() if p.requires_grad and n in selected];saved=[p.detach().clone() for p in ps];norm=gradient_norm(selected_ps);selected_norm=float(norm);eps=[torch.zeros_like(p) if p.grad is None or id(p) not in {id(q) for q in selected_ps} else (p.grad.double()*(m.sam_rho/(norm+1e-12))).to(p.dtype) for p in ps]
  perturb_norm=float(sum(e.double().square().sum() for e in eps).sqrt())
  if probe:
   assert all(not torch.count_nonzero(e) for p,e in zip(ps,eps) if id(p) not in {id(q) for q in selected_ps})
   assert perturb_norm<=m.sam_rho+1e-6
  try:
   with torch.no_grad():
    for p,v,e in zip(ps,saved,eps):p.copy_(v+e)
   o.zero_grad(set_to_none=True);p4._restore_rng_state(rng_before);data,parts,log=data_gradient(False)
   if probe:
    second=p4._rng_state()
    assert pickle.dumps(second['python'])==pickle.dumps(rng_after['python']) and pickle.dumps(second['numpy'])==pickle.dumps(rng_after['numpy'])
    assert torch.equal(second['torch_cpu'],rng_after['torch_cpu']) and torch.equal(second['torch_cuda'],rng_after['torch_cuda'])
    assert torch.equal(m.mix_cpu.get_state(),mix_before), 'unexpected independent-generator consumption'
  finally:
   with torch.no_grad():
    for p,v in zip(ps,saved):p.copy_(v)
   p4._restore_rng_state(rng_after)
  if probe:restored=all(torch.equal(p,v) for p,v in zip(ps,saved));assert restored
  del saved,eps
  data_norm=float(gradient_norm(ps))
 drift=displacement(m) if probe or local%500==0 or local in (1500,1501,6000) else None
 if do_sp:
  anchors=m.anchors()
  with torch.no_grad():
   for n,p in m.named_parameters():
    if n in m.extra_names and p.requires_grad:
     assert p.grad is not None;p.grad=(p.grad.double()+m.sp_alpha*(p.detach().double()-anchors[n].double())).to(p.dtype)
 grads={}
 if probe:
  for n,p in m.named_parameters():
   if p.requires_grad:
    assert p.grad is not None and torch.isfinite(p.grad).all(),n
    grads[n]={'finite':True,'nonzero':bool(torch.count_nonzero(p.grad)),'norm':float(p.grad.double().norm())}
 norm=torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True);post=float(gradient_norm(ps));o.step();counts['updates']+=1
 if probe or local%100==0:m.check_bn()
 exposure=getattr(m,'exposure',{'grades':[0]*5,'sources':{},'style_anchors':0,'style_anchor_by_grade':[0]*5,'clip_count':0,'samples':0})
 if not exposure:exposure={'grades':[0]*5,'sources':{},'style_anchors':0,'style_anchor_by_grade':[0]*5,'clip_count':0,'samples':0}
 for k in range(5):exposure['grades'][k]+=int((batch['grade']==k).sum());exposure['style_anchor_by_grade'][k]+=m.last_exposure['style_anchor_by_grade'][k]
 for source in batch['source']:exposure['sources'][source]=exposure['sources'].get(source,0)+1
 exposure['samples']+=len(y);exposure['style_anchors']+=m.last_exposure['style_anchors'];exposure['clip_count']+=int(float(norm)>1);m.exposure=exposure
 return {'loss':data,'parts':parts,'preclip_norm':float(norm),'postclip_norm':post,'clip_ratio':min(1.,1/(float(norm)+1e-6)),'composition':log,'gradients':grads,'numeric_observation':numeric,'drift':drift,'data_gradient_norm':data_norm,'SAM':{'active':do_sam,'first_D':first_loss,'second_D':data if do_sam else None,'perturbation_norm':perturb_norm,'selected_gradient_norm':selected_norm if do_sam else None,'selected_parameter_count':len(selected_ps) if do_sam else 0,'parameters_restored_bitwise':restored,'same_inputs_and_RNG_replay':do_sam,'unselected_epsilon_exact_zero':True if do_sam else None},'optimizer_steps':sorted({int(v['step']) for v in o.state.values() if 'step' in v}),'SHAM':o.last_audit if m.arm=='SHAM' and m.phase==2 and (probe or local%100==0) else None}


def train(task):
 precision();b=task['backbone'];s=int(task['parent_seed']);rep=int(task['replicate']);arm=task.get('arm','PREFIX');target=int(task['target']);root=task_path(task['id']);root.mkdir(parents=True,exist_ok=True)
 if (root/'DONE.json').exists():
  d=read(root/'DONE.json');assert d['local_update']==target and sha(root/'final.pt')==d['sha256'];return
 assert read(REPORT/'QUALIFICATION.json')['status']=='PASS'
 assert read(REPORT/'RESOURCE_FORECAST.json')['status']=='PASS'
 assert read(REPORT/'IMPLEMENTATION_FROZEN.json')['training_hashes']==SOURCE_HASHES_AT_IMPORT
 start=time.monotonic();cpu=cpu_seconds();counts=counter();prior={};local=start_local=last_saved=0;status='FAIL';error=None
 try:
  if (root/'latest.pt').exists():m,o,local,prior=restore(root/'latest.pt',b,arm,s)
  elif target==1500:m,o=initialize_stream(b,arm,s,rep)
  else:
   pre=task['prefix'];binding=read(RUN/'contracts/REUSE_BINDINGS.json')['prefixes'].get(pre)
   path=Path(binding['path']) if binding else task_path(pre)/'latest.pt'
   if not binding:assert read(task_path(pre)/'DONE.json')['local_update']==1500
   m,o,local,_=restore(path,b,arm,s,True);assert local==1500;m.replicate=rep;m.continuation_seed=int(task['continuation_seed']);o=m.configure(2)
   write(root/'PREFIX_FORK.json',dict(prefix=pre,sha256=sha(path),model=p4._state_digest(m.state_dict()),RNG=state_digest(m,o),parent_seed=s,continuation_seed=m.continuation_seed,replicate=rep,physical_prefix_billed=0))
  assert m.parent_seed==s and m.replicate==rep and m.continuation_seed==int(task['continuation_seed'])
  from assets import parents
  pp,rr=parents(b,s,'C2');write(root/f'INITIALIZATION_{os.getpid()}.json',dict(parent_sha256=sha(pp),recovery_sha256=sha(rr),source_hashes=SOURCE_HASHES_AT_IMPORT,started_local=local,parent_update=15001))
  before=frozen_digest(m);start_local=last_saved=local;write(root/f'PHASE{m.phase}_PARAMETERS.json',m.lock);dl=loader(m.continuation_seed,local,target)
  for batch in dl:
   if STOP_REQUESTED:save(root/'latest.pt',m,o,local,counts,prior);last_saved=local;raise InterruptedError('owned graceful checkpoint')
   local+=1;result=step(m,o,batch,local,counts,probe=False)
   if local%100==0 or local in (1,1500,1501,3000,target):append(root/'TRAINING_CURVE.jsonl',dict(local_update=local,parent_seed=s,replicate=rep,continuation_seed=m.continuation_seed,sampler_zero_index=15001+local-1,gamma_index=m.continuation_seed*100000+15001+local,result=result,counts=dict(counts),utc=now()))
   if local in (1,1501,3000,6000):append(root/'RNG_PAIRING.jsonl',dict(local=local,batch_image=p4._tensor_sha256(batch['image']),batch_grade=p4._tensor_sha256(batch['grade']),state=state_digest(m,o),stream=m.continuation_seed))
   if local%10==0:write(root/'PROGRESS.json',dict(pid=os.getpid(),local=local,counts=counts,gpu_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu))
   if local%500==0 or local==target:
    save(root/'latest.pt',m,o,local,counts,prior);last_saved=local
    if local in (1500,3000):p4._atomic_torch_save(dict(model=m.state_dict(),backbone=b,arm=arm,seed=s,parent_seed=s,replicate=rep,continuation_seed=m.continuation_seed,local_update=local,phase=m.phase),root/f'stage_{local}.pt')
   if torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('actual VRAM cap')
  del dl;assert local==target and frozen_digest(m)==before;m.check_bn()
  p4._atomic_torch_save(dict(model=m.state_dict(),backbone=b,arm=arm,seed=s,parent_seed=s,replicate=rep,continuation_seed=m.continuation_seed,local_update=local,phase=m.phase),root/'final.pt')
  write(root/'DONE.json',dict(sha256=sha(root/'final.pt'),local_update=local,task=task,parameter_lock=m.lock,counts={k:counts[k]+prior.get(k,0) for k in counts},recovery_sha256=sha(root/'latest.pt'),exposure=m.exposure,source_hashes=SOURCE_HASHES_AT_IMPORT,utc=now()));status='PASS'
 except BaseException as e:error=repr(e);traceback.print_exc();raise
 finally:write(root/f'COST_{os.getpid()}.json',dict(status=status,error=error,counts=counts,gpu_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu,task=task,start_local=start_local,last_saved_local=last_saved,retained_formal_updates=max(0,last_saved-start_local),source_hashes_at_import=SOURCE_HASHES_AT_IMPORT,utc=now()))
