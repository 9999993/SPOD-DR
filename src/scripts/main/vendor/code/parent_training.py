"""Accounted, atomic checkpoints; exact shared sampler and separate views."""
from common import *
from parent_models import Model,p4
from parent_objectives import losses
from gradient_composition import backward
import torch,numpy as np,resource,traceback,subprocess
SOURCE_HASHES_AT_IMPORT={p.name:sha(p) for p in (REPORT/'code').glob('*.py')}
def initialization_receipt(m,recovery=None,prefix=None):
 from assets import parents,public
 bindings={'public':public()[m.b],'seed':m.seed,'parent_new':m.parent_new,'native_tensor_origins':m.initialization,'source_hashes_at_import':SOURCE_HASHES_AT_IMPORT}
 if not m.parent_new:
  parent,rr=parents(m.b,m.seed,'normal' if m.arm=='Nplus' else 'C2')
  bindings['parent']={'path':str(parent),'sha256':sha(parent),'recovery_path':str(rr),'recovery_sha256':sha(rr)}
 for key,p in (('resumed_checkpoint',recovery),('shared_prefix',prefix)):
  if p is not None:bindings[key]={'path':str(p),'sha256':sha(p)}
 if hasattr(m,'d'):bindings['D_initial_tensors']={k:p4._tensor_sha256(v) for k,v in m.d.state_dict().items()}
 return bindings
def precision():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
def cpu_seconds():return sum(resource.getrusage(w).ru_utime+resource.getrusage(w).ru_stime for w in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))
def loader(s,start,stop,parent=False,rs=None):
 offset=0 if parent else 15001
 sampler=p4.P4BatchSampler(len(rs) if rs is not None else len(manifest('TRAIN')),s,offset+start,offset+stop)
 return torch.utils.data.DataLoader(p4.P4ManifestDataset(rs or manifest('TRAIN')),batch_sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(s+710000))
def snapshot(m,o,local,counts):
 return {'model':m.state_dict(),'optimizer':o.state_dict(),'rng':p4._rng_state(),'mix_rng':m.mix_cpu.get_state(),'local_update':local,'next_update':local,'global_next_zero_index':(0 if m.parent_new else 15001)+local,'phase':m.phase,'backbone':m.b,'arm':m.arm,'seed':m.seed,'parent_new':m.parent_new,'sampler':{'size':len(manifest('TRAIN')),'next_zero_index':(0 if m.parent_new else 15001)+local},'parameter_lock':m.lock,'counts':dict(counts),'code_hashes':{p.name:sha(p) for p in (REPORT/'code').glob('*.py')}}
def save(path,m,o,local,counts):p4._atomic_torch_save(snapshot(m,o,local,counts),Path(path))
def restore(path,b,arm,s,parent=False):
 ck=torch.load(path,map_location='cpu',weights_only=False);m=Model(b,arm,s,parent).cuda();o=m.configure(ck['phase']);m.load_state_dict(ck['model'],strict=True);o.load_state_dict(ck['optimizer']);p4._restore_rng_state(ck['rng']);m.mix_cpu.set_state(ck['mix_rng']);return m,o,ck['local_update'],ck['counts']
def prefix_path(b,s,arm):
 if arm=='Nplus':return task_path(f'head__{b}__{s}__Nplus')/'final.pt'
 group='GAUSS_G' if arm in ('P','C2_SUM') else 'NATIVE_C'
 if b=='B3':return C17/'checkpoints'/f'{"G" if group=="GAUSS_G" else "B0"}__{s}'/'phase1500.pt'
 return task_path(f'prefix__{b}__{s}__{group}')/'final.pt'
def initialize(b,arm,s,kind):
 parent=kind=='parent_new';m=Model(b,arm,s,parent).cuda();m.restore_parent_rng();o=m.configure(1);local=0
 if kind=='suffix_new':
  path=prefix_path(b,s,arm);ck=torch.load(path,map_location='cpu',weights_only=False)
  assert ck['local_update']==1500 and ck['sampler']['next_zero_index']==16501
  own=m.state_dict();loaded=ck['model'];extra=set(own)-set(loaded)
  allowed={k for k in own if k.startswith('d.') or k in ('gate','gamma0','beta0')}
  if not extra<=allowed or set(loaded)-set(own):raise RuntimeError('prefix key mismatch '+str(extra))
  own.update(loaded);m.load_state_dict(own,strict=True);p4._restore_rng_state(ck['rng']);local=1500;o=m.configure(2)
 return m,o,local
def step(m,o,batch,local,counts,probe=False):
 x=batch['image'].cuda(non_blocking=True);y=batch['grade'].cuda(non_blocking=True);aux=None
 if not (m.parent_new and not m.full):
  _,_,orbit,calibrate=p4._bootstrap_old_sources()
  # C17's actual sampler uses zero-based indices, orbit uses update number.
  orbit_index=m.seed*100000+(0 if m.parent_new else 15001)+local
  with torch.no_grad():aux=p4._normalize_from_01(orbit(p4._denormalize(x),torch.ones((len(x),1,512,512),device=x.device,dtype=x.dtype),'gamma',orbit_index,calibrate(seed=m.seed)).image)
 o.zero_grad(set_to_none=True);a=m(x);b=m(aux,aux=True) if aux is not None else a
 counts['training_forward']+=len(x)*(2 if aux is not None else 1)
 main,other,parts=losses(m,a,b,y,batch['source'],p4);total=main+other
 if not torch.isfinite(total):raise FloatingPointError('nonfinite loss')
 log={}
 if m.arm in ('P','C2_SUM') and m.phase==2:
  log=backward(main,other,m,m.arm=='P',probe);counts['reverse_sweeps']+=2
 else:total.backward();counts['reverse_sweeps']+=1
 grads={}
 if probe:
  for n,p in m.named_parameters():
   if p.requires_grad:
    if p.grad is None:
     if n in m.parent_legal_none:grads[n]={'none':True,'declared_consumer':'unused historical parent module'};continue
     raise RuntimeError('missing active gradient '+n)
    if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite '+n)
    grads[n]={'nonzero':bool(torch.count_nonzero(p.grad)),'finite':True,'norm':float(p.grad.double().norm())}
 norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True);o.step();m.project_gate();counts['updates']+=1
 return float(total.detach()),{k:float(v.detach()) for k,v in parts.items()},float(norm),log,grads
def frozen_digest(m):
 active={n for n,p in m.named_parameters() if p.requires_grad}
 actual={n:x.training for n,x in m.named_modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)}
 assert actual==m.lock['BN'],'BN train mode differs from the static phase consumer lock'
 mutable={n for n,training in actual.items() if training}
 return p4._state_digest({n:v for n,v in m.state_dict().items() if n not in active and not any(n.startswith(b+'.') for b in mutable)})
def fresh_resume(path,batch_path,out):
 precision();c=torch.load(path,map_location='cpu',weights_only=False);m,o,pos,counts=restore(path,c['backbone'],c['arm'],c['seed'],c['parent_new']);batch=torch.load(batch_path,map_location='cpu',weights_only=False)
 step(m,o,batch,pos+1,counts,True)
 write(out,{'model_digest':p4._state_digest(m.state_dict()),'optimizer_digest':optimizer_digest(o),'rng_digest':p4._state_digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'mix':m.mix_cpu.get_state()}),'counts':counts})
def optimizer_digest(o):
 state={}
 for i,p in enumerate([p for g in o.param_groups for p in g['params']]):
  for k,v in o.state.get(p,{}).items():
   if torch.is_tensor(v):state[f'{i}:{k}']=v
 return p4._state_digest(state)
def parent_protocol_audit(m,batch,d,counts):
 """Compare the consumed actual runner against original Arm, not a loss-name stub."""
 import inspect
 assert m.parent_new and m.full
 state={k:v.detach().clone() for k,v in m.state_dict().items()};rng=p4._rng_state()
 x=batch['image'].cuda();y=batch['grade'].cuda()
 _,_,orbit,calibrate=p4._bootstrap_old_sources()
 with torch.no_grad():aux=p4._normalize_from_01(orbit(p4._denormalize(x),torch.ones((len(x),1,512,512),device=x.device,dtype=x.dtype),'gamma',m.seed*100000+1,calibrate(seed=m.seed)).image)
 params=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
 def one(actual):
  m.load_state_dict(state);p4._restore_rng_state(rng)
  if actual:
   a=m(x);b=m(aux,aux=True);loss,_,parts=losses(m,a,b,y,batch['source'],p4)
   outputs={'clean_probs':a['historical_grade_probs'],'aux_probs':b['historical_grade_probs'],'clean_mu':a['mu'],'aux_mu':b['mu']}
  else:
   xx=(p4._denormalize(x)-.5)/.5 if m.b=='InceptionV3' else x
   vv=(p4._denormalize(aux)-.5)/.5 if m.b=='InceptionV3' else aux
   a=m.inner(xx);b=m.inner(vv);loss,parts,_=p4.p4_loss('C2',a,y,batch['source'],b)
   outputs={'clean_probs':a['grade_probs'],'aux_probs':b['grade_probs'],'clean_mu':a['severity_score'],'aux_mu':b['severity_score']}
  counts['extra_forward']+=2*len(x)
  values={k:v.detach().cpu() for k,v in outputs.items()};values['loss']=loss.detach().cpu();values.update({'part_'+k:v.detach().cpu() for k,v in parts.items()})
  grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True);counts['reverse_sweeps']+=1
  digest={n:None if g is None else p4._tensor_sha256(g) for (n,_),g in zip(params,grads)}
  for (n,_),g in zip(params,grads):
   if g is None:assert n in m.parent_legal_none,('undeclared original-parent None',n)
   else:assert torch.isfinite(g).all(),n
  return values,digest,p4._state_digest({n:v for n,v in m.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n}),p4._state_digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()})
 try:
  actual=one(True);original=one(False)
  assert actual[0].keys()==original[0].keys()
  for k in actual[0]:assert torch.equal(actual[0][k],original[0][k]),('original parent operand/loss mismatch',k)
  assert actual[1]==original[1],'original parent parameter-gradient mismatch'
  assert actual[2:]==original[2:],'original parent BN/RNG mismatch'
  decoder=m.inner.base._probs.__globals__['compute_ordinal_gaussian_probs'];files={str(Path(inspect.getsourcefile(f))):sha(inspect.getsourcefile(f)) for f in (decoder,p4.p4_loss,m.inner.forward,m.inner._build_from_features)}
  write(d/'ORIGINAL_PARENT_PROTOCOL.json',{'status':'PASS_ACTUAL_FULL_GRAPH_BITWISE','utc':now(),'configured_CLI':'run.py qualify --backbone '+m.b+' --arm C2 --parent; run.py train-task --task parent__'+m.b+'__2101__C2','called_actual':'training.step -> Model.forward -> objectives.losses -> bound original p4_loss(C2)','called_reference':'same initialized historical Arm.forward -> bound original p4_loss(C2)','source_hashes_at_import':SOURCE_HASHES_AT_IMPORT,'original_callable_file_sha256':files,'operands_losses_parts_bitwise':True,'every_consumed_parameter_gradient_bitwise':True,'BN_and_RNG_bitwise':True,'optimizer_updates':0,'TRAIN_image_view_forwards':4*len(x),'reverse_traversals':2,'gradient_hashes':actual[1],'initialization_sha256':sha(d/'INITIALIZATION.json')})
 finally:
  m.load_state_dict(state);p4._restore_rng_state(rng);del state

def qualify(b,arm,s=2101,parent=False):
 precision();d=RUN/'qualification'/jid(b,s,arm);d.mkdir(parents=True,exist_ok=True);start=time.monotonic();cpu=cpu_seconds();counts={'updates':0,'training_forward':0,'reverse_sweeps':0,'extra_forward':0};status='FAIL';error=None;phases=[]
 try:
  m=Model(b,arm,s,parent).cuda();m.restore_parent_rng()
  write(d/'INITIALIZATION.json',initialization_receipt(m))
  rs=manifest('TRAIN');panel=sorted(rs,key=lambda r:hashlib.sha256(str(r.get('image_id',r['cache_relative_path'])).encode()).digest())[:32]
  write(d/'PANEL.json',{'manifest_sha256':ROLES['TRAIN'][2],'rows_hash':hashlib.sha256(json.dumps([r['row_index'] for r in panel]).encode()).hexdigest(),'images':len(panel),'selection':'lowest SHA256(image_id or registered cache relative path); no outcome selection'})
  batches=list(torch.utils.data.DataLoader(p4.P4ManifestDataset(panel),batch_size=16,shuffle=False,num_workers=0))
  for phase in ((1,2) if not parent else (1,)):
   o=m.configure(phase);write(d/f'PHASE{phase}_PARAMETERS.json',m.lock)
   if parent and m.full:parent_protocol_audit(m,batches[0],d,counts)
   before=frozen_digest(m);torch.cuda.reset_peak_memory_stats();times=[]
   for i in range(3):
    torch.cuda.synchronize();t=time.monotonic();values=step(m,o,batches[i%2],1501+i if phase==2 else 1+i,counts,True);torch.cuda.synchronize();times.append(time.monotonic()-t)
   assert frozen_digest(m)==before,'frozen tensor/BN changed'
   if hasattr(m,'gate') and m.gate.requires_grad:assert values[-1]['gate']['nonzero'],'zero gate gradient'
   save(d/'resume.pt',m,o,1503 if phase==2 else 3,counts)
   batch=batches[1];torch.save(batch,d/'batch.pt');step(m,o,batch,1504 if phase==2 else 4,counts,True)
   expected={'model_digest':p4._state_digest(m.state_dict()),'optimizer_digest':optimizer_digest(o),'rng_digest':p4._state_digest({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'mix':m.mix_cpu.get_state()})}
   write(d/f'PHASE{phase}_EXPECTED_RESUME.json',expected)
   memory={'reserved':torch.cuda.max_memory_reserved(),'allocated':torch.cuda.max_memory_allocated(),'total':torch.cuda.get_device_properties(0).total_memory}
   assert memory['reserved']<=.9*memory['total'],'insufficient 10% VRAM headroom'
   del m,o;torch.cuda.empty_cache()
   cmd=[sys.executable,str(REPORT/'code/run.py'),'resume-probe','--snapshot',str(d/'resume.pt'),'--batch',str(d/'batch.pt'),'--output',str(d/'FRESH_RESUME.json')]
   subprocess.run(cmd,check=True)
   actual=read(d/'FRESH_RESUME.json');assert all(actual[k]==v for k,v in expected.items()),'fresh process resume mismatch'
   counts['updates']+=1;counts['training_forward']+=32 if not parent or arm=='C2' else 16;counts['reverse_sweeps']+=2 if phase==2 and arm in ('P','C2_SUM') else 1
   phases.append({'phase':phase,'seconds_per_step':times,'memory':memory,'frozen_bitwise':True,'fresh_process_resume_bitwise':True,'loss':values[0],'parts':values[1],'preclip_norm':values[2],'projection':values[3],'gradients':values[4]})
   m,o,_,_=restore(d/'resume.pt',b,arm,s,parent)
  status='PASS'
 except BaseException as e:error=repr(e);traceback.print_exc()
 finally:
  write(d/'RECEIPT.json',{'status':status,'backbone':b,'arm':arm,'seed':s,'counts':counts,'forward':counts.get('extra_forward',0),'phases':phases,'error':error,'gpu_seconds':time.monotonic()-start,'cpu_seconds':cpu_seconds()-cpu,'utc':now(),'scope':'discarded real TRAIN qualification','BN_policy':'registered frozen prefix eval; registered suffix train with two view forwards; dual backward adds no forwards','source_hashes_at_import':SOURCE_HASHES_AT_IMPORT,'source_hashes_at_receipt':{p.name:sha(p) for p in (REPORT/'code').glob('*.py')}})
  temporary=[]
  for name in ('resume.pt','batch.pt'):
   path=d/name
   if path.exists():
    temporary.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size});path.unlink()
  write(d/'TEMPORARY_PROBE_DISPOSAL.json',{'utc':now(),'discarded_qualification_tensors':temporary,'receipt_sha256':sha(d/'RECEIPT.json'),'formal_model_or_input_deleted':False})
 if status!='PASS':raise RuntimeError(error)
def train(task):
 precision();b=task['backbone'];s=int(task['seed']);arm=task['arm'];kind=task['kind'];root=task_path(task['task_id']);root.mkdir(parents=True,exist_ok=True)
 if (root/'INVALID_CONTRACT.json').exists():raise RuntimeError('invalid protocol asset must remain quarantined and cannot be reused as a completed model')
 if (root/'DONE.json').exists():assert sha(root/'final.pt')==read(root/'DONE.json')['sha256'];return
 start=time.monotonic();cpu=cpu_seconds();counts={'updates':0,'training_forward':0,'reverse_sweeps':0,'extra_forward':0};status='FAIL';error=None
 start_local=None;last_saved_local=None
 try:
  if not read(REPORT/'QUALIFICATION.json')['status']=='PASS':raise RuntimeError('qualification not complete')
  if (root/'latest.pt').exists():m,o,local,_=restore(root/'latest.pt',b,arm,s,kind=='parent_new')
  else:m,o,local=initialize(b,arm,s,kind)
  start_local=local;last_saved_local=local
  write(root/f'INITIALIZATION_{os.getpid()}.json',initialization_receipt(m,root/'latest.pt' if (root/'latest.pt').exists() else None,prefix_path(b,s,arm) if kind=='suffix_new' else None))
  target=15001 if kind=='parent_new' else 1500 if kind in ('prefix_new','head_new') else 6000
  for batch in loader(s,local,target,parent=kind=='parent_new'):
   if getattr(sys.modules.get('__main__'),'STOP_REQUESTED',False):
    save(root/'latest.pt',m,o,local,counts);last_saved_local=local
    raise InterruptedError('graceful stop preserved the last complete optimizer/RNG/BN/sampler state')
   local+=1
   if kind!='parent_new' and local==1501:o=m.configure(2)
   loss,parts,norm,grad,_=step(m,o,batch,local,counts,probe=local in (1,1501))
   if local%100==0 or local in (1,1501):
    record={'local_update':local,'sampler_zero_index':(0 if m.parent_new else 15001)+local-1,'orbit_update_index':(0 if m.parent_new else 15001)+local,'loss':loss,'parts':parts,'preclip_norm':norm,'projection':grad,'counts':dict(counts),'utc':now()}
    if hasattr(m,'gate'):record['gate']={'min':float(m.gate.min()),'max':float(m.gate.max()),'mean':float(m.gate.mean())}
    append(root/'TRAINING_CURVE.jsonl',record);write(root/'PROGRESS.json',{'local':local,'counts':counts,'gpu_seconds':time.monotonic()-start,'cpu_seconds':cpu_seconds()-cpu})
   if local%500==0 or local==target:
    save(root/'latest.pt',m,o,local,counts);last_saved_local=local
   if local<target and getattr(sys.modules.get('__main__'),'STOP_REQUESTED',False):
    save(root/'latest.pt',m,o,local,counts);last_saved_local=local
    raise InterruptedError('graceful stop preserved the last complete optimizer/RNG/BN/sampler state')
   if torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('VRAM hard cap')
  if kind in ('prefix_new','head_new'):save(root/'final.pt',m,o,local,counts)
  else:p4._atomic_torch_save({'model':m.state_dict(),'backbone':b,'arm':arm,'seed':s,'local_update':local,'phase':m.phase},root/'final.pt')
  write(root/'DONE.json',{'sha256':sha(root/'final.pt'),'local_update':local,'task':task,'parameter_lock':m.lock,'utc':now()});status='PASS'
 except BaseException as e:error=repr(e);traceback.print_exc();raise
 finally:write(root/f'COST_{os.getpid()}.json',{'status':status,'error':error,'counts':counts,'gpu_seconds':time.monotonic()-start,'cpu_seconds':cpu_seconds()-cpu,'utc':now(),'task':task,'start_local':start_local,'last_saved_local':last_saved_local,'retained_formal_updates':max(0,last_saved_local-start_local) if start_local is not None else 0,'source_hashes_at_import':SOURCE_HASHES_AT_IMPORT})
