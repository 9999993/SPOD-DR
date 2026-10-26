#!/usr/bin/env python3
"""S2 clean-R0 preparation and guarded world1 C0/C2 execution.

Never uses a historical manifest, model, threshold or evaluation default.
Formal execution uses the user-authorized, locally frozen training contract.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, random, resource, socket
import subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import psutil
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from torch import nn
import timm
from safetensors.torch import load_file
from clean_s2_reference import p4_loss
from clean_s2_vendor.tasks.diabetic_retinopathy.models.clinical_risk.crrl_model import ClinicalRiskRepresentationLearningModel
from clean_s2_vendor.tasks.diabetic_retinopathy.models.distributional_progression import distributional_progression_model as dpm
from clean_s2_vendor.tasks.diabetic_retinopathy.sparrow_dr.model import SparrowDRModel
from clean_s2_vendor.tasks.diabetic_retinopathy.sparrow_dr.severity import OrderedSeverityAxis
from clean_s2_vendor.tasks.diabetic_retinopathy.sparrow_dr.destyle import ResidualDestylizer
from clean_s2_vendor.tasks.diabetic_retinopathy.sparrow_dr.orbits import apply_orbit, calibrate_source_quantiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2].parent
DATA = ROOT.parents[1] / 'data_manifests'
PLAN = '910a4f6a32bc5c6b9d42a4f893bbc6f8833479d2'
BASE = '6109168c90ce79fb2d21937b92c921d75c253ebc'
DATA_HASHES = {
 'READY.json':'78fbe504e57490f5140561a3d874184fa16b0fa6be0d015dce9ca164c313508f',
 'FILE_SHA256.json':'296345aae26d4188b7c9b74dde8847dcc952ba091c086a7959ff87bbeed2809c',
 'train_manifest.csv':'e6757408c17a22bcc9672bbbd3084adfc5064113ecea1d0224a9e8c0d9e78ebc',
 'dev_manifest.csv':'0e3f01299ac7ab7dd1322166a2e15fdc200e8441661a93d790296d98850ff112',
 'threshold_manifest.csv':'8082f405faf1738d0483637a01e9a5152c15195c12bf88f3d9cf32ba78657109'}
CHANNELS = {'resnet50':(1024,2048), 'densenet121':(1024,1024), 'swin_tiny':(384,768)}
NODES = {'resnet50':['layer3','layer4'], 'densenet121':['features.denseblock3','features.norm5'], 'swin_tiny':['layers.2','layers.3 -> norm']}
RECIPE = dict(updates=15001, input_size=512,batch=16,world=1,accum=1,precision='FP32',tf32=False,
 optimizer='AdamW',lr=3e-5,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,clip=1.,scheduler=None,
 ema=False,swa=False,deterministic_algorithms=True,cudnn_deterministic=True,cublas_workspace_config=':4096:8',checkpoint='last',checkpoint_every=500,train_bn='native train()',eval_bn='eval()',
 sampler='PCG64 SeedSequence([seed,epoch,0x5034]); drop tail',steps_per_epoch=1499,
 augmentation='C0 cached image; C2 main cached image and gamma orbit; seed*100000+update; orbit config seed=seed',
 losses={'grade':1.,'ordinal':.2,'style':.1,'orbit':.2,'severity':.1},feature_dim=256,local_queries=6,attention_heads=4,
 pooling='gem',fusion_beta=1.,crrl_mode='consistency')

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest_json(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with tmp.open('w') as f: json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with tmp.open('wb') as f: torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def rng(): return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def set_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
 if r['cuda'] is not None:torch.cuda.set_rng_state(r['cuda'].cpu())
def live_child_cpu_s():
 total=0.
 for p in psutil.Process().children(recursive=True):
  try:c=p.cpu_times();total+=c.user+c.system
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 return total
def cpu_s():
 return sum(resource.getrusage(w).ru_utime+resource.getrusage(w).ru_stime for w in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))
def precision():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.set_default_dtype(torch.float32)
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
def gpu_records():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
 return [dict(zip(['index','uuid','name','memory_mib','used_mib'],[v.strip() for v in l.split(',')])) for l in s.splitlines()]
def models():return json.loads((HERE/'clean_s2_public_models.json').read_text())
def runtime_identity():
 import platform,importlib.metadata as md
 import timm.models.resnet as rn
 import timm.models.densenet as dn
 import timm.models.swin_transformer as sw
 return {'native_model_module_sha256':{p.__name__:sha(p.__file__) for p in (rn,dn,sw)},'torch_git_revision':torch.version.git_version,'python':platform.python_version(),'torch_cuda':torch.version.cuda,'libraries':{k:md.version(k) for k in ('torch','torchvision','timm','numpy','safetensors','psutil')},'torch_file_sha256':sha(torch.__file__),'timm_file_sha256':sha(timm.__file__)}
def code_identity():return {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py')) if p.name.startswith('clean_s2') or 'clean_s2_vendor' in str(p)}
def check_data():
 for n,h in DATA_HASHES.items():
  if sha(DATA/n)!=h:raise RuntimeError('data identity mismatch: '+n)
 if json.loads((DATA/'READY.json').read_text())['status']!='READY_READ_ONLY':raise RuntimeError('data not READY')
 receipt=ROOT.parent/'control'/'DATA_VALIDATION_COMPLETE.json'
 status='PENDING'
 if receipt.exists():
  r=json.loads(receipt.read_text());b=r['binding']
  same=(b['filesystem_id']==os.statvfs(DATA).f_fsid and b['st_dev']==DATA.stat().st_dev and b['st_ino']==DATA.stat().st_ino)
  if r['status']!='PASS' or b['path']!=str(DATA) or b['inventory_sha256']!=DATA_HASHES['FILE_SHA256.json'] or b['ready_sha256']!=DATA_HASHES['READY.json']:raise RuntimeError('S1 input binding mismatch')
  if same:status={'path':str(receipt),'sha256':sha(receipt),'reused_same_mount':True}
  else:
   local=ROOT/'LOCAL_DATA_VALIDATION.json'
   lr=json.loads(local.read_text())
   if lr['status']!='PASS' or lr['root']!=str(DATA) or lr['filesystem_id']!=os.statvfs(DATA).f_fsid or lr['st_dev']!=DATA.stat().st_dev or lr['st_ino']!=DATA.stat().st_ino or lr['inventory_sha256']!=DATA_HASHES['FILE_SHA256.json']:raise RuntimeError('local full validation mismatch')
   status={'S1_path':str(receipt),'S1_sha256':sha(receipt),'reused_same_mount':False,'local_path':str(local),'local_sha256':sha(local)}
 return {'root':str(DATA),'hashes':DATA_HASHES,'mount':subprocess.check_output(['findmnt','-T',str(DATA),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True).strip(),'stat_dev':DATA.stat().st_dev,'full_scan_performed':False,'S1_scan_receipt':status}

class Pyramid(nn.Module):
 def __init__(self,name):
  super().__init__();self.name=name;meta=models()[name];model_name=meta['model_id'].split('/')[1]
  kw={'img_size':512} if name=='swin_tiny' else {}
  self.backbone=timm.create_model(model_name,pretrained=False,**kw)
  from training_resources import initialization_for
  backbone={'resnet50':'ResNet50','swin_tiny':'SwinTiny'}.get(name)
  if backbone is None:raise ValueError('Backbone is not exposed by the release training entry')
  wp=Path(initialization_for(backbone)['path'])
  if sha(wp)!=meta['artifacts']['model.safetensors']['sha256']:raise RuntimeError('public weight identity mismatch')
  state=load_file(str(wp));self.excluded_source_buffers=[]
  if name=='swin_tiny':
   self.excluded_source_buffers=[k for k in state if 'relative_position_index' in k or 'attn_mask' in k]
   state={k:v for k,v in state.items() if k not in self.excluded_source_buffers}
  self.backbone.load_state_dict(state,strict=True)
  loaded=self.backbone.state_dict()
  if set(loaded)!=set(state) or any(not torch.equal(v,loaded[k]) for k,v in state.items()):raise RuntimeError('consumed public tensor mismatch')
  self.public_state_sha256=state_hash(loaded);self.backbone.reset_classifier(0)
 def forward(self,x):
  b=self.backbone
  if self.name=='resnet50':
   x=b.maxpool(b.act1(b.bn1(b.conv1(x))));x=b.layer2(b.layer1(x));s3=b.layer3(x);s4=b.layer4(s3)
  elif self.name=='densenet121':
   s3=None
   for n,m in b.features.named_children():
    x=m(x)
    if n=='denseblock3':s3=x
   s4=x
  else:
   x=b.patch_embed(x);s3=None
   for i,m in enumerate(b.layers):
    x=m(x)
    if i==2:s3=x.permute(0,3,1,2).contiguous()
   s4=b.norm(x).permute(0,3,1,2).contiguous()
  expected=CHANNELS[self.name]
  for s,c,stride in zip((s3,s4),expected,(16,32)):
   if s.ndim!=4 or s.shape[1]!=c:raise RuntimeError('native channel/layout mismatch')
  return s3,s4

class Arm(SparrowDRModel):
 def __init__(self,base,arm,channels):
  nn.Module.__init__(self);self.base=base;self.arm_id=arm;self.method_enabled=arm=='C2'
  self.axis_stage3=OrderedSeverityAxis(channels[0]);self.axis_stage4=OrderedSeverityAxis(channels[1])
  self.destylize_stage3=ResidualDestylizer(channels[0]);self.destylize_stage4=ResidualDestylizer(channels[1])
 def forward(self,x):
  if self.arm_id=='C0':
   o=self.base(x);o['q2']=o['grade_probs'][:,2:].sum(dim=1);return o
  s3,s4=self.base.feature_pyramid(x);out=self._build_from_features(s3,s4)
  out['contrastive_stage3']=s3;out['contrastive_stage4']=s4;out['q2']=out['grade_probs'][:,2:].sum(dim=1)
  return out

def build(name,arm,seed):
 if name not in CHANNELS or arm not in ('C0','C2'):raise ValueError('unassigned backbone/arm')
 seed_all(seed);pyramid=Pyramid(name);c3,c4=CHANNELS[name]
 with patch.object(dpm,'_build_feature_pyramid',return_value=(pyramid,c3,c4)):
  base=ClinicalRiskRepresentationLearningModel(pretrained=False,backbone_name=name,backbone_img_size=512,
   pooling='gem',local_query_count=6,feature_dim=256,attention_heads=4,mode='local_global',beta=1.,crrl_mode='consistency',risk_hidden_dim=256,risk_embedding_dim=256)
 m=Arm(base,arm,(c3,c4))
 receipt={'public':models()[name],'public_consumed_values_sha256':pyramid.public_state_sha256,'excluded_source_buffers':pyramid.excluded_source_buffers,
  'swin_512_adaptation':'recompute nonpersistent window indices/attention masks at 512 with window7; no learned tensor interpolation',
  'model_state_sha256':state_hash(m.state_dict()),'backbone_state_sha256':state_hash(pyramid.state_dict()),
  'common_head_state_sha256':state_hash({k:v for k,v in base.state_dict().items() if not k.startswith('feature_pyramid.')}),
  'modules_initialized_in_same_order':True,'native_channels':[c3,c4],'nodes':NODES[name],
  'native_layout':'NHWC' if name=='swin_tiny' else 'NCHW','consumer_layout':'NCHW',
  'adaptation':'existing local 1x1 input projection and global input linear use native channels; axis and destylizer native widths; no extra modules',
  'parameters':sum(p.numel() for p in m.parameters()),'backbone_parameters':sum(p.numel() for p in pyramid.parameters()),
  'train_bn_count':sum(isinstance(p,nn.modules.batchnorm._BatchNorm) for p in m.modules()),
  'batchnorm_defaults':{n:{'eps':p.eps,'momentum':p.momentum,'track_running_stats':p.track_running_stats} for n,p in m.named_modules() if isinstance(p,nn.modules.batchnorm._BatchNorm)},
  'dropout_defaults':{n:{'type':type(p).__name__,'p':getattr(p,'p',None),'drop_prob':getattr(p,'drop_prob',None)} for n,p in m.named_modules() if isinstance(p,nn.Dropout) or type(p).__name__=='DropPath'}}
 return m,receipt

class Dataset(torch.utils.data.Dataset):
 def __init__(self):
  with (DATA/'train_manifest.csv').open(newline='') as f:self.rows=list(csv.DictReader(f))
  if len(self.rows)!=23987 or any(r['role']!='TRAIN' for r in self.rows):raise RuntimeError('TRAIN role/count mismatch')
  required={'cache_relative_path','grade','source'}
  if not required.issubset(self.rows[0]):raise RuntimeError('new manifest columns missing')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];p=(DATA/r['cache_relative_path']).resolve()
  if not p.is_relative_to(DATA.resolve()):raise RuntimeError('cache path escapes clean bundle')
  x=np.load(p,allow_pickle=False)
  if x.shape!=(3,512,512) or x.dtype!=np.float32:raise RuntimeError('clean cache shape/dtype mismatch')
  return {'image':torch.from_numpy(x),'grade':int(r['grade']),'source':r['source']}
class Sampler:
 def __init__(self,size,seed,start,stop):self.size=size;self.seed=seed;self.start=start;self.stop=stop
 def __len__(self):return self.stop-self.start
 def __iter__(self):
  last=-1;perm=None;n=self.size//16
  for u in range(self.start,self.stop):
   epoch=u//n
   if epoch!=last:perm=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,epoch,0x5034]))).permutation(self.size);last=epoch
   off=(u%n)*16;yield perm[off:off+16].tolist()
def second_view(x,seed,u):
 mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
 im=(x*std+mean).clamp(0,1);fov=torch.ones((len(x),512,512),dtype=torch.bool,device=x.device)
 v=apply_orbit(im,fov,'gamma',seed*100000+u,calibrate_source_quantiles(seed=seed)).image
 return (v-mean)/std
def loss_step(m,x,grade,source,seed,u):
 out=m(x);v=m(second_view(x,seed,u)) if m.arm_id=='C2' else None
 return p4_loss(m.arm_id,out,grade,source,v),out,v

def jobid(n,a,s):return f'cleanr0__{n}__{a}__{s}'
def prepare():
 check=check_data();write_json(ROOT/'DATA_IDENTITY.json',check);write_json(ROOT/'DEVICE_IDENTITY.json',{'hostname':socket.gethostname(),'gpus':gpu_records(),'utc':now()})
 lock=ROOT/'lock_visibility_probe';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 try:
  try:os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);raise RuntimeError('O_EXCL broken')
  except FileExistsError:pass
 finally:lock.unlink()
 rows=[]
 for n in CHANNELS:
  for a in ('C0','C2'):
   for s in (2101,2102,2103):
    config={'job_id':jobid(n,a,s),'owner':'S2','backbone':n,'arm':a,'seed':s,'recipe':RECIPE,'views':1 if a=='C0' else 2,
     'data':check,'runtime':runtime_identity(),'public_init':models()[n],'source_hashes':code_identity(),'plan_commit':PLAN,'evidence_base':BASE,
     'consumers':{'grade':'stage3 -> local_branch / stage4 -> global_branch; D(x) for C2','ordinal':'native unnormalized stage3/4 -> cumulative BCE',
      'style':'native unnormalized full stage3/4 spatial mean -> L2 -> concat -> L2; cross-source same-grade',
      'severity':'fused_mu absolute difference across views','orbit':'symmetric grade KL; gradients both directions'},
     'status':'FROZEN_FOR_AUTHORIZED_CONTINUOUS_EXECUTION','evaluation':'S1 frozen evaluator and efficiency contract; no local metrics'}
    ch=digest_json(config);write_json(ROOT/'jobs'/config['job_id']/'CONFIG.json',config)
    rows.append({'job_id':config['job_id'],'config_sha256':ch,'backbone':n,'arm':a,'seed':s,'status':'NOT_STARTED'})
 write_json(ROOT/'JOB_MATRIX.json',rows);print('frozen',len(rows),'jobs; ready for authorized execution')

def synthetic():
 precision();t=time.monotonic();c=cpu_s();report={'started':now(),'pairs':{},'wiring':{},'gradients':{},'recovery':{}}
 for n,(c3,c4) in CHANNELS.items():
  for s in (2101,2102,2103):
   m,r=build(n,'C0',s);h=r['model_state_sha256'];del m
   m,r2=build(n,'C2',s);assert h==r2['model_state_sha256'];report['pairs'][f'{n}:{s}']={'full_state_equal':True,'state_sha256':h,'common_head_sha256':r2['common_head_state_sha256']}
   if s!=2101:del m;continue
   m.eval()
   with torch.no_grad():
    feat=m.base.feature_pyramid(torch.randn(1,3,512,512));assert [list(v.shape[-2:]) for v in feat]==[[32,32],[16,16]]
   report['wiring'][n]={**r2,'shapes':[list(v.shape) for v in feat]};del feat
   # Native feature leaves test exact method consumers without full-image CPU training.
   f3=torch.rand(4,c3,4,4,requires_grad=True);f4=torch.rand(4,c4,2,2,requires_grad=True)
   grade=torch.tensor([0,0,1,1]);sources=['EyePACS','APTOS','EyePACS','DeepDRiD_Set_A']
   captures={}
   hs=[m.base.local_branch.register_forward_pre_hook(lambda _,a:captures.update(local=a[0])),m.base.global_branch.register_forward_pre_hook(lambda _,a:captures.update(global_=a[0]))]
   def out(a,b):
    o=m._build_from_features(a,b);o.update(contrastive_stage3=a,contrastive_stage4=b);return o
   o=out(f3,f4);assert torch.equal(o['aux_cumulative_stage3'],m.axis_stage3.cumulative(f3));assert torch.equal(o['aux_cumulative_stage4'],m.axis_stage4.cumulative(f4));assert torch.equal(captures['local'],o['transformed_stage3']);assert torch.equal(captures['global_'],o['transformed_stage4'])
   assert torch.equal(o['transformed_stage3'],m.destylize_stage3(f3));assert torch.equal(o['residual_stage3'],f3)
   v3=(f3.detach()*.83+.02).requires_grad_();v4=(f4.detach()*.87+.03).requires_grad_();v=out(v3,v4)
   loss,parts,support=p4_loss('C2',o,grade,sources,v);assert support['style_anchors']==4
   targets={'grade':[m.destylize_stage3.gamma,m.destylize_stage4.gamma,m.base.local_branch.project.weight,m.base.global_branch.project[0].weight,f3,f4],
    'ord':[m.axis_stage3.u,m.axis_stage4.u,f3,f4],'style':[f3,f4],
    'orbit':[o['grade_probs'],v['grade_probs']], 'severity':[o['fused_mu'],v['fused_mu']]}
   grads={}
   for k,p in targets.items():
    g=torch.autograd.grad(parts[k],p,retain_graph=True,allow_unused=True);grads[k]=[None if x is None else float(x.abs().sum()) for x in g]
    assert all(x is not None and torch.isfinite(x).all() and x.abs().sum()>0 for x in g),(n,k,grads[k])
   report['gradients'][n]={'per_component_target_l1':grads,'style_anchors':4,'consumer_hooks_equal':True,'q2_definition':'sum grade_probs[:,2:5]; separate from screening_probability'}
   opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   path=ROOT/'synthetic'/n/'recovery.pt';save(path,{'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng(),'next_update':1,'sampler':{'size':23987,'seed':2101,'next_update':1},'identity':r2['model_state_sha256']})
   def step():
    opt.zero_grad(set_to_none=True);z=out(f3.detach(),f4.detach());w=out(v3.detach(),v4.detach());l,_,_=p4_loss('C2',z,grade,sources,w);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();return state_hash(m.state_dict()),float(l.detach())
   uninterrupted=step();ck=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);set_rng(ck['rng']);resumed=step();assert uninterrupted==resumed
   # Repeat one-step recovery with the actual C0 grade-only objective and registered inactive modules.
   m.arm_id='C0';opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
   def c0step():
    opt.zero_grad(set_to_none=True);z=m._build_from_features(f3.detach(),f4.detach());l,_,_=p4_loss('C0',z,grade,sources,None);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();return state_hash(m.state_dict()),float(l.detach())
   c0step();p0=path.with_name('c0_recovery.pt');save(p0,{'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng()});u0=c0step();ck=torch.load(p0,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);set_rng(ck['rng']);assert u0==c0step()
   report['recovery'][n]={'C0_bitwise_resume':True,'C2_bitwise_resume':True,'c2_state_bytes':path.stat().st_size,'scope':'CPU native-feature head/operator updates; full-backbone CUDA recovery checked separately in TRAIN preflight'}
   for h in hs:h.remove()
   del m,opt,ck,o,v,parts,loss
   write_json(ROOT/'SYNTHETIC_VALIDATION.json',report);print(n,'paired init, native shapes, all C2 gradients and recovery PASS',flush=True)
 report.update(wall_seconds=time.monotonic()-t,cpu_core_seconds=cpu_s()-c,status='PASS');write_json(ROOT/'SYNTHETIC_VALIDATION.json',report)

class Claim:
 def __init__(self,d,meta):self.p=d/'CLAIM.json';self.meta=meta
 def __enter__(self):
  self.started=time.monotonic();self.cpu0=cpu_s()
  self.p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(self.p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'w') as f:json.dump(self.meta,f);f.flush();os.fsync(f.fileno())
  return self
 def __exit__(self,typ,v,t):
  write_json(self.p.parent/'CLAIM_FINISHED.json',{'claim':self.meta,'ended':now(),'success':typ is None,'error':str(v) if v else None,'wall_seconds':time.monotonic()-self.started,'cpu_core_seconds':cpu_s()-self.cpu0})
  # Claim retained permanently; no stale-lease takeover or automatic deletion.

def job_config(name,arm,seed):
 p=ROOT/'jobs'/jobid(name,arm,seed)/'CONFIG.json';r=json.loads(p.read_text())
 if r['source_hashes']!=code_identity():raise RuntimeError('prepared source changed; rerun prepare before launch')
 return r,digest_json(r)
def load_release(path,expected):
 if not path or not expected or sha(path)!=expected:raise RuntimeError('training contract missing/hash mismatch')
 r=json.loads(Path(path).read_text())
 if not r.get('authorized') or r.get('server')!='S2' or r.get('source_hashes')!=code_identity():raise RuntimeError('training authorization/source mismatch')
 if subprocess.run(['git','-C',str(HERE),'merge-base','--is-ancestor',r.get('implementation_commit','INVALID'),'HEAD'],capture_output=True).returncode:raise RuntimeError('training implementation commit mismatch')
 for pin in r.get('peer_device_receipts',{}).values():
  if sha(pin['path'])!=pin['sha256']:raise RuntimeError('peer device receipt pin mismatch')
 pin=r['data_verification']
 if sha(pin['path'])!=pin['sha256']:raise RuntimeError('data verification pin mismatch')
 if r.get('limits')!={'gpu_hours':120,'cpu_core_hours':240,'incremental_gib':120,'engineering_train_updates':1800}:raise RuntimeError('training budget limits mismatch')
 if r.get('engineering_train_updates_used',1801)>1800:raise RuntimeError('engineering update pool exhausted')
 return r

def actual_budget_check(release,out,started,cpu0,meta,update):
 live={'meta':meta,'logical_update':update,'cpu_core_seconds':cpu_s()-cpu0+live_child_cpu_s(),'gpu_job_seconds':time.monotonic()-started,'utc':now()}
 write_json(out/'RESOURCE_LIVE.json',live)
 gpu=release['preparation_actual']['gpu_hours']*3600
 cpu=release['preparation_actual']['cpu_core_hours_measured_lower_bound']*3600
 for d in (ROOT/'formal').iterdir():
  if not d.is_dir():continue
  finished=d/'CLAIM_FINISHED.json';running=d/'RESOURCE_LIVE.json'
  if finished.exists():
   r=json.loads(finished.read_text());gpu+=r['wall_seconds'];cpu+=r['cpu_core_seconds']
  elif running.exists():
   r=json.loads(running.read_text());elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(r['meta']['utc'])).total_seconds()
   gpu+=max(elapsed,r['gpu_job_seconds']);cpu+=r['cpu_core_seconds']
 if gpu>=release['limits']['gpu_hours']*3600 or cpu>=release['limits']['cpu_core_hours']*3600:
  raise RuntimeError('actual cumulative S2 compute cap reached; retain last complete checkpoint')
 if update==0 or update%500==0:
  used=int(subprocess.check_output(['du','-sb',str(ROOT)],text=True).split()[0])
  if used>=release['limits']['incremental_gib']*1024**3:raise RuntimeError('actual S2 output cap reached')
 write_json(ROOT/'ACTUAL_RESOURCE_LIVE.json',{'utc':now(),'gpu_hours':gpu/3600,'cpu_core_hours_measured_lower_bound':cpu/3600,'limits':release['limits'],'forecast_gate':False})

def worker(name,arm,seed,gpu,updates,mode,resume=False,release=None):
 precision();check_data();gpus=gpu_records();g=next(v for v in gpus if int(v['index'])==gpu)
 # Scheduler rejects already occupied GPUs; an explicit UUID identifies the actual device for each claim.
 torch.cuda.set_device(gpu);device=torch.device('cuda',gpu);config,ch=job_config(name,arm,seed)
 if config['runtime']!=runtime_identity():raise RuntimeError('prepared runtime changed')
 if mode=='formal':
  if release is None or jobid(name,arm,seed) not in release['jobs']:raise RuntimeError('job not in unified release')
  if updates!=15001:raise RuntimeError('formal update count must remain 15001')
  if release.get('config_sha256',{}).get(jobid(name,arm,seed))!=ch:raise RuntimeError('release config hash mismatch')
 else:
  if not 3<=updates<=50:raise ValueError('TRAIN preflight must be 3..50 updates per path')
 out=ROOT/( 'formal' if mode=='formal' else 'preflight')/jobid(name,arm,seed)
 if resume and ((out/'DONE.json').exists() or (out/'MODEL_LOCK.json').exists()):raise RuntimeError('completed job cannot be resumed/retrained')
 if not resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
 meta=dict(host=socket.gethostname(),gpu_uuid=g['uuid'],gpu_index=gpu,pid=os.getpid(),utc=now(),config_sha256=ch,job_id=jobid(name,arm,seed),mode=mode)
 if resume:
  # User/operator must explicitly confirm original process is gone. Never silently steal a claim.
  claim_path=out/'CLAIM.json';prior=json.loads(claim_path.read_text())
  if prior['host']==socket.gethostname():
   try:os.kill(prior['pid'],0)
   except ProcessLookupError:pass
   else:raise RuntimeError('original claimed PID still exists')
  else:raise RuntimeError('cross-host recovery requires separately recorded original-process exit evidence')
  attempt=out/f'resume_{int(time.time())}';attempt.mkdir();claim_dir=attempt
 else:claim_dir=out
 with Claim(claim_dir,meta):
  started=time.monotonic();cpu0=cpu_s();torch.cuda.reset_peak_memory_stats(device)
  if mode=='formal':actual_budget_check(release,out,started,cpu0,meta,0)
  m,init=build(name,arm,seed);m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
  start=0;state_path=out/'recovery.pt';replay_updates=0
  bn0={k:int(v) for k,v in m.state_dict().items() if k.endswith('num_batches_tracked')}
  if resume:
   ck=torch.load(state_path,map_location=device,weights_only=False)
   if ck['config_sha256']!=ch or ck['data_hashes']!=DATA_HASHES or ck['sampler']['size']!=23987 or ck['sampler']['seed']!=seed or ck['sampler']['next_update']!=ck['next_update']:raise RuntimeError('recovery identity mismatch')
   m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);set_rng(ck['rng']);start=ck['next_update']
  else:seed_all(seed)
  ds=Dataset();gen=torch.Generator().manual_seed(seed^0x50443431);loader=torch.utils.data.DataLoader(ds,batch_sampler=Sampler(len(ds),seed,start,updates),num_workers=2,pin_memory=True,generator=gen)
  times=[];supports=0;losses=[];first_gradient=None;cuda_recovery=None;recovery_bytes=0
  for offset,b in enumerate(loader):
   u=start+offset+1;
   if mode=='formal' and (u==start+1 or u%50==0):
    actual_budget_check(release,out,started,cpu0,meta,u)
   tick=time.monotonic();m.train();opt.zero_grad(set_to_none=True)
   x=b['image'].to(device,non_blocking=True);y=b['grade'].to(device,non_blocking=True)
   if not torch.isfinite(x).all():raise FloatingPointError('nonfinite TRAIN input')
   (loss,parts,support),o,v=loss_step(m,x,y,list(b['source']),seed,u)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   loss.backward()
   if u==1:
    bnd={k:int(v)-bn0[k] for k,v in m.state_dict().items() if k in bn0}
    if any(v!=(1 if arm=='C0' else 2) for v in bnd.values()):raise RuntimeError('native BN view count mismatch')
    first_gradient={k:('NONE' if p.grad is None else 'ZERO' if not torch.count_nonzero(p.grad) else 'NONZERO') for k,p in m.named_parameters()}
    if not any(k.startswith('base.feature_pyramid') and val=='NONZERO' for k,val in first_gradient.items()):raise RuntimeError('backbone receives no gradient')
   norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-tick);supports+=support['style_anchors']
   losses.append({k:float(val.detach()) for k,val in parts.items()})
   if u%500==0 or u==updates or (mode=='preflight' and u==2):
    state={'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng(),'next_update':u,'config_sha256':ch,'data_hashes':DATA_HASHES,'sampler':{'size':len(ds),'seed':seed,'next_update':u,'steps_per_epoch':1499},'arm':arm,'seed':seed}
    save(state_path,state);recovery_bytes=state_path.stat().st_size
   if mode=='preflight' and u==3:
    # Replay the identical third physical update from the update-2 checkpoint; verify all parameters, BN and optimizer slots.
    expected=state_hash(m.state_dict());expected_loss=float(loss.detach());optim_expected=state_hash({str(i)+':'+k:v for i,d in opt.state_dict()['state'].items() for k,v in d.items() if torch.is_tensor(v)})
    ck=torch.load(state_path,map_location=device,weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);set_rng(ck['rng']);opt.zero_grad(set_to_none=True)
    (l,_,_),_,_=loss_step(m,x,y,list(b['source']),seed,u);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize();replay_updates+=1
    optim_resumed=state_hash({str(i)+':'+k:v for i,d in opt.state_dict()['state'].items() for k,v in d.items() if torch.is_tensor(v)})
    if expected!=state_hash(m.state_dict()) or expected_loss!=float(l.detach()) or optim_expected!=optim_resumed:
     write_json(out/'RECOVERY_MISMATCH.json',{'model_equal':expected==state_hash(m.state_dict()),'loss_before':expected_loss,'loss_after':float(l.detach()),'optimizer_equal':optim_expected==optim_resumed,'physical_updates':u-start+replay_updates})
     raise RuntimeError('CUDA TRAIN resume mismatch')
    cuda_recovery={'model_bn_bitwise_equal':True,'optimizer_bitwise_equal':True,'loss_equal':True,'replayed_update':3}
    del ck
   write_json(out/'HEARTBEAT.json',dict(meta,logical_update=u,utc=now()))
   del o,v,loss,parts
  if mode=='formal':
   save(out/'model_last.pt',{'model':m.state_dict(),'next_update':15001,'config_sha256':ch,'data_hashes':DATA_HASHES,'init':init});write_json(out/'MODEL_LOCK.json',{'sha256':sha(out/'model_last.pt'),'config_sha256':ch,'logical_update':15001,'utc':now()})
  rec={'status':'PREFLIGHT_PASS' if mode=='preflight' else 'TRAINING_DONE_PENDING_SHARED_EVAL','meta':meta,'init':init,'logical_updates':updates-start,'physical_updates':updates-start+replay_updates,
   'cpu_core_seconds':cpu_s()-cpu0,'wall_seconds':time.monotonic()-started,'gpu_job_seconds':time.monotonic()-started,
   'update_seconds':times,'steady_mean_seconds':float(np.mean(times[3:])) if len(times)>3 else None,'steady_p95_seconds':float(np.quantile(times[3:],.95)) if len(times)>3 else None,
   'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),
   'style_anchor_total':supports,'loss_components_first':losses[0] if losses else None,'loss_components_last':losses[-1] if losses else None,'first_gradients':first_gradient,
   'cuda_resume':cuda_recovery,'bn_counter_delta_first_update':bnd if start==0 else None,'recovery_bytes':recovery_bytes,'recipe':RECIPE,'efficiency_scope':'engineering TRAIN timings including transfer/forward/backward/clip/optimizer/sync; S1 efficiency measurement still required',
   'gpu_accounting_scope':'allocated job elapsed incl CPU setup/I/O; not claimed CUDA kernel-active time'}
  write_json(out/'RECEIPT.json',rec);write_json(out/'DONE.json',{'receipt_sha256':sha(out/'RECEIPT.json'),'utc':now(),'status':rec['status']});print(name,arm,rec['status'],rec['steady_mean_seconds'],flush=True)


def schedule(mode,updates,release_path=None,release_hash=None):
 release=load_release(release_path,release_hash) if mode=='formal' else None
 if mode=='formal':
  load_release(release_path,release_hash)
  if not release.get('continuous_execution_authorized'):raise RuntimeError('continuous user authorization missing')
 candidates=gpu_records();devices=[int(g['index']) for g in candidates if float(g['used_mib'])<1024]
 if len(devices)!=4:raise RuntimeError('four assigned free GPUs required; device conflict needs recorded resolution')
 local={g['uuid'] for g in candidates}
 if release is not None:
  for role,pin in release['peer_device_receipts'].items():
   peer=json.loads(Path(pin['path']).read_text());overlap=local&{g['uuid'] for g in peer.get('gpus',peer.get('devices',[]))}
   if overlap:raise RuntimeError('released GPU UUID overlap with '+role)
 for role in ['S1','S3']:
  p=ROOT.parent/role/'DEVICE_IDENTITY.json'
  if not p.exists():p=ROOT.parent/role/'DEVICES.json'
  if p.exists():
   other=json.loads(p.read_text());overlap=local&{g['uuid'] for g in other.get('gpus',other.get('devices',[]))}
   if overlap:raise RuntimeError('GPU UUID overlap with '+role+': '+str(overlap))
 tasks=[(n,a,s) for n in CHANNELS for a in ('C0','C2') for s in ((2101,2102,2103) if mode=='formal' else (2101,))]
 if mode=='formal' and set(release['jobs'])!={jobid(*v) for v in tasks}:raise RuntimeError('release does not bind exact 18-job matrix')
 active={};pending=list(tasks);t=time.monotonic();cpu0=cpu_s();results=[]
 if mode=='formal' and not all((ROOT/'preflight'/jobid(n,a,2101)/'DONE.json').exists() for n in CHANNELS for a in ('C0','C2')):raise RuntimeError('six TRAIN path preflights must pass')
 def launch(task,gpu):
  n,a,s=task;cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--backbone',n,'--arm',a,'--seed',str(s),'--gpu',str(gpu),'--updates',str(updates),'--mode',mode]
  if mode=='formal':cmd+=['--release',str(release_path),'--release-sha256',release_hash]
  log=ROOT/'logs'/f'{mode}_{jobid(*task)}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  return {'job_id':jobid(*task),'gpu':gpu,'exit_code':r.returncode,'log':str(log)}
 with ThreadPoolExecutor(max_workers=4) as pool:
  for g in devices:
   if pending:active[pool.submit(launch,pending.pop(0),g)]=g
  while active:
   completed,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
   for f in completed:
    g=active.pop(f);r=f.result();results.append(r);print(r,flush=True)
    if pending:active[pool.submit(launch,pending.pop(0),g)]=g
 write_json(ROOT/f'{mode.upper()}_QUEUE.json',{'results':results,'wall_seconds':time.monotonic()-t,'cpu_core_seconds_coordinator_and_reaped_children':cpu_s()-cpu0,'four_gpu_dynamic_queue':True,'formal_launched':mode=='formal'})
 if any(r['exit_code'] for r in results):raise RuntimeError('one or more paths failed; receipts/logs retained')

def evaluate(path,expected):
 r=load_release(path,expected);spec=importlib.util.spec_from_file_location('s1_shared_evaluator',r['evaluator']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 locks=[ROOT/'formal'/jobid(n,a,s)/'MODEL_LOCK.json' for n in CHANNELS for a in ('C0','C2') for s in (2101,2102,2103)]
 if not all(p.exists() for p in locks):raise RuntimeError('all 18 terminal models must be locked before CAL/evaluation')
 # S1 owns CAL -> threshold locks -> external ordering and all metrics. This adapter supplies only model construction.
 mod.evaluate_s2_block(root=ROOT,build_model=build,release=r)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','synthetic','preflight','formal','worker','evaluate']);p.add_argument('--backbone',choices=list(CHANNELS));p.add_argument('--arm',choices=['C0','C2']);p.add_argument('--seed',type=int,default=2101);p.add_argument('--gpu',type=int,default=0);p.add_argument('--updates',type=int,default=20);p.add_argument('--mode',choices=['preflight','formal'],default='preflight');p.add_argument('--resume',action='store_true');p.add_argument('--release');p.add_argument('--release-sha256');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='synthetic':synthetic()
 elif a.command in ('preflight','formal'):schedule(a.command,15001 if a.command=='formal' else a.updates,a.release,a.release_sha256)
 elif a.command=='evaluate':evaluate(a.release,a.release_sha256)
 else:worker(a.backbone,a.arm,a.seed,a.gpu,a.updates,a.mode,a.resume,load_release(a.release,a.release_sha256) if a.mode=='formal' else None)
if __name__=='__main__':main()
