"""SPOD_DATASET_V1 main training with frozen required model consumers."""
import os, sys, json, time, socket
from pathlib import Path
WORK = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[1] / 'main/vendor/assets'
ROOT = Path(os.environ['SPOD_RUN_DIR']).resolve()
CODE = Path(__file__).parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(CODE.parent))
sys.path.insert(0, str(CODE/'vendor/code'))
from data_protocol import (PROTOCOL, prepared_rows, require_runtime, checkpoint_identity, validate_checkpoint, evaluation_roles, training_size)
require_runtime(ROOT)
from common import sha, read, write, append, csvread, csvwrite, now, manifest
import torch, numpy as np, models as cm, parent_models as pm, training as ct, parent_training as pt
from metrics import select_threshold, decode_threshold
from scipy.special import logsumexp
ct.precision()
P4 = ct.p4
TESTS = evaluation_roles(ROOT)
BACKBONES = ('B0', 'B3', 'ConvNeXtTiny', 'SwinTiny', 'ResNet50')

def config(b): return read(CODE.parents[2]/'configs'/'main'/(b+'.json'))
def taskdir(b): return ROOT/'checkpoints'/b
def rows(role):
    return prepared_rows(ROOT, role)
def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def np_save(path, z):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**z);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class UniformSampler:
    """Infinite deterministic shuffle stream; carry the tail into the next pass."""
    def __init__(self, size, seed, start, stop): self.size,self.seed,self.start,self.stop=size,seed,start,stop
    def __len__(self): return self.stop-self.start
    def __iter__(self):
        cache={}
        for step in range(self.start,self.stop):
            batch=[]
            for cursor in range(step*16,(step+1)*16):
                epoch,offset=divmod(cursor,self.size)
                if epoch not in cache:
                    cache={epoch:np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,epoch,0x5034]))).permutation(self.size)}
                batch.append(int(cache[epoch][offset]))
            yield batch
def loader(seed,start,stop):
    rr=rows('TRAIN')
    return torch.utils.data.DataLoader(P4.P4ManifestDataset(rr),batch_sampler=UniformSampler(len(rr),seed,start,stop),num_workers=int(os.environ.get('SPOD_WORKERS','6')),pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(seed+710000))

class Parent(pm.Model):
    def forward(self,x,aux=False):
        if self.b=='B0':
            from b0 import forward_b0
            return forward_b0(self,x,aux)
        return pm.Model.forward(self,x,aux)
def parent(b):
    if b=='B0': import b0
    m=Parent(b,'C2',config(b)['parent_seed'],True)
    return m
class Continuation(cm.Model):
    def __init__(self,b,state):
        # No call to the historical Model constructor: it loads old trained parents.
        torch.nn.Module.__init__(self)
        p=parent(b);self.__dict__.update(p.__dict__)
        self.load_state_dict(state,strict=True)
        self.parent_new=False;self.arm='SAM';self.parent_rng=None
        cm.ReferenceModel.configure(self,1)
        self.head_names={n for n,p in self.named_parameters() if p.requires_grad}
        cm.ReferenceModel.configure(self,2)
        self.extra_names={n for n,p in self.named_parameters() if p.requires_grad}-self.head_names
        self.__dict__['_anchors']={n:p.detach().cpu().clone() for n,p in self.named_parameters() if n in self.extra_names}
        bn=[n for n,m in self.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        self.parent_bn={n:v.detach().cpu().clone() for n,v in self.named_buffers() if any(n.startswith(k+'.') for k in bn)}
        self.sp_alpha=.01;self.sam_rho=.025;self.parent_seed=self.seed
        self.continuation_seed=config(b)['O2_seed'];self.replicate=1
        self.configure(1)
    def forward(self,x,aux=False):
        if self.b=='B0':
            from b0 import forward_b0
            return forward_b0(self,x,aux)
        return cm.ReferenceModel.forward(self,x,aux)

def start_parent(b):
    m=parent(b).cuda();m.restore_parent_rng();return m,m.configure(1)
def start_o2(b,state):
    m=Continuation(b,state).cuda();P4._set_seed(m.continuation_seed);m.mix_cpu.manual_seed(m.continuation_seed+970000)
    return m,m.configure(1)
def checkpoint(m,o,phase,step,counts):
    return {**checkpoint_identity(ROOT), 'version':PROTOCOL,'config_sha256':sha(CODE.parents[2]/'configs'/'main'/(m.b+'.json')),'split_sha256':sha(ROOT/'SPLIT_RECORD.json'),'backbone':m.b,'phase_name':phase,'phase_step':step,'global_step':step if phase=='P' else 15001+step,'model':m.state_dict(),'optimizer':o.state_dict(),'rng':P4._rng_state(),'mix_rng':m.mix_cpu.get_state(),'phase':m.phase,'counts':dict(counts),'exposure':getattr(m,'exposure',{})}
def restore(b,path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(ck, ROOT)
    assert ck['version'] == PROTOCOL and ck['backbone']==b and ck['config_sha256']==sha(CODE.parents[2]/'configs'/'main'/(b+'.json')) and ck['split_sha256']==sha(ROOT/'SPLIT_RECORD.json')
    if ck['phase_name']=='P':m,o=start_parent(b)
    else:
        parent_ck=torch.load(taskdir(b)/'parent_15001.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(parent_ck, ROOT)
        parent_state=parent_ck['model']
        m,o=start_o2(b,parent_state)
    o=m.configure(ck['phase']);m.load_state_dict(ck['model'],strict=True);o.load_state_dict(ck['optimizer']);m.mix_cpu.set_state(ck['mix_rng']);m.exposure=ck.get('exposure',{});P4._restore_rng_state(ck['rng'])
    return m,o,ck
