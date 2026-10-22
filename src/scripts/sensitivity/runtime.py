"""SPOD_DATASET_V1 sensitivity runtime (four non-Swin backbones)."""
import os, sys, json, time, socket
from pathlib import Path
WORK = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[1] / 'main/vendor/assets'
DATA_ROOT = Path(os.environ['SPOD_RUN_DIR']).resolve()
OWNER = os.environ.get('SPOD_OWNER', 'B')
TASK = os.environ.get('SPOD_TASK', 'B0__K_4')
ROOT = Path(os.environ['SPOD_RUN_DIR']).resolve()
CODE = Path(__file__).parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(CODE.parent))
sys.path.insert(0, str(CODE.parent/'main/vendor/code'))
from data_protocol import (PROTOCOL, prepared_rows, require_runtime, checkpoint_identity, validate_checkpoint, evaluation_roles, training_size)
VERSION = PROTOCOL
SHARED = ROOT.parents[1] / 'shared'
require_runtime(ROOT)
from common import sha, read, write, append, csvread, csvwrite, now, manifest
import torch, numpy as np, models as cm, parent_models as pm, training as ct, parent_training as pt
from metrics import select_threshold, decode_threshold
from scipy.special import logsumexp
ct.precision()
P4 = ct.p4
TESTS = evaluation_roles(ROOT)
BACKBONES = ('B0', 'B3', 'ConvNeXtTiny', 'ResNet50')

def config(b):
    c=read(CODE.parents[2]/'configs'/'sensitivity'/(TASK+'.json'));assert c['backbone']==b;return c
def taskid(b):
    assert config(b)['backbone']==b;return TASK
def taskdir(b): return ROOT/'checkpoints'/taskid(b)
def preddir(b): return ROOT/'predictions'/taskid(b)
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
    from k_model import install, assert_shape
    install(config(b)['K'],pm)
    if b=='B0': import b0
    m=Parent(b,'C2',config(b)['parent_seed'],True)
    assert_shape(m,config(b)['K']);return m
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
        self.sp_alpha=.01;self.parent_seed=self.seed
        from rho_state import configure_rho
        configure_rho(self,config(b)['sam_rho'])
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
    from rho_state import pack_complete
    ck=pack_complete(sys.modules[__name__],m,o,phase,step,counts,taskdir(m.b)/'parent_15001.pt' if phase!='P' else None,task_id=TASK)
    ck['config_sha256']=sha(CODE.parents[2]/'configs'/'sensitivity'/(TASK+'.json'))
    return ck

def restore(b,path):
    from rho_state import restore_complete
    m,o,ck=restore_complete(sys.modules[__name__],b,path,rho=config(b)['sam_rho'],expected_K=config(b)['K'],expected_task=TASK)
    assert ck['config_sha256']==sha(CODE.parents[2]/'configs'/'sensitivity'/(TASK+'.json'))
    return m,o,ck
