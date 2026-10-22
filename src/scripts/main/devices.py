"""Actual allocated GPU ownership; never preempt a live workload."""
import os,sys,fcntl,json,uuid,subprocess,contextlib,socket
from pathlib import Path
ROOT=Path(os.environ['SPOD_RUN_DIR']);ASSETS=Path(os.environ['SPOD_RUNS'])
@contextlib.contextmanager
def device():
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);token=uuid.uuid4().hex;handles=[];owner=None
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines():
        uid=line.strip()
        if uid in occupied:continue
        hs=[]
        try:
            for path in (ASSETS/'GPU_DEVICE_LOCKS'/('GPU_'+uid+'.lock'),ASSETS/'.gpu-claims'/(uid+'.lock')):
                path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+');hs.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            for f in hs:f.close()
            continue
        handles=hs;owner=ASSETS/'.gpu-claims'/(uid+'.owner.json');owner.write_text(json.dumps({'owner':'SPOD-DR','token':token,'pid':os.getpid(),'host':socket.gethostname(),'uuid':uid}));break
    else:raise RuntimeError('NO_FREE_DEVICE')
    try:yield uid,[f.fileno() for f in handles]
    finally:
        if owner.exists() and json.loads(owner.read_text())['token']==token:owner.unlink()
        for f in handles:f.close()
def launch(script,args,uid,fds,label):
    p=ROOT/'logs'/(label+'.log');p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uid,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script)),*args],env=env,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(fds));return worker,f
