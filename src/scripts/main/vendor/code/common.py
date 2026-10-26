"""C18 owned namespace, immutable inputs and atomic receipts."""
import csv, hashlib, json, os, sys, time, importlib.util
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
CODE = Path(__file__).resolve().parent
REPORT = Path(__file__).resolve().parent.parent
ARCHIVE = CODE/'docs/manuscripts/sparrow-v1-graefes/post-c19-dual-objective-web-source'
ASSETS = CODE.parent / 'assets'
G1 = ASSETS/'GROUP_1'
RUN = G1/'gradient_training'
C23 = G1/'sam_training'
C23_REPORT=C23/'code/reports/c23-structured-sam-r1'
C18_REPORT = Path(os.environ.get('C21_C18_REPORT_ROOT',ASSETS.parent.parent/'backbone_sources/reports/c18-six-backbone-three-mechanism-r1')).resolve()
C18 = G1/'backbone_training'
PLAN = CODE/'docs/superpowers/plans/group1/c24-gradient-routing-preconditioning'
C17 = G1/'training_environment'
S1 = G1/'model_dependencies/S1'
S2 = G1/'model_dependencies/S2'
NORMAL = G1/'baseline_training'
DATA = G1/'data_manifests'
BACKBONES = ('B3','ResNet50','DenseNet121','SwinTiny','ConvNeXtTiny','InceptionV3')
NAMES = dict(zip(BACKBONES, ('efficientnet_b3','resnet50','densenet121','swin_tiny','convnext_tiny','inception_v3')))
from data_protocol import validate_prepared
_DATA_ROOT = Path(os.environ['SPOD_RUN_DIR'])
_DATA_RECORD = validate_prepared(_DATA_ROOT)
ROLES = {role:(f'data/{role}.json', _DATA_RECORD['counts'][role], _DATA_RECORD['files'][f'data/{role}.json']) for role in ('TRAIN','DEV','CAL')}
TESTS = tuple(_DATA_RECORD['evaluation_roles'])
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('w') as f:json.dump(v,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rs:p.write_text('');return
 names=list(dict.fromkeys(k for r in rs for k in r))
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader();w.writerows(rs);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def jid(b,s,a):return f'{b}__{s}__{a}'
def task_path(task):return RUN/'checkpoints'/task
def backend():
 sys.path.insert(0,str(S1/'code/scripts/dr/sparrow'))
 import p4_execution as p4
 p4._bootstrap_old_sources()
 return p4
def s2_backend():
 backend();sys.path.insert(0,str(S2/'code/scripts/dr/sparrow'))
 import clean_s2
 return clean_s2
def manifest(role):
 from data_protocol import prepared_rows
 return prepared_rows(Path(os.environ['SPOD_RUN_DIR']),role)

C20 = G1/'model_contracts'
C20_REPORT = C20/'code/reports/c20-full-c2-dual-objective-r1'
C21 = G1/'full_training'
C21_REPORT = C21/'code/reports/c21-intact-c2-training-r1'
C22 = G1/'continuation_factors'
C22_REPORT = C22/'code/reports/c22-continuation-factors-r2'
PLAN_SHA = '8541b4b97c94d1ae0cd8a93d6ecf7bdf9191d04c'
SEEDS=(2101,2102,2103)
CANDIDATES=('SGF','SHAM')
ARMS=('T','K','SAM','SGF','MGF','SHAM','PREFIX','PREFIX_STABLE','C2S','C2_SUM')
BN_BACKBONES=('B3','ResNet50','DenseNet121','InceptionV3')
def tasks():return csvread(PLAN/'TRAINING_MATRIX.csv')
def prefixes():return csvread(PLAN/'PREFIXES.csv')
def readouts():return []
def c24id(b,p,r,a):return f'{b}__p{p}__r{r}__{a}'
