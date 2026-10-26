"""Clean R0 identities, strict input resolution, and S1-owned publication helpers."""
from __future__ import annotations
import csv, hashlib, json, os, socket, time
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path('/PROJECT_INPUTS/model_project')
ROOT = PROJECT / 'SPARROW_ASSETS/GROUP_1/model_dependencies'
DATA = PROJECT / 'SPARROW_ASSETS/GROUP_1/data_manifests'
PLAN_COMMIT = '910a4f6a32bc5c6b9d42a4f893bbc6f8833479d2'
EVIDENCE_COMMIT = '6109168c90ce79fb2d21937b92c921d75c253ebc'
READY_SHA = '78fbe504e57490f5140561a3d874184fa16b0fa6be0d015dce9ca164c313508f'
INVENTORY_SHA = '296345aae26d4188b7c9b74dde8847dcc952ba091c086a7959ff87bbeed2809c'
MANIFESTS = {
 'TRAIN': ('train_manifest.csv',23987,'e6757408c17a22bcc9672bbbd3084adfc5064113ecea1d0224a9e8c0d9e78ebc'),
 'DEV': ('dev_manifest.csv',2984,'0e3f01299ac7ab7dd1322166a2e15fdc200e8441661a93d790296d98850ff112'),
 'CAL': ('threshold_manifest.csv',2524,'8082f405faf1738d0483637a01e9a5152c15195c12bf88f3d9cf32ba78657109'),
}
ARMS=('C0','C1','C2','M3','S-X','F-R','SNR-B3-adapted')
SEEDS=tuple(range(2101,2106))

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def immutable(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=canonical(value)
 if path.exists():
  if path.read_bytes()!=data:raise FileExistsError(f'immutable publication differs: {path}')
  return
 # Publish only a fully fsynced file. Link is O_EXCL even across hosts.
 tmp=path.with_name(path.name+f'.{socket.gethostname()}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path)
 finally:tmp.unlink()
def append(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('ab',buffering=0) as f:f.write(canonical(value))
def identity(data=DATA):
 data=Path(data)
 if data.resolve()!=DATA:raise ValueError('only the fixed clean R0 bundle is permitted')
 if sha(data/'READY.json')!=READY_SHA or sha(data/'FILE_SHA256.json')!=INVENTORY_SHA:raise ValueError('clean data identity mismatch')
 st=data.stat();fs=os.statvfs(data)
 return {'path':str(data.resolve()),'st_dev':st.st_dev,'st_ino':st.st_ino,'filesystem_id':fs.f_fsid,'ready_sha256':READY_SHA,'inventory_sha256':INVENTORY_SHA}
def reuse_validation(receipt=ROOT/'control/DATA_VALIDATION_COMPLETE.json',check_stats=True):
 r=json.loads(Path(receipt).read_text())
 if r['status']!='PASS' or r['binding']!=identity():raise ValueError('full verification receipt is absent or bound to another mount')
 if check_stats:
  inventory=json.loads((DATA/'FILE_SHA256.json').read_text())
  h=hashlib.sha256()
  for rel in sorted(inventory):
   p=DATA/rel;s=p.stat()
   h.update(canonical([rel,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]))
  if h.hexdigest()!=r['file_stat_binding_sha256']:raise ValueError('bundle changed after full validation; full revalidation required')
 return r

def read_clean(role,data=DATA):
 if role not in MANIFESTS:raise ValueError(role)
 identity(data)
 name,n,expected=MANIFESTS[role];path=Path(data)/name
 if sha(path)!=expected:raise ValueError(f'{role} manifest hash mismatch')
 with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 if len(rows)!=n:raise ValueError(f'{role} count mismatch')
 seen=set()
 for i,row in enumerate(rows):
  key=(row['source'],row['source_row'])
  if key in seen:raise ValueError('duplicate source/source_row key')
  seen.add(key)
  g=int(row['grade'])
  if g not in range(5) or int(row['label'])!=int(g>=2):raise ValueError('invalid grade/q2 label')
  if row['role']!=('CAL' if role=='CAL' else role):raise ValueError('role mismatch')
  if role=='CAL' and row['source']!='EyePACS':raise ValueError('CAL must be threshold EyePACS only')
  rel=Path(row['cache_relative_path'])
  if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='cache':raise ValueError('invalid clean cache relative path')
  resolved=(Path(data)/rel).resolve()
  if not resolved.is_relative_to(Path(data).resolve()):raise ValueError('cache escapes bundle')
  if row['cache_path']!=str(resolved):raise ValueError('clean absolute/relative cache disagree; no legacy fallback')
  if not resolved.is_file() or len(row['component_hash'])!=64:raise ValueError('cache/component missing')
  # Global manifest ordinal avoids source-local row_index collisions in predictions.
  row['source_row_index']=row['row_index'];row['row_index']=str(i)
  row['cache_path']=str(resolved)
 return rows
