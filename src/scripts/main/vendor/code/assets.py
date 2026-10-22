"""Read-only C23 source and parent identities; C24 owns new outputs."""
from common import *
import subprocess,socket

def parents(b,s,kind):
    row=next(x for x in read(C23_REPORT/'ASSET_BINDING_FINAL.json')['bindings'] if x['backbone']==b and int(x['seed'])==s and x['role']==kind)
    return Path(row['model']),Path(row['recovery'])

def public():
    return read(Path(os.environ['SPOD_RUN_DIR'])/'INITIALIZATION.json')


def prefix_input(b,s):
    fork=next(x for x in read(C23_REPORT/'PREFIX_REUSE_AND_FORK.json')['forks'] if x['id']==jid(b,s,'SAM_ON'))
    return Path(fork['prefix_checkpoint_sha256'] and C23/'checkpoints'/fork['prefix'])/'latest.pt'

def reference_terminal(b,s,a):
    if a in ('normal','C2'):return parents(b,s,a)[0]
    return Path(read(RUN/'contracts/REUSE_BINDINGS.json')['terminals'][jid(b,s,a)]['model'])
