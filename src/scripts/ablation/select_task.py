"""Freeze each task's J using only its two registered source DEV candidates."""
from runtime import *
from selection import dev_metric, ranks, tie
import shutil,sys

def choose(b):
    root=taskdir(b);dest=preddir(b)
    if (root/'J_SELECTION.json').exists():return read(root/'J_SELECTION.json')['selected']
    candidates=[]
    for step in (3000,6000):
        path=root/f'stage_O{step}.pt';pred=dest/f'DEV_O{step}.npz';receipt=read(pred.with_suffix('.json'))
        assert receipt['actual_loaded_sha']==sha(path) and receipt['prediction_sha256']==sha(pred)
        candidates.append({'O_step':step,'global_step':15001+step,'path':str(path),'sha256':sha(path),'DEV_path':str(pred),'metric':dev_metric(arrays(pred))})
    good=[c for c in candidates if c['metric']['status']=='VALID'];assert good,b+' has no valid J'
    for c,r in zip(good,ranks([c['metric'] for c in good])):c['R']=float(r)
    selected=min(good,key=lambda c:(c['R'],*tie(c['metric'],c['O_step'])))
    shutil.copy2(selected['path'],root/'selected.pt');selected={**selected,'path':str(root/'selected.pt')}
    write(root/'J_SELECTION.json',{'version':VERSION,'variant':VARIANT,'backbone':b,'candidates':candidates,'selected':selected,'rule':'LT1_J only O3000/6000, source DEV','utc':now()});return selected
if __name__=='__main__':choose(sys.argv[1])
