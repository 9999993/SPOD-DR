"""Reload selected DEV, freeze source CAL, then evaluate three whole cohorts."""
from runtime import *
from infer import run
import argparse
def finish(b):
    root=taskdir(b);dest=ROOT/'predictions'/b;selected=read(root/'J_SELECTION.json')['selected'];path=Path(selected['path'])
    assert sha(path)==selected['sha256']
    run(b,path,['DEV_RELOAD','CAL'],dest)
    original=arrays(selected['DEV_path']);loaded=arrays(dest/'DEV_RELOAD.npz')
    assert np.array_equal(original['logp'],loaded['logp']) and np.array_equal(original['score'],loaded['score']),'selected DEV reload differs'
    z=arrays(dest/'CAL.npz');policy=select_threshold(z['score'],z['grade']);assert policy['status']=='PASS'
    write(root/'POLICY.json',{'backbone':b,'model_sha256':selected['sha256'],'global_step':selected['global_step'],'O_step':selected['O_step'],'CAL_prediction_sha256':sha(dest/'CAL.npz'),'selection':policy,'utc':now()})
    run(b,path,list(TESTS),dest)
    write(root/'EVALUATION_DONE.json',{'status':'PASS','selected_DEV_reload':'BITWISE_PASS','model_sha256':selected['sha256'],'cohorts':list(TESTS),'utc':now()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone');a=p.parse_args();finish(a.backbone)
