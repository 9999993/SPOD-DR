"""A-owned complete-state API. No model-only checkpoint can enter restore_complete.

R is the caller's runtime module (original SPOD-DR or sensitivity). All parent
anchors and fixed BN buffers are reconstructed from the original parent bytes.
"""
from pathlib import Path
import math
import torch
API_VERSION = 'SPOD_SENSITIVITY_FULL_STATE_R1'
REQUIRED = {'model','optimizer','rng','mix_rng','phase','phase_name','phase_step',
            'global_step','counts','sampler','parent_path','parent_sha256',
            'split_sha256','backbone','K','sam_rho','api_version'}

def configure_rho(model, rho):
    rho = float(rho)
    assert math.isfinite(rho) and rho > 0
    model.sam_rho = rho
    assert model.sam_rho == rho
    return rho

def pack_complete(R, m, optimizer, phase, step, counts, parent_path, *, split_path=None, task_id=None):
    parent_path = Path(parent_path) if parent_path else None
    seed = R.config(m.b)['parent_seed' if phase == 'P' else 'O2_seed']
    n = len(R.rows('TRAIN'))
    k = int(m.inner.base.local_branch.queries.shape[0])
    return {**R.checkpoint_identity(R.ROOT), 'api_version':API_VERSION,'version':R.PROTOCOL,
            'task_id':task_id,'backbone':m.b,'K':k,'sam_rho':float(getattr(m,'sam_rho',.025)),
            'phase_name':phase,'phase_step':step,'global_step':step if phase=='P' else 15001+step,
            'model':m.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':R.P4._rng_state(),'mix_rng':m.mix_cpu.get_state(),'phase':m.phase,
            'counts':dict(counts),'exposure':getattr(m,'exposure',{}),
            'sampler':{'algorithm':'uniform_shuffle_tail_carry_v1',
                       'seed':seed,'next_batch':step,'next_image_cursor':step*16,'size':n,'batch':16},
            'parent_path':str(parent_path) if parent_path else None,
            'parent_sha256':R.sha(parent_path) if parent_path else None,
            'split_sha256':R.sha(split_path or R.ROOT/'SPLIT_RECORD.json')}

def restore_complete(R, b, path, *, rho=None, expected_K=None, expected_task=None, split_path=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    R.validate_checkpoint(ck, R.ROOT)
    assert REQUIRED <= ck.keys(), 'model-only/incomplete checkpoint is forbidden'
    assert ck['api_version']==API_VERSION and ck['backbone']==b
    assert ck['version'] == R.PROTOCOL
    assert ck['split_sha256']==R.sha(split_path or R.ROOT/'SPLIT_RECORD.json')
    if expected_K is not None: assert ck['K']==int(expected_K)
    if expected_task is not None: assert ck['task_id']==expected_task
    seed = R.config(b)['parent_seed' if ck['phase_name']=='P' else 'O2_seed']
    assert ck['sampler']=={'algorithm':'uniform_shuffle_tail_carry_v1',
                          'seed':seed,'next_batch':ck['phase_step'],'next_image_cursor':ck['phase_step']*16,
                          'size':len(R.rows('TRAIN')),'batch':16}
    if ck['phase_name']=='P': m,o=R.start_parent(b)
    else:
        p=Path(ck['parent_path']);assert R.sha(p)==ck['parent_sha256']
        parent_ck=torch.load(p,map_location='cpu',weights_only=False)
        R.validate_checkpoint(parent_ck, R.ROOT)
        parent=parent_ck['model']
        m,o=R.start_o2(b,parent)
    o=m.configure(ck['phase']);m.load_state_dict(ck['model'],strict=True)
    assert list(m.inner.base.local_branch.queries.shape)==[ck['K'],256]
    o.load_state_dict(ck['optimizer'])
    configure_rho(m,ck['sam_rho'] if rho is None else rho)
    m.mix_cpu.set_state(ck['mix_rng']);m.exposure=ck.get('exposure',{})
    R.P4._restore_rng_state(ck['rng'])
    return m,o,ck

def compare_model(R, actual, historical_path):
    historical=torch.load(historical_path,map_location='cpu',weights_only=False)['model']
    assert set(actual)==set(historical)
    differences=[];maximum=0.
    for key,value in actual.items():
        x=value.detach().cpu();y=historical[key]
        assert x.shape==y.shape and x.dtype==y.dtype
        if not torch.equal(x,y):
            delta=float((x.double()-y.double()).abs().max());maximum=max(maximum,delta)
            differences.append({'tensor':key,'max_abs':delta})
    return {'status':'BITWISE_PASS' if not differences else 'MISMATCH',
            'historical_path':str(historical_path),'historical_sha256':R.sha(historical_path),
            'tensor_count':len(actual),'unequal_tensor_count':len(differences),
            'maximum_absolute_difference':maximum,'differences':differences}
