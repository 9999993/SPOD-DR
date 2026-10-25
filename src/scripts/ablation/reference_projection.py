"""Read saved reference predictions, policies and model identities."""
from runtime import *
from functools import lru_cache

def project(entry,role):
    if entry.get('projection')=='ORDER_components':
        zz=[]
        for e in entry['members']:
            assert sha(e['path'])==e['sha256'];zz.append(arrays(e['path']))
        a,b=zz
        for k in ('grade','row_index','component_hash'):assert np.array_equal(a[k],b[k])
        p=(a['probabilities']+b['probabilities'])/2
        z={**a,'probabilities':p,'pred':p.argmax(1),'score':np.log(p[:,2:].sum(1))-np.log(p[:,:2].sum(1))}
    else:
        path=entry['path'];assert sha(path)==entry['sha256'];z=arrays(path);mode=entry.get('projection','standard')
        if mode=='native_legacy':z.update(probabilities=z['grade_probs'],score=z['log_odds'],pred=z['grade_pred'])
        elif mode=='C2_S2_chunks':
            pieces=[]
            for p in entry['chunks']:assert sha(p['path'])==p['sha256'];pieces.append(arrays(p['path']))
            z.update(probabilities=z['grade_probs'],pred=z['grade_pred'],score=np.concatenate([p['S2'] for p in pieces]))
            assert np.array_equal(np.concatenate([p['grade_probs_replay'] for p in pieces]),z['probabilities'])
        else:
            if 'probabilities' not in z:z['probabilities']=np.exp(z['logp'])
        assert np.isfinite(z['score']).all()
    rr=rows(role);assert np.array_equal(z['grade'],[int(r['grade']) for r in rr]);assert np.array_equal(z['component_hash'],[r['component_hash'] for r in rr]);assert np.array_equal(z['row_index'],np.arange(len(rr)))
    z['image_key']=np.array([r.get('image_id',str(i)) for i,r in enumerate(rr)])
    return z
