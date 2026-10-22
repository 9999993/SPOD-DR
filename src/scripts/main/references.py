"""Read only historical predictions and policies; ORDER never runs a network."""
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

def prepare():
    if (ROOT/'REFERENCE_INDEX.json').exists():return
    ms=ASSETS/'O2_MULTISEED_BOOST';registry=read(ms/'MODEL_AND_SYSTEM_REGISTRY.json')['models'];policies={p['id']:p for p in read(ms/'POLICY_REGISTRY.json')['policies']};preds=read(ms/'PREDICTION_INDEX.json');out=[]
    for b in ('B3','ConvNeXtTiny','SwinTiny','DenseNet121'):
        seed=config(b)['parent_seed']
        for kind,tag in [('OLD_O2','S'),('NATIVE','NATIVE')]:
            ident=f'{tag}__{b}__{seed}';m=next(m for m in registry if m['id']==ident)
            out.append({'id':ident,'kind':kind,'backbone':b,'seed':seed,'weight_path':m['path'],'weight_sha256':m['model_sha256'],'selected_step':m['selected_step'],'policy':policies[ident]['selection'],'policy_sha256':policies[ident]['sha256'],'predictions':preds[ident]})
    b0=ASSETS/'B0_C24_O2';reg=read(b0/'MODEL_REGISTRY.json')['models'];pp=read(b0/'POLICY_REGISTRY.json')['policies']
    for kind,tag in [('OLD_O2','O2'),('NATIVE','NATIVE')]:
        m=next(m for m in reg if m['kind']==tag);p=next(p for p in pp if p['id']==m['id']);pred={}
        for role in ('CAL',*TESTS):
            path=b0/'predictions'/tag/(role+'.npz');rec=read(path.with_suffix('.json'));assert rec['actual_loaded_sha']==m['selected']['sha256'];pred[role]={'path':str(path),'sha256':rec['prediction_sha256']}
        out.append({'id':tag+'__B0__2101','kind':kind,'backbone':'B0','seed':2101,'weight_path':m['selected']['path'],'weight_sha256':m['selected']['sha256'],'selected_step':m['selected']['step'],'policy':p['selection'],'policy_sha256':p['sha256'],'predictions':pred})
    for m in registry:
        if m['kind']!='ORDER':continue
        out.append({'id':m['id'],'kind':'ORDER','backbone':'B0+B0','seed':m['parent_seed'],'weight_sha256':m['model_sha256'],'policy':policies[m['id']]['selection'],'policy_sha256':policies[m['id']]['sha256'],'predictions':preds[m['id']]})
    # ResNet50 has no historical fixed O2. Preserve its original native and C2 identities.
    old=ASSETS/'GROUP_1/baseline_training/outputs/evaluation/normal__resnet50__2101'
    native=read(old/'THRESHOLD.json');pred={}
    for role in ('CAL',*TESTS[:4]):
        p=old/(role+'.npz');rec=read(p.with_suffix('.receipt.json'));pred[role]={'path':str(p),'sha256':rec['prediction_sha256'],'projection':'native_legacy'}
    wp=ASSETS/'GROUP_1/baseline_training/outputs/formal/normal__resnet50__2101/model_last.pt'
    out.append({'id':'NATIVE__ResNet50__2101','kind':'NATIVE','backbone':'ResNet50','seed':2101,'weight_path':str(wp),'weight_sha256':native['checkpoint_sha256'],'selected_step':15001,'policy':native['selection'],'policy_sha256':sha(old/'THRESHOLD.json'),'predictions':pred})
    stable=ASSETS/'GROUP_1/model_comparison';m=next(m for m in read(stable/'contracts/FROZEN_CONTRACT.json')['models'] if m['backbone']=='resnet50' and m['seed']==2101);pred={}
    pp=stable/'contracts/THRESHOLDS_LOCKED.json';pol=read(pp)['thresholds'][m['job_id']]['versions']['S2']
    for role in ('CAL',*TESTS[:4]):
        raw=m['predictions'][role];done=read(stable/'outputs'/m['job_id']/(role+'.DONE.json'));pred[role]={'path':raw['path'],'sha256':raw['sha256'],'projection':'C2_S2_chunks','chunks':done['chunks']}
    out.append({'id':'C2_S2__ResNet50__2101','kind':'C2_S2','backbone':'ResNet50','seed':2101,'weight_path':m['checkpoint']['path'],'weight_sha256':m['checkpoint']['sha256'],'selected_step':15001,'policy':pol,'policy_sha256':sha(pp),'predictions':pred})
    for m in out:
        if m.get('weight_path'):assert sha(m['weight_path'])==m['weight_sha256']
        for role in TESTS[4:]:
            if role not in m['predictions']:
                p=ROOT/'references'/m['id']/(role+'.npz')
                if p.with_suffix('.json').exists():
                    receipt=read(p.with_suffix('.json'));assert receipt['actual_loaded_sha']==m['weight_sha256'];m['predictions'][role]={'path':str(p),'sha256':receipt['prediction_sha256']}
    write(ROOT/'REFERENCE_INDEX.json',{'models':out,'ORDER_new_forwards':0,'missing':[{'id':m['id'],'cohort':r} for m in out for r in TESTS if r not in m['predictions']],'utc':now()})
    print('Historical reference identities',len(out),flush=True)
if __name__=='__main__':prepare()
