"""Only genuinely absent fixed ResNet50 reference predictions; no ORDER forward."""
from runtime import *
import argparse
def run(ident):
    entry=next(m for m in read(ROOT/'REFERENCE_INDEX.json')['models'] if m['id']==ident)
    assert entry['backbone']=='ResNet50' and entry['kind'] in ('NATIVE','C2_S2')
    path=entry['weight_path'];assert sha(path)==entry['weight_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);state=ck.get('model',ck)
    m=pm.Model('ResNet50','normal' if entry['kind']=='NATIVE' else 'C2',2101,True).cuda()
    if entry['kind']=='NATIVE':m.native.load_state_dict({k.removeprefix('native.'):v for k,v in state.items()},strict=True)
    else:m.inner.load_state_dict(cm.canonical_c2_state(state),strict=True)
    m.eval();m.phase=0
    import p4_analysis as evaluation
    for role in TESTS[4:]:
        path=ROOT/'references'/ident/(role+'.npz')
        if path.with_suffix('.json').exists():
            receipt=read(path.with_suffix('.json'));assert receipt['actual_loaded_sha']==entry['weight_sha256'] and sha(path)==receipt['prediction_sha256'];continue
        rr=rows(role);dl=torch.utils.data.DataLoader(evaluation._EvalDataset(rr),batch_size=16,num_workers=6,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(812101))
        chunks=[];mus=[];sigmas=[];start=time.monotonic()
        with torch.inference_mode():
            for batch in dl:
                x=batch['image'].cuda(non_blocking=True)
                if entry['kind']=='NATIVE':out={'logp':torch.log_softmax(m.native(x).double(),1)}
                else:out=m(x)
                chunks.append(out['logp'].cpu().numpy())
                if 'mu' in out:mus.append(out['mu'].cpu().numpy());sigmas.append(out['sigma'].cpu().numpy())
        lp=np.concatenate(chunks);z={'logp':lp,'probabilities':np.exp(lp),'score':logsumexp(lp[:,2:],1)-logsumexp(lp[:,:2],1),'pred':lp.argmax(1),'grade':np.array([int(r['grade']) for r in rr]),'row_index':np.arange(len(rr)),'component_hash':np.array([r['component_hash'] for r in rr]),'image_key':np.array([r['image_id'] for r in rr])}
        if mus:
            z.update(mu=np.concatenate(mus),sigma=np.concatenate(sigmas))
            from reference_numeric import stable_scores
            z['score']=stable_scores(z['mu'],z['sigma'])[1]
        np_save(path,z);write(path.with_suffix('.json'),{'status':'PASS','actual_loaded_sha':entry['weight_sha256'],'prediction_sha256':sha(path),'role':role,'forward':len(rr),'seconds':time.monotonic()-start,'reason':'missing fixed historical ResNet reference, separately counted','utc':now()})
    print('Missing historical reference predictions completed',ident,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('identity');a=p.parse_args();run(a.identity)
