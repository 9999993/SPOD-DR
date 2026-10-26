"""One-pass fixed-weight predictions, cached by checkpoint SHA and role."""
from runtime import *
import argparse
def run(b,path,roles,dest):
    ck=torch.load(path,map_location='cpu',weights_only=False);weight_sha=sha(path)
    validate_checkpoint(ck, ROOT)
    if ck['phase_name']=='P':m=parent(b).cuda()
    else:
        parent_ck=torch.load(taskdir(b)/'parent_15001.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(parent_ck, ROOT)
        state=parent_ck['model'];m=Continuation(b,state).cuda()
    m.load_state_dict(ck['model'],strict=True);m.eval();m.phase=0
    for role in roles:
        p=Path(dest)/(role+'.npz')
        if p.exists() and p.with_suffix('.json').exists():
            r=read(p.with_suffix('.json'));assert r['actual_loaded_sha']==weight_sha and sha(p)==r['prediction_sha256'];continue
        rr=rows('DEV' if role.startswith('DEV') else role)
        import p4_analysis as evaluation
        dl=torch.utils.data.DataLoader(evaluation._EvalDataset(rr),batch_size=16,shuffle=False,num_workers=6,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(812101))
        chunks=[];mus=[];sigmas=[];start=time.monotonic()
        with torch.inference_mode():
            for batch in dl:
                out=m(batch['image'].cuda(non_blocking=True));chunks.append(out['logp'].cpu().numpy());mus.append(out['mu'].cpu().numpy());sigmas.append(out['sigma'].cpu().numpy())
        del dl;lp=np.concatenate(chunks);assert np.isfinite(lp).all()
        assert np.allclose(np.exp(lp).sum(1),1,atol=2e-6 if ck['phase_name']=='P' else 1e-12,rtol=0)
        z={'logp':lp,'probabilities':np.exp(lp),'score':logsumexp(lp[:,2:],axis=1)-logsumexp(lp[:,:2],axis=1),'pred':lp.argmax(1),'grade':np.array([int(r['grade']) for r in rr]),'row_index':np.arange(len(rr)),'component_hash':np.array([r['component_hash'] for r in rr]),'image_key':np.array([r.get('image_id',str(i)) for i,r in enumerate(rr)]),'mu':np.concatenate(mus),'sigma':np.concatenate(sigmas)}
        receipt={'status':'PASS','backbone':b,'role':role,'checkpoint':str(path),'actual_loaded_sha':weight_sha,'forward':len(rr),'seconds':time.monotonic()-start,'global_step':ck['global_step'],'utc':now()}
        if role.startswith('DEV'):
            from selection import dev_metric
            receipt['metric']=dev_metric(z)
            if role!='DEV_RELOAD':append(taskdir(b)/'DEV_CURVE.jsonl',{'role':role,'global_step':ck['global_step'],'metric':receipt['metric'],'weight_sha256':weight_sha,'utc':receipt['utc']})
        np_save(p,z);receipt['prediction_sha256']=sha(p);write(p.with_suffix('.json'),receipt);print(b,role,len(rr),'seconds',round(time.monotonic()-start,2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone');p.add_argument('--path',required=True);p.add_argument('--roles',nargs='+',required=True);p.add_argument('--dest',required=True);a=p.parse_args();run(a.backbone,Path(a.path),a.roles,a.dest)
