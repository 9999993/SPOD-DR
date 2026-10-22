"""One complete chain, durable same-task recovery and ordinary training logs."""
from runtime import *
import argparse, traceback, fcntl

def train(b):
    root=taskdir(b);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'DONE.json').exists():return
    counts=ct.counter();phase='P';pos=0;status='FAIL';start=time.monotonic();cpu=ct.cpu_seconds();error=None
    try:
        if (root/'latest.pt').exists():
            m,o,ck=restore(b,root/'latest.pt');phase=ck['phase_name'];pos=ck['phase_step'];counts=ck['counts']
            append(root/'TRAINING_CURVE.jsonl',{'event':'RESUME','phase':phase,'phase_step':pos,'global_step':ck['global_step'],'source':str(root/'latest.pt'),'utc':now()})
        else:
            m,o=start_parent(b)
            public=read(ROOT/'INITIALIZATION.json')[b]
            write(root/'INITIALIZATION.json',{'public':public,'parent_seed':config(b)['parent_seed'],'O2_seed':config(b)['O2_seed'],'historical_trained_weights_loaded':False,'utc':now()})
        first_updates=counts['updates'];first=pos
        def save(name):P4._atomic_torch_save(checkpoint(m,o,phase,pos,counts),root/name)
        def stage(name):
            P4._atomic_torch_save({**checkpoint_identity(ROOT), 'model':m.state_dict(),'backbone':b,'phase_name':phase,'phase_step':pos,'global_step':pos if phase=='P' else 15001+pos,'version':PROTOCOL},root/name)
        while True:
            limit=15001 if phase=='P' else 6000
            if pos==limit:
                if phase=='P':
                    stage('parent_15001.pt');state={k:v.detach().cpu() for k,v in m.state_dict().items()}
                    del m,o;torch.cuda.empty_cache();m,o=start_o2(b,state);del state;phase='O';pos=0;first=0
                    append(root/'TRAINING_CURVE.jsonl',{'event':'P_TO_O1','phase':'O1','global_step':15001,'parent_path':str(root/'parent_15001.pt'),'parent_sha256':sha(root/'parent_15001.pt'),'seed':config(b)['O2_seed'],'optimizer':'fresh AdamW','BN':'parent statistics fixed','utc':now()});save('latest.pt')
                    continue
                break
            seed=config(b)['parent_seed' if phase=='P' else 'O2_seed'];loop_start=pos
            before=ct.frozen_digest(m) if phase!='P' else None
            train_start=time.monotonic()
            for batch in loader(seed,pos,limit):
                if ct.STOP_REQUESTED:save('latest.pt');raise InterruptedError('same-task durable checkpoint saved')
                pos+=1
                if phase=='P':
                    values=pt.step(m,o,batch,pos,counts,False);loss,parts=values[:2]
                else:
                    values=ct.step(m,o,batch,pos,counts,False);loss,parts=values['loss'],values['parts']
                if pos%100==0 or pos in (1,1500,1501,7500,15001,3000,6000):
                    seconds=time.monotonic()-train_start
                    log={'phase':'P' if phase=='P' else 'O1' if pos<=1500 else 'O2','phase_step':pos if phase=='P' or pos<=1500 else pos-1500,'O_step':pos if phase!='P' else None,'global_step':pos if phase=='P' else 15001+pos,'loss':loss,'parts':parts,'lr':[g['lr'] for g in o.param_groups],'seconds':seconds,'timed_updates':pos-loop_start,'counts':dict(counts),'utc':now()}
                    append(root/'TRAINING_CURVE.jsonl',log)
                    write(root/'PROGRESS.json',{'backbone':b,'pid':os.getpid(),'host':socket.gethostname(),'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'phase':log['phase'],'global_step':log['global_step'],'phase_step':pos,'seconds':seconds,'timed_updates':pos-loop_start,'counts':counts,'utc':now()})
                    print(b,log['phase'],pos,'loss',round(loss,6),'s/update',round(seconds/max(1,pos-loop_start),3),flush=True)
                if phase!='P' and pos==1500:
                    assert ct.frozen_digest(m)==before;m.check_bn();o=m.configure(2);before=ct.frozen_digest(m)
                    append(root/'TRAINING_CURVE.jsonl',{'event':'O1_TO_O2','phase':'O2','global_step':16501,'O_step':1500,'optimizer':'rebuilt AdamW','lr':[g['lr'] for g in o.param_groups],'SAM_rho':.025,'BN':'fixed','utc':now()})
                if pos%1000==0 or pos==limit or (phase!='P' and pos==1500):save('latest.pt')
                points=(7500,15001) if phase=='P' else (1500,3000,6000)
                if pos in points:stage(f'stage_{phase}{pos}.pt')
                if torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('VRAM headroom exceeded')
            del batch
            if phase!='P':assert ct.frozen_digest(m)==before;m.check_bn()
        stage('last.pt');save('latest.pt')
        assert counts['updates']==21001 and counts['training_forward']==816032
        write(root/'DONE.json',{'status':'PASS','backbone':b,'global_step':21001,'counts':counts,'last':str(root/'last.pt'),'last_sha256':sha(root/'last.pt'),'utc':now()});status='PASS'
    except BaseException as e:error=repr(e);traceback.print_exc();raise
    finally:
        write(root/f'COST_{os.getpid()}.json',{'status':status,'error':error,'physical_updates':counts['updates']-locals().get('first_updates',0),'counts':counts,'seconds':time.monotonic()-start,'cpu_seconds':ct.cpu_seconds()-cpu,'peak_reserved':torch.cuda.max_memory_reserved(),'utc':now()})

def check():
    root=ROOT/'check';root.mkdir(exist_ok=True);b='B0';counts=ct.counter()
    batches=iter(loader(2101,0,3));batch=next(batches);m,o=start_parent(b)
    pt.step(m,o,batch,1,counts,True)
    path=root/'parent.pt';P4._atomic_torch_save(checkpoint(m,o,'P',1,counts),path)
    next_batch=next(batches);pt.step(m,o,next_batch,2,counts,False);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    del m,o;torch.cuda.empty_cache();m,o,ck=restore(b,path);pt.step(m,o,next_batch,2,counts,False)
    assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in expected.items()),'parent next-step replay differs'
    state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};del m,o;torch.cuda.empty_cache();m,o=start_o2(b,state)
    ct.step(m,o,batch,1,counts,True);o=m.configure(2);ct.step(m,o,batch,1501,counts,True)
    ck=checkpoint(m,o,'O',1501,counts);P4._atomic_torch_save(ck,root/'sam.pt')
    ct.step(m,o,next_batch,1502,counts,False);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    del m,o;torch.cuda.empty_cache();ck=torch.load(root/'sam.pt',map_location='cpu',weights_only=False);validate_checkpoint(ck, ROOT);m,o=start_o2(b,state);o=m.configure(2);m.load_state_dict(ck['model'],strict=True);o.load_state_dict(ck['optimizer']);P4._restore_rng_state(ck['rng']);m.mix_cpu.set_state(ck['mix_rng']);m.exposure=ck.get('exposure',{})
    ct.step(m,o,next_batch,1502,counts,False)
    assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in expected.items()),'SAM next-step replay differs';m.check_bn()
    rr=rows('TRAIN');sam=UniformSampler(len(rr),22101,0,(len(rr)+15)//16)
    flat=[x for batch in sam for x in batch];assert len(set(flat[:len(rr)]))==len(rr)
    assert len(rr)==training_size(ROOT)
    write(ROOT/'CHECK.json',{'status':'PASS','technical_counts':counts,'parent_and_SAM_next_step':'BITWISE_PASS','complete_parent_loss_and_O2_gradient_check':'PASS','uniform_tail_carry':'PASS','parent_BN_train_O2_BN_fixed':'PASS','utc':now()});print('Basic check PASS',counts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone',nargs='?',choices=BACKBONES);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    else:train(a.backbone)
