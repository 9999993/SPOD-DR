"""One shared real NO_SAM phase/gradient/recovery check; technical cost separate."""
from runtime import *
def main():
    b='B3';root=ROOT/'check/NO_SAM';root.mkdir(parents=True,exist_ok=True)
    counts=ct.counter();start=time.monotonic();cpu=ct.cpu_seconds()
    m,o=start_parent(b);batch=next(iter(loader(2101,0,1)))
    pt.step(m,o,batch,1,counts,False)
    state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};del m,o;torch.cuda.empty_cache()
    m,o=start_o2(b,state);o_step(m,o,batch,1,counts,True);m.check_bn()
    o=m.configure(2);before=ct.frozen_digest(m);result=o_step(m,o,batch,1501,counts,True)
    assert ct.frozen_digest(m)==before;m.check_bn()
    assert result['SAM']['active'] is False and result['SAM']['perturbation_norm']==0
    ck=checkpoint(m,o,'O',1501,counts);P4._atomic_torch_save(ck,root/'step.pt')
    o_step(m,o,batch,1502,counts,False);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    del m,o;torch.cuda.empty_cache();m,o=start_o2(b,state);o=m.configure(2)
    ck=torch.load(root/'step.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);o.load_state_dict(ck['optimizer']);P4._restore_rng_state(ck['rng']);m.mix_cpu.set_state(ck['mix_rng'])
    o_step(m,o,batch,1502,counts,False)
    assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in expected.items())
    result={'status':'PASS','variant':'NO_SAM','backbone':b,'O1_O2_output_gradient_connection':'PASS','single_clean_gamma_AdamW_update':'PASS','same_parameter_groups_lr_fixed_BN':'PASS','next_step_reload':'BITWISE_PASS','counts':counts,'seconds':time.monotonic()-start,'cpu_seconds':ct.cpu_seconds()-cpu,'peak_reserved':torch.cuda.max_memory_reserved(),'utc':now()}
    write(root/'CHECK.json',result);print(result,flush=True)
if __name__=='__main__':main()
