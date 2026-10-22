"""Summarize a completed run at its own CAL threshold."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from data_protocol import PROTOCOL, require_runtime, read, sha, prepared_rows, evaluation_roles, training_size


def validate_predictions(z, rows):
    import numpy as np
    from scipy.special import logsumexp
    if z['image_key'].tolist()!=[r['image_id'] for r in rows]:
        raise ValueError('Predictions must cover the entire manifest in frozen row order')
    if not np.array_equal(z['grade'],[int(r['grade']) for r in rows]):
        raise ValueError('Prediction labels differ from frozen manifest')
    lp=z['logp'];p=z['probabilities']
    if lp.shape!=(len(rows),5) or not np.isfinite(lp).all():raise ValueError('Invalid log probabilities')
    if not np.allclose(np.exp(lp),p,rtol=0,atol=2e-6) or not np.allclose(p.sum(1),1,rtol=0,atol=2e-6):
        raise ValueError('Probability identity mismatch')
    expected=logsumexp(lp[:,2:],axis=1)-logsumexp(lp[:,:2],axis=1)
    if not np.allclose(expected,z['score'],rtol=0,atol=1e-12):raise ValueError('Referral score identity mismatch')
    if not np.array_equal(lp.argmax(1),z['pred']):raise ValueError('Grade prediction identity mismatch')


def summarize(root):
    import numpy as np
    root=Path(root);require_runtime(root)
    contract=read(root/'EXECUTION_CONTRACT.json')
    b=contract['backbone'];task=contract['task']
    tag=b if contract['family'] in ('main','native') else task
    ck=root/'checkpoints'/tag;pred=root/'predictions'/tag
    selection=read(ck/'J_SELECTION.json')['selected'];policy=read(ck/'POLICY.json')
    if sha(selection['path'])!=selection['sha256'] or policy['model_sha256']!=selection['sha256']:
        raise ValueError('Selected checkpoint / CAL policy mismatch')
    if sha(pred/'CAL.npz')!=policy['CAL_prediction_sha256']:
        raise ValueError('CAL prediction changed')
    with np.load(pred/'CAL.npz',allow_pickle=False) as raw:
        cal={k:raw[k] for k in raw.files}
    validate_predictions(cal,prepared_rows(root,'CAL'))
    path=Path(__file__).parent/'main/vendor/code/metrics.py'
    spec=importlib.util.spec_from_file_location('source_metrics',path)
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    fitted=metrics.select_threshold(cal['score'],cal['grade'])
    if fitted!=policy['selection']:raise ValueError('Policy was not fitted on this CAL')
    records=[]
    cohorts = evaluation_roles(root)
    if not cohorts: raise ValueError('No evaluation manifests declared for this run')
    for cohort in cohorts:
        p=pred/f'{cohort}.npz';receipt=read(p.with_suffix('.json'))
        if receipt['actual_loaded_sha']!=selection['sha256'] or receipt['prediction_sha256']!=sha(p):
            raise ValueError('Prediction provenance mismatch')
        parent_info = {'parent_data_binding': receipt.get('parent_data_binding', 'not_recorded')} if contract['family'] == 'native' else {}
        with np.load(p,allow_pickle=False) as raw:z={k:raw[k] for k in raw.files}
        validate_predictions(z,prepared_rows(root,cohort))
        value=metrics.rates(z['grade'],z['pred'],z['score'],policy['selection']['threshold'],z['probabilities'])
        records.append({'protocol':PROTOCOL,'family':contract['family'],'task':task,'backbone':b,
            'cohort':cohort,'role':'FULL_EVALUATION_NO_TRAINING_MEMBERS','training_images':training_size(root),
            'external_training_images':0,'weight_sha256':selection['sha256'],
            'policy_sha256':sha(ck/'POLICY.json'), **parent_info,
            **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in value.items()}})
    out=root/'results';out.mkdir(exist_ok=True)
    with (out/'METRICS.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(out/'METRICS.csv')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    summarize(p.parse_args(argv).run_dir)


if __name__=='__main__':main()
