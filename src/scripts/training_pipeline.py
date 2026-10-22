"""Unified preparation, training and evaluation entry using separate TRAIN, DEV and CAL roles."""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from data_protocol import (PROTOCOL, prepare, read, write, sha, evaluation_roles,
                           require_runtime, validate_prepared, run_root, verify_cache_files)

BACKBONES = ('B0','B3','ConvNeXtTiny','SwinTiny','ResNet50')
VARIANTS = ('NO_DESTYLE','NO_LOCAL','NO_SAM')
from training_resources import resolve_initialization, inspect_resources, require_resources
from native_parent import describe_parent


def task_plan(family, backbone, variant, rho, queries):
    if backbone not in BACKBONES: raise ValueError('Unsupported backbone')
    if backbone == 'SwinTiny' and family == 'sensitivity':
        raise ValueError('SwinTiny sensitivity is not supported')
    env = {}; task = backbone
    if family == 'main':
        if any(x is not None for x in (variant,rho,queries)): raise ValueError('Main uses fixed K6/rho0.025')
        directory='main'
    elif family == 'ablation':
        if variant not in VARIANTS or rho is not None or queries is not None: raise ValueError('Choose exactly one ablation variant')
        directory='ablation'
        env={'SPOD_VARIANT':variant, 'SPOD_OWNER':'B'}; task=f'{backbone}__{variant}'
    elif family=='sensitivity':
        if variant is not None or (rho is None)==(queries is None): raise ValueError('Choose either --rho or --queries')
        if rho is not None:
            if rho not in (.00625,.0125,.05,.1): raise ValueError('Use main for default rho0.025')
            task=f'{backbone}__rho_{rho:g}'
        else:
            if queries not in (2,4,8,10): raise ValueError('Use main for default K6')
            task=f'{backbone}__K_{queries}'
        directory='sensitivity'; env={'SPOD_TASK':task,'SPOD_OWNER':'B'}
    elif family in ('native', 'native_base'):
        if any(x is not None for x in (variant,rho,queries)): raise ValueError('Native has no SPOD ablations')
        directory=family
    else: raise ValueError('Unknown family')
    return {'family':family,'backbone':backbone,'task':task,'module_dir':directory,'env':env}


def initialize_metadata(root, package, initialization_dir=None):
    init = resolve_initialization(package, initialization_dir)
    p = root / 'INITIALIZATION.json'
    if p.exists() and read(p) != init:
        raise ValueError('Initialization mapping changed; use a new run directory')
    if not p.exists(): write(p, init)


def bind_task(root, plan, config):
    source_root=Path(__file__).resolve().parent
    source_hashes={str(p.relative_to(source_root)):sha(p) for p in sorted(source_root.rglob('*.py'))}
    code_sha256=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    identity={'protocol':PROTOCOL, **plan, 'code_sha256':code_sha256, 'config_sha256':sha(config),
              'data_contract_sha256':sha(root/'SPLIT_RECORD.json')}
    p=root/'EXECUTION_CONTRACT.json'
    if p.exists():
        if read(p)!=identity: raise ValueError('Run directory belongs to a different task/config/data')
    elif (root/'checkpoints').exists() or (root/'predictions').exists():
        raise ValueError('Refusing unbound historical checkpoints/predictions')
    else: write(p,identity)


def execute(plan, action):
    """Heavy imports happen only after all explicit data-role checks passed."""
    code=Path(__file__).resolve().parent/plan['module_dir']
    root=Path(os.environ['SPOD_RUN_DIR'])
    tag=plan['backbone'] if plan['family'] in ('main','native','native_base') else plan['task']
    if action=='evaluate' and read(root/'checkpoints'/tag/'DONE.json').get('status')!='PASS':
        raise ValueError('Training must complete before final selection/evaluation')
    sys.path.insert(0,str(code))
    worker=importlib.import_module('worker')
    b=plan['backbone']
    if action=='train':
        worker.train(b)
        return
    if plan['family']=='native':
        worker.finish(b)
        return
    runtime=importlib.import_module('runtime')
    infer=importlib.import_module('infer')
    dest=runtime.ROOT/'predictions'/b if plan['family']=='main' else runtime.preddir(b)
    for step in (3000,6000):
        infer.run(b,runtime.taskdir(b)/f'stage_O{step}.pt',[f'DEV_O{step}'],dest)
    selection=importlib.import_module('controller' if plan['family']=='main' else 'select_task')
    selection.choose(b)
    importlib.import_module('finish_chain').finish(b)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    def data_arguments(p):
        group=p.add_mutually_exclusive_group(required=True)
        group.add_argument('--data-config',type=Path,help='User dataset JSON: manifests, dev_folds, optional counts/hashes')
        group.add_argument('--inputs',type=Path,help='Directory: TRAIN/DEV/CAL.json and DEV_FOLDS.csv; benchmark evaluation JSONs optional')
    p=sub.add_parser('prepare',help='Validate and freeze user manifests; no model import')
    data_arguments(p);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-cache',action='store_true',help='Also hash actual image caches, without decoding')
    p=sub.add_parser('check',help='Read-only source/weight file checks; no model imports or downloads')
    p.add_argument('--family',choices=['main','ablation','sensitivity','native','native_base'],default='main')
    p.add_argument('--backbone',choices=BACKBONES,help='Omit to check every supported backbone in the family')
    p.add_argument('--initialization-dir',type=Path,help='Local initialization root; defaults to checkpoints/initialization in the package')
    p.add_argument('--parent-checkpoint',type=Path,help='Completed custom native base checkpoint; native family only')
    for action in ('train','evaluate'):
        p=sub.add_parser(action)
        data_arguments(p)
        p.add_argument('--runs',type=Path,required=True,help='New training output root, separate from existing assets')
        p.add_argument('--family',choices=['main','ablation','sensitivity','native','native_base'],default='main')
        p.add_argument('--backbone',choices=BACKBONES,required=True)
        p.add_argument('--variant',choices=VARIANTS)
        p.add_argument('--rho',type=float);p.add_argument('--queries',type=int)
        p.add_argument('--verify-cache',action='store_true')
        p.add_argument('--initialization-dir',type=Path,help='Local initialization root; defaults to checkpoints/initialization in the package')
        p.add_argument('--parent-checkpoint',type=Path,help='Completed custom native base checkpoint; native family only')
    args=parser.parse_args(argv)
    if getattr(args,'parent_checkpoint',None) is not None and (args.family != 'native' or args.backbone is None):
        parser.error('--parent-checkpoint requires --family native and one --backbone')
    if args.action == 'evaluate' and args.family == 'native_base':
        parser.error('native_base produces a parent checkpoint; evaluate the native continuation')
    if args.action=='check':
        result=inspect_resources(initialization_dir=args.initialization_dir,backbone=args.backbone,family=args.family,parent_checkpoint=args.parent_checkpoint)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['issues']: raise SystemExit(1)
        return
    if args.action=='prepare':
        record=prepare(args.inputs,args.output,data_config=args.data_config)
        if args.verify_cache:verify_cache_files(args.output)
        print(json.dumps({'protocol':PROTOCOL,'counts':record['counts'],'output':str(args.output)},ensure_ascii=False))
        return
    plan=task_plan(args.family,args.backbone,args.variant,args.rho,args.queries)
    require_resources(initialization_dir=args.initialization_dir,backbone=args.backbone,family=args.family,parent_checkpoint=args.parent_checkpoint)
    if sys.platform != 'linux':
        parser.error('Training/evaluation currently require Linux; prepare and check do not require Linux')
    if args.parent_checkpoint is not None:
        plan['parent_override'] = describe_parent(args.parent_checkpoint,args.backbone)
    os.environ['SPOD_RUNS']=str(args.runs.resolve())
    root=run_root(plan['family'],plan['task'])
    prepare(args.inputs,root,data_config=args.data_config);validate_prepared(root,production=True)
    if args.action=='evaluate' and not evaluation_roles(root):
        parser.error('Evaluation requires at least one evaluation manifest in the dataset configuration; use a new run directory when adding data')
    if args.verify_cache:verify_cache_files(root)
    package=Path(__file__).resolve().parents[2]
    initialize_metadata(root,package,args.initialization_dir)
    config_name=plan['task'] if args.family in ('ablation','sensitivity') else args.backbone
    config=(package/'configs/native_base.json' if args.family=='native_base'
            else package/'configs'/plan['module_dir']/f'{config_name}.json')
    bind_task(root,plan,config)
    os.environ.update(plan['env']);os.environ['SPOD_RUN_DIR']=str(root)
    require_runtime(root)
    execute(plan,args.action)


if __name__=='__main__':main()
