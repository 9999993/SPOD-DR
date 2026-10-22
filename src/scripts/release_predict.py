"""Load delivered weights directly and predict preprocessed NPY caches.

This interface does not train, select models, fit thresholds or calculate metrics.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local_file(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f'Invalid release asset: {value}')
    return path


def model_spec(family, backbone, variant=None, rho=None, queries=None):
    """Resolve exactly one registered experiment; no model imports or file reads."""
    if family not in ('main', 'ablation', 'sensitivity', 'native'):
        raise ValueError('Unsupported family')
    if backbone not in ('B0', 'B3', 'ConvNeXtTiny', 'ResNet50', 'SwinTiny'):
        raise ValueError('Unsupported backbone')
    if backbone == 'SwinTiny' and family == 'sensitivity':
        raise ValueError('SwinTiny sensitivity is not registered')
    task = backbone
    resolved_variant, axis, query_count, resolved_rho = 'FULL', '', 6, .025
    if family in ('main', 'native'):
        if any(value is not None for value in (variant, rho, queries)):
            raise ValueError('Main/native do not accept --variant, --rho or --queries')
    elif family == 'ablation':
        if variant not in ('NO_LOCAL', 'NO_DESTYLE', 'NO_SAM') or rho is not None or queries is not None:
            raise ValueError('Ablation requires exactly one --variant and no sensitivity options')
        resolved_variant = variant
        task = f'{backbone}__{variant}'
    else:
        if variant is not None or (rho is None) == (queries is None):
            raise ValueError('Sensitivity requires exactly one of --rho or --queries')
        if rho is not None:
            if type(rho) not in (int, float) or rho not in (.00625, .0125, .05, .1):
                raise ValueError('Unregistered rho; default rho=.025 belongs to main')
            resolved_rho, axis = float(rho), 'rho'
            task = f'{backbone}__rho_{rho:g}'
        else:
            if type(queries) is not int or queries not in (2, 4, 8, 10):
                raise ValueError('Unregistered query count; default K=6 belongs to main')
            query_count, axis = queries, 'K'
            task = f'{backbone}__K_{queries}'
    return {'artifact_id': f'{family}/{task}', 'family': family, 'backbone': backbone,
            'variant': resolved_variant, 'axis': axis, 'query_count': query_count,
            'rho': resolved_rho if family != 'native' else None}


def index_row(spec):
    with (ROOT / 'checkpoints/models.csv').open(encoding='utf-8-sig', newline='') as stream:
        candidates = [r for r in csv.DictReader(stream) if r['artifact_id'] == spec['artifact_id']]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one registered artifact_id')
    row = candidates[0]
    if row['family'] != spec['family'] or row['backbone'] != spec['backbone']:
        raise ValueError('Artifact identity contradicts family/backbone')
    if spec['family'] == 'ablation' and row['variant'] != spec['variant']:
        raise ValueError('Artifact variant mismatch')
    if spec['family'] == 'sensitivity':
        if (row['axis'] != spec['axis'] or int(row['K']) != spec['query_count']
                or float(row['rho']) != spec['rho']):
            raise ValueError('Artifact sensitivity parameters mismatch')
    return row


def selected_assets(family, backbone, variant=None, rho=None, queries=None):
    row = index_row(model_spec(family, backbone, variant, rho, queries))
    weight, policy_path = local_file(row['weight_file']), local_file(row['policy_file'])
    weight_sha, policy_sha = digest(weight), digest(policy_path)
    policy = read(policy_path)
    if weight_sha != row['weight_sha256'] or policy_sha != row['policy_sha256']:
        raise ValueError('Release index / actual asset SHA256 mismatch')
    if policy['model_sha256'] != weight_sha or policy['backbone'] != backbone:
        raise ValueError('POLICY belongs to another model')
    rule = policy['selection']
    if rule.get('status') != 'PASS' or rule.get('comparison') != '>=' or rule.get('scale') != 'float64 log odds':
        raise ValueError('Unsupported frozen POLICY')
    tau = -math.inf if rule['threshold'] == '-inf' else float(rule['threshold'])
    if not math.isfinite(tau) and tau != -math.inf:
        raise ValueError('Invalid POLICY threshold')
    return weight, policy_path, weight_sha, policy_sha, tau


def validate_checkpoint_metadata(checkpoint, spec):
    if (not isinstance(checkpoint, dict) or checkpoint.get('backbone') != spec['backbone']
            or not isinstance(checkpoint.get('model'), dict)):
        raise ValueError('Unexpected checkpoint structure/backbone')
    if spec['family'] == 'ablation' and checkpoint.get('variant') != spec['variant']:
        raise ValueError('Checkpoint ablation variant mismatch')
    if spec['family'] == 'sensitivity':
        # The two delivered exporters use task/rho and task_id/sam_rho respectively.
        tasks = [checkpoint[key] for key in ('task', 'task_id') if key in checkpoint]
        rhos = [checkpoint[key] for key in ('rho', 'sam_rho') if key in checkpoint]
        if (not tasks or not rhos or checkpoint.get('K') != spec['query_count']
                or any(task != spec['artifact_id'].split('/', 1)[1] for task in tasks)
                or any(rho != spec['rho'] for rho in rhos)):
            raise ValueError('Checkpoint sensitivity task/K/rho mismatch')


def load_inputs(path):
    path = path.resolve()
    records = read(path)
    if not isinstance(records, list) or not records:
        raise ValueError('Inputs must be a nonempty JSON array')
    ids = set()
    for record in records:
        if not isinstance(record.get('image_id'), str) or not record['image_id'] or record['image_id'] in ids:
            raise ValueError('image_id must be a unique nonempty string')
        ids.add(record['image_id'])
        cache = Path(record['cache_path'])
        cache = (cache if cache.is_absolute() else path.parent / cache).resolve()
        if cache.suffix.lower() != '.npy' or not cache.is_file():
            raise ValueError(f'NPY cache missing: {cache}')
        if digest(cache) != str(record['cache_sha256']).lower():
            raise ValueError(f'Cache SHA256 mismatch: {record["image_id"]}')
        record['_cache'] = cache
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['main', 'ablation', 'sensitivity', 'native'], default='main')
    parser.add_argument('--backbone', choices=['B0', 'B3', 'ConvNeXtTiny', 'ResNet50', 'SwinTiny'], required=True)
    parser.add_argument('--variant', choices=['NO_LOCAL', 'NO_DESTYLE', 'NO_SAM'])
    sensitivity = parser.add_mutually_exclusive_group()
    sensitivity.add_argument('--rho', type=float, choices=[.00625, .0125, .05, .1])
    sensitivity.add_argument('--queries', type=int, choices=[2, 4, 8, 10])
    parser.add_argument('--inputs', type=Path, help='JSON array: image_id, cache_path, cache_sha256')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--verify-load-only', action='store_true', help='Strictly load weights and write receipt; no forward pass')
    args = parser.parse_args()
    try:
        spec = model_spec(args.family, args.backbone, args.variant, args.rho, args.queries)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if not args.verify_load_only and args.inputs is None:
        parser.error('--inputs is required for predictions')
    if args.verify_load_only and args.inputs is not None:
        parser.error('omit --inputs with --verify-load-only')
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new directory')
    weight, policy_path, wh, ph, threshold = selected_assets(args.family, args.backbone, args.variant, args.rho, args.queries)
    records = [] if args.verify_load_only else load_inputs(args.inputs)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from release_models import build_model
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # Safe tensor-only loading; never fall back to arbitrary pickle execution.
    checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(checkpoint, spec)
    model = build_model(args.family, args.backbone, variant=spec['variant'], query_count=spec['query_count'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    model.to(args.device)
    output_rows = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            arrays = []
            for record in batch:
                array = np.load(record['_cache'], allow_pickle=False)
                if array.dtype != np.float32 or array.shape != (3, 512, 512) or not np.isfinite(array).all():
                    raise ValueError('Caches must be finite float32 CHW (3,512,512), already preprocessed/normalized')
                arrays.append(array)
            x = torch.from_numpy(np.stack(arrays)).to(args.device)
            raw = model(x)
            logp = torch.log_softmax(raw.double(), dim=1) if args.family == 'native' else raw
            if logp.shape != (len(batch), 5) or not torch.isfinite(logp).all():
                raise ValueError('Invalid model output')
            probabilities = logp.exp()
            if not torch.allclose(probabilities.sum(1), torch.ones(len(batch), device=args.device, dtype=probabilities.dtype), atol=1e-10, rtol=0):
                raise ValueError('Probabilities do not sum to one')
            score = torch.logsumexp(logp[:, 2:], 1) - torch.logsumexp(logp[:, :2], 1)
            grades = logp.argmax(1)
            for record, probs, lp, s, g in zip(batch, probabilities.cpu().tolist(), logp.cpu().tolist(), score.cpu().tolist(), grades.cpu().tolist()):
                output_rows.append({'image_id': record['image_id'], 'cache_sha256': record['cache_sha256'],
                    'predicted_grade': g, 'referral_log_odds': s, 'refer': s >= threshold,
                    **{f'p{i}': probs[i] for i in range(5)}, **{f'logp{i}': lp[i] for i in range(5)}})
    receipt = {'artifact_id': spec['artifact_id'], 'family': args.family, 'backbone': args.backbone,
               'variant': spec['variant'] if args.family == 'ablation' else None, 'sensitivity_axis': spec['axis'] or None,
               'local_query_count': 0 if args.family == 'native' or spec['variant'] == 'NO_LOCAL' else spec['query_count'],
               'training_rho': None if spec['variant'] == 'NO_SAM' else spec['rho'], 'weight_sha256': wh,
               'policy_sha256': ph, 'checkpoint_version': checkpoint.get('version'),
               'strict_state_load': True, 'device': args.device, 'torch': torch.__version__,
               'forward_images': len(output_rows), 'threshold_source': 'DELIVERED_POLICY_UNCHANGED',
               'training_or_calibration_performed': False, 'input_manifest_sha256': digest(args.inputs) if args.inputs else None}
    args.output.mkdir(parents=True, exist_ok=False)
    if output_rows:
        with (args.output / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
