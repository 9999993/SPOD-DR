"""Ordinary classifiers; paper-matched suffixes, no method modules."""
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/main'))
import runtime as old
import b0 as old_b0
from native_parent import parent_path, check_parent_hash, validate_parent, configured_parent
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ASSETS = old.ASSETS
ROOT = old.ROOT
REPORT = WORK / 'reports/native'
CODE = Path(__file__).parent
BACKBONES = ('B0', 'B3', 'ConvNeXtTiny', 'SwinTiny', 'ResNet50')
TESTS = old.TESTS
P4 = old.P4
read, write, sha, now, append, csvwrite = old.read, old.write, old.sha, old.now, old.append, old.csvwrite


def config(b):
    current = read(CODE.parents[2] / 'configs' / 'native' / (b + '.json'))
    return configured_parent(current, read(ROOT / 'EXECUTION_CONTRACT.json'), b)


def taskdir(b):
    return ROOT / 'checkpoints' / b


def rows(role):
    # TRAIN_MIXED is deliberately not an accepted role.
    assert role in ('TRAIN', 'DEV', 'CAL', *TESTS)
    return old.rows(role)


def native(b, state):
    if b == 'B0':
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    else:
        import timm
        kwargs = {'img_size': 512} if b == 'SwinTiny' else {}
        m = timm.create_model(old.cm.FACTORIES[b], pretrained=False, num_classes=5, **kwargs)
    m.load_state_dict({k.removeprefix('native.'): v for k, v in state.items()}, strict=True)
    return m


def suffix(b, m):
    # Paper baselines use the entire contiguous classifier suffix.
    if b == 'B0':
        assert len(m.features) == 9
        return list(m.features.children())[4:]
    if b == 'B3':
        assert len(m.blocks) == 7
        children = list(m.named_children())
        tail = children[[n for n, _ in children].index('blocks') + 1:]
        assert [n for n, _ in tail] == ['conv_head', 'bn2', 'global_pool', 'classifier']
        return list(m.blocks.children())[3:] + [v for n, v in tail if n != 'classifier']
    return old.cm.suffix_modules(b, m, True)


def head(b, m):
    return m.classifier if b == 'B0' else m.get_classifier()


def parameter_names(m, modules):
    ids = {id(p) for module in modules for p in module.parameters()}
    return [n for n, p in m.named_parameters() if id(p) in ids]


def configure(b, m, phase):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    active = [head(b, m)] + (suffix(b, m) if phase == 2 else [])
    for module in active:
        module.train()
        for p in module.parameters():
            p.requires_grad_(True)
    for module in m.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    head_ids = {id(p) for p in head(b, m).parameters()}
    groups = []
    for is_head in (False, True):
        ps = [p for p in m.parameters() if p.requires_grad and (id(p) in head_ids) == is_head]
        if ps:
            groups.append({'params': ps, 'lr': 1e-4 if phase == 1 else 3e-5 if is_head else 1e-5,
                           'weight_decay': 1e-4})
    expected = config(b)['head_parameters'] + (config(b)['suffix_parameters'] if phase == 2 else [])
    assert set(expected) == {n for n, p in m.named_parameters() if p.requires_grad}
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def logits(b, m, x):
    # Historical B3 native continuation uses classifier.training for dropout,
    # rather than the top-level model.training flag (the prefix remains eval).
    if b == 'B3':
        features = m.global_pool(m.forward_features(x))
        features = F.dropout(features, p=m.drop_rate, training=m.classifier.training)
        return m.classifier(features)
    return m(x)


def loader(start, stop):
    rr = rows('TRAIN')
    return torch.utils.data.DataLoader(P4.P4ManifestDataset(rr),
        batch_sampler=old.UniformSampler(len(rr), 12101, start, stop),
        num_workers=int(os.environ.get('NLM_WORKERS', '6')), pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
        generator=torch.Generator().manual_seed(12101 + 710000))


def frozen_digest(m):
    active = {n for n, p in m.named_parameters() if p.requires_grad}
    return P4._state_digest({n: v for n, v in m.state_dict().items() if n not in active})


def snapshot(b, m, optimizer, pos, phase, counts):
    return {**old.checkpoint_identity(ROOT), 'version': config(b)['version'], 'backbone': b, 'model': m.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': P4._rng_state(), 'O_step': pos,
            'global_step': 15001 + pos, 'phase': phase, 'counts': dict(counts),
            'config_sha256': sha(CODE.parents[2] / 'configs' / 'native' / (b + '.json')),
            'engine_sha256': sha(__file__), 'worker_sha256': sha(CODE / 'worker.py'),
            'parent_sha256': config(b)['parent']['actual_sha256'],
            'sampler': {'seed': 12101, 'next_zero_index': pos, 'size': old.training_size(ROOT)}}


def load_model(b, path=None):
    parent = config(b)['parent']
    target = Path(path) if path is not None else parent_path(parent, ASSETS, CODE.parents[2])
    if not target.is_file():
        raise FileNotFoundError(
            f'{b}: checkpoint file not found: {target}. '
            'For a native parent, set parent.path in its native configuration; see docs/USAGE.md#native-continuation.')
    if path is None:
        check_parent_hash(parent, sha(target))
    ck = torch.load(target, map_location='cpu', weights_only=False)
    if path is not None:
        # Resuming/evaluating this run still requires its complete data identity.
        old.validate_checkpoint(ck, ROOT)
    else:
        validate_parent(ck, parent, b, old.checkpoint_identity(ROOT))
    model = native(b, ck['model'])  # strict parameter loading also checks architecture
    return model, ck


def predict(b, path, role):
    from scipy.special import logsumexp
    dest = ROOT / 'predictions' / b / (role + '.npz')
    weight_sha = sha(path)
    if dest.exists() and dest.with_suffix('.json').exists():
        receipt = read(dest.with_suffix('.json'))
        assert receipt['actual_loaded_sha'] == weight_sha
        return old.arrays(dest)
    import time
    start = time.monotonic()
    m, ck = load_model(b, path)
    m = m.cuda().eval()
    rr = rows('DEV' if role.startswith('DEV') else role)
    import p4_analysis as evaluation
    dl = torch.utils.data.DataLoader(evaluation._EvalDataset(rr), batch_size=16,
        shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(812101))
    chunks = []
    with torch.inference_mode():
        for batch in dl:
            output = logits(b, m, batch['image'].cuda(non_blocking=True))
            chunks.append(torch.log_softmax(output.double(), 1).cpu().numpy())
    lp = np.concatenate(chunks)
    assert np.isfinite(lp).all()
    probabilities = np.exp(lp)
    assert np.allclose(probabilities.sum(1), 1, atol=2e-14, rtol=0)
    z = {'logp': lp, 'probabilities': probabilities,
         'score': logsumexp(lp[:, 2:], axis=1) - logsumexp(lp[:, :2], axis=1),
         'pred': lp.argmax(1), 'grade': np.array([int(r['grade']) for r in rr]),
         'row_index': np.arange(len(rr)), 'component_hash': np.array([r['component_hash'] for r in rr]),
         'image_key': np.array([r['image_id'] for r in rr])}
    old.np_save(dest, z)
    receipt = {'backbone': b, 'role': role, 'actual_loaded_sha': weight_sha,
               'prediction_sha256': sha(dest), 'forward': len(rr),
               'seconds': time.monotonic() - start, 'global_step': ck['global_step'],
               'utc': now()}
    write(dest.with_suffix('.json'), receipt)
    print(b, role, len(rr), round(receipt['seconds'], 2), 'seconds', flush=True)
    del m, dl
    torch.cuda.empty_cache()
    return z
