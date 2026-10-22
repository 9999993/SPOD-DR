"""Continuation: 6000 training updates, two-point DEV, CAL, tests."""
from engine import *
import argparse
import fcntl
import resource
import time
import traceback


def cpu_seconds():
    return sum(resource.getrusage(w).ru_utime + resource.getrusage(w).ru_stime
               for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def train(b):
    root = taskdir(b)
    root.mkdir(parents=True, exist_ok=True)
    task_lock=(root/'worker.lock').open('a+')
    fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root / 'DONE.json').exists():
        return
    start, cpu = time.monotonic(), cpu_seconds()
    status, error, pos = 'FAIL', None, 0
    counts = {'updates': 0, 'training_forward': 0, 'reverse_sweeps': 0}
    m, parent = load_model(b)
    m = m.cuda()
    parent_bn = {n: v.detach().clone() for n, v in m.named_buffers()
                 if n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    optimizer = configure(b, m, 1)
    phase = 1
    P4._set_seed(12101)
    if (root / 'latest.pt').exists():
        ck = torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
        old.validate_checkpoint(ck, ROOT)
        assert ck['config_sha256'] == sha(CODE.parents[2] / 'configs' / 'native' / (b + '.json'))
        assert ck['engine_sha256'] == sha(CODE / 'engine.py')
        assert ck['worker_sha256'] == sha(__file__)
        if ck['parent_sha256'] != config(b)['parent']['actual_sha256'] or ck['parent_data_binding'] != m._parent_data_binding:
            raise ValueError('Native resume parent identity changed')
        phase, pos, counts = ck['phase'], ck['O_step'], ck['counts']
        optimizer = configure(b, m, phase)
        m.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        P4._restore_rng_state(ck['rng'])
        append(root / 'TRAINING_CURVE.jsonl', {'event': 'RESUME', 'O_step': pos, 'utc': now()})
    else:
        append(root / 'TRAINING_CURVE.jsonl', {'event': 'PARENT_REUSE_O1_START',
            'parent_sha256': config(b)['parent']['actual_sha256'],
            'parent_data_binding': m._parent_data_binding, 'global_step': 15001,
            'seed': 12101, 'optimizer': 'new AdamW', 'BN': 'fixed parent buffers', 'utc': now()})
    first = pos
    before = frozen_digest(m)
    orbit_start, orbit_first = time.monotonic(), pos
    _, _, orbit, calibrate = P4._bootstrap_old_sources()

    def check_bn():
        current = dict(m.named_buffers())
        assert all(torch.equal(current[n], v) for n, v in parent_bn.items()), 'parent BN changed'

    def save(name, full=True):
        ck = snapshot(b, m, optimizer, pos, phase, counts) if full else {
            **old.checkpoint_identity(ROOT), 'version': config(b)['version'], 'model': m.state_dict(), 'backbone': b,
            'O_step': pos, 'global_step': 15001 + pos, 'phase': phase,
            'parent_sha256': config(b)['parent']['actual_sha256'],
            'parent_data_binding': m._parent_data_binding,
            'config_sha256': sha(CODE.parents[2] / 'configs' / 'native' / (b + '.json'))}
        P4._atomic_torch_save(ck, root / name)

    try:
        for batch in loader(pos, 6000):
            if old.ct.STOP_REQUESTED:
                save('latest.pt')
                raise InterruptedError('saved own optimizer/RNG and sampler cursor')
            pos += 1
            x, y = batch['image'].cuda(non_blocking=True), batch['grade'].cuda(non_blocking=True)
            with torch.no_grad():
                gamma = P4._normalize_from_01(orbit(P4._denormalize(x),
                    torch.ones((len(x), 1, 512, 512), device=x.device, dtype=x.dtype),
                    'gamma', 12101 * 100000 + 15001 + pos, calibrate(seed=12101)).image)
            optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(logits(b, m, x), y)
            gamma_loss = F.cross_entropy(logits(b, m, gamma), y)
            loss = .5 * (clean_loss + gamma_loss)
            assert torch.isfinite(loss)
            loss.backward()
            if pos in (1, 1501):
                grads = {n: {'present': p.grad is not None,
                             'finite': bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                             'norm': float(p.grad.norm()) if p.grad is not None else None}
                         for n, p in m.named_parameters() if p.requires_grad}
                assert all(g['present'] and g['finite'] for g in grads.values())
                assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                write(root / ('O%d_FIRST_GRADIENTS.json' % phase), {
                    'formal_step': pos, 'counts_as_training': True, 'gradients': grads,
                    'loss': float(loss.detach()), 'BN_fixed': True, 'utc': now()})
            ps = [p for p in m.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(ps, 1., error_if_nonfinite=True)
            optimizer.step()
            counts['updates'] += 1
            counts['training_forward'] += 2 * len(x)
            counts['reverse_sweeps'] += 1
            if pos % 100 == 0 or pos in (1, 1500, 1501, 3000, 6000):
                torch.cuda.synchronize()
                seconds = time.monotonic() - orbit_start
                record = {'backbone': b, 'phase': 'O%d' % phase, 'O_step': pos,
                    'global_step': 15001 + pos, 'loss': float(loss.detach()),
                    'CE_clean': float(clean_loss.detach()), 'CE_gamma': float(gamma_loss.detach()),
                    'preclip_norm': float(norm), 'lr': [g['lr'] for g in optimizer.param_groups],
                    'timed_updates': pos - orbit_first, 'seconds': seconds,
                    'seconds_per_update': seconds / (pos - orbit_first), 'counts': dict(counts),
                    'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'utc': now()}
                append(root / 'TRAINING_CURVE.jsonl', record)
                write(root / 'PROGRESS.json', record)
                check_bn()
                print(b, record['phase'], pos, 'loss', round(record['loss'], 6),
                      's/update', round(record['seconds_per_update'], 3), flush=True)
            if pos == 1500:
                assert frozen_digest(m) == before, 'frozen O1 weights or buffers changed'
                check_bn()
                # Neither configure nor AdamW construction consumes model RNG.
                rng = P4._rng_state()
                phase = 2
                optimizer = configure(b, m, phase)
                assert torch.equal(rng['torch_cpu'], torch.get_rng_state())
                assert torch.equal(rng['torch_cuda'], torch.cuda.get_rng_state())
                before = frozen_digest(m)
                append(root / 'TRAINING_CURVE.jsonl', {'event': 'O1_TO_O2', 'O_step': pos,
                    'global_step': 16501, 'optimizer': 'new AdamW', 'RNG_reset': False,
                    'BN': 'fixed', 'lr': [g['lr'] for g in optimizer.param_groups], 'utc': now()})
                orbit_start, orbit_first = time.monotonic(), pos
            if pos % 500 == 0 or pos == 1500:
                save('latest.pt')
            if pos in (3000, 6000):
                save('stage_O%d.pt' % pos, full=False)
        assert frozen_digest(m) == before, 'frozen O2 weights or buffers changed'
        check_bn()
        save('last.pt', full=False)
        assert counts == {'updates': 6000, 'training_forward': 192000, 'reverse_sweeps': 6000}
        write(root / 'DONE.json', {'status': 'PASS', 'counts': counts,
            'parent_steps_reused': 15001, 'total_steps': 21001,
            'last_sha256': sha(root / 'last.pt'), 'utc': now()})
        status = 'PASS'
    except BaseException as e:
        error = repr(e)
        traceback.print_exc()
        raise
    finally:
        write(root / ('TRAIN_COST_%d.json' % os.getpid()), {'status': status, 'error': error,
            'physical_updates': pos - first if status == 'PASS' else counts['updates'] - first,
            'seconds': time.monotonic() - start, 'cpu_seconds': cpu_seconds() - cpu,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'utc': now()})


def finish(b):
    from selection import dev_metric, ranks, tie
    from metrics import select_threshold
    root = taskdir(b)
    if (root / 'EVALUATION_DONE.json').exists():
        return
    candidates = []
    for step in (3000, 6000):
        path = root / ('stage_O%d.pt' % step)
        z = predict(b, path, 'DEV_O%d' % step)
        metric = dev_metric(z)
        assert metric['status'] == 'VALID', metric
        candidates.append({'O_step': step, 'global_step': 15001 + step,
            'path': str(path), 'sha256': sha(path), 'metric': metric})
    rank = ranks([c['metric'] for c in candidates])
    for c, value in zip(candidates, rank):
        c['R'] = float(value)
    selected = min(candidates, key=lambda c: (c['R'], *tie(c['metric'], c['O_step'])))
    write(root / 'J_SELECTION.json', {'candidates': candidates, 'selected': selected, 'utc': now()})
    original = old.arrays(ROOT / 'predictions' / b / ('DEV_O%d.npz' % selected['O_step']))
    reloaded = predict(b, selected['path'], 'DEV_RELOAD')
    assert np.array_equal(original['logp'], reloaded['logp'])
    cal = predict(b, selected['path'], 'CAL')
    policy = select_threshold(cal['score'], cal['grade'])
    assert policy['status'] == 'PASS'
    write(root / 'POLICY.json', {'backbone': b, 'model_sha256': selected['sha256'],
        'selection': policy, 'CAL_prediction_sha256': sha(ROOT / 'predictions' / b / 'CAL.npz'),
        'global_step': selected['global_step'], 'utc': now()})
    for cohort in TESTS:
        predict(b, selected['path'], cohort)
    write(root / 'EVALUATION_DONE.json', {'status': 'PASS', 'selected_DEV_reload': 'BITWISE_PASS',
        'cohorts': list(TESTS), 'model_sha256': selected['sha256'], 'utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('backbone', choices=BACKBONES)
    args = parser.parse_args()
    root = taskdir(args.backbone)
    root.mkdir(parents=True, exist_ok=True)
    train(args.backbone)
    torch.cuda.empty_cache()
    finish(args.backbone)
