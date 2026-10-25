"""Check structural ablation outputs, gradients and training stages."""
from runtime import *
from structural import API_VERSION


def main():
    assert VARIANT in ('NO_LOCAL', 'NO_DESTYLE')
    b = 'B3' if VARIANT == 'NO_LOCAL' else 'B0'
    root = ROOT/'check'/VARIANT
    assert not (root/'CHECK.json').exists(), 'reuse the existing one-time check'
    root.mkdir(parents=True, exist_ok=True)
    counts = ct.counter()
    start, cpu = time.monotonic(), ct.cpu_seconds()
    m, o = start_parent(b)
    batch = next(iter(loader(2101, 0, 1)))
    result = pt.step(m, o, batch, 1, counts, True)
    parent_gradients = result[4]
    assert parent_gradients
    state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    del m, o
    torch.cuda.empty_cache()
    m, o = start_o2(b, state)
    o1 = o_step(m, o, batch, 1, counts, True)
    m.check_bn()
    o = m.configure(2)
    before = ct.frozen_digest(m)
    o2 = o_step(m, o, batch, 1501, counts, True)
    assert ct.frozen_digest(m) == before
    m.check_bn()
    assert o2['SAM']['active'] and abs(o2['SAM']['perturbation_norm']-.025) < 1e-6
    write(root/'CHECK.json', {'status': 'PASS', 'API_VERSION': API_VERSION,
          'variant': VARIANT, 'backbone': b, 'parent_gradients': parent_gradients,
          'O1_gradients': o1['gradients'], 'O2_gradients': o2['gradients'],
          'O2_SAM': o2['SAM'], 'P_O1_O2_connection': 'PASS',
          'fixed_O_BN_and_frozen_tensors': 'PASS', 'counts': counts,
          'seconds': time.monotonic()-start, 'cpu_seconds': ct.cpu_seconds()-cpu,
          'peak_reserved': torch.cuda.max_memory_reserved(), 'utc': now()})
    print(VARIANT, 'PASS', counts, flush=True)


if __name__ == '__main__':
    main()
