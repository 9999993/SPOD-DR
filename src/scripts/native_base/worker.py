"""Native classifier base training from fixed public initialization."""
import csv
import os
from pathlib import Path
import random

from data_protocol import checkpoint_identity, read, require_runtime, sha, write
from training_resources import initialization_for

PACKAGE = Path(__file__).resolve().parents[3]


def validate_resume(checkpoint, identity, backbone, config_sha, init_sha, size, seed, updates):
    expected = {**identity, 'backbone': backbone, 'config_sha256': config_sha,
                'initialization_sha256': init_sha}
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError('Base checkpoint belongs to different data, code, configuration, or initialization')
    position = checkpoint.get('global_step')
    if type(position) is not int or not 0 <= position <= updates:
        raise ValueError('Invalid base checkpoint step')
    if checkpoint.get('sampler') != {'size': size, 'seed': seed, 'next_update': position}:
        raise ValueError('Base checkpoint sampler position differs')
    if any(k not in checkpoint for k in ('model', 'optimizer', 'rng')):
        raise ValueError('Base resume checkpoint is incomplete')
    return position


class TrainingDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import torch
        row = self.rows[index]
        x = np.load(row['cache_path'], allow_pickle=False)
        if x.shape != (3, 512, 512) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'Invalid training cache at row {index}')
        return {'image': torch.from_numpy(x), 'grade': int(row['grade'])}


class BatchSampler:
    """Deterministic epoch permutations with incomplete final batches dropped."""
    def __init__(self, size, seed, start, stop, batch_size):
        self.size, self.seed, self.start, self.stop = size, seed, start, stop
        self.batch_size = batch_size
        self.steps_per_epoch = size // batch_size
        if self.steps_per_epoch < 1:
            raise ValueError('TRAIN has fewer than one full batch')

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        import numpy as np
        previous, permutation = -1, None
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, self.steps_per_epoch)
            if epoch != previous:
                permutation = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([self.seed, epoch, 0x5034]))).permutation(self.size)
                previous = epoch
            offset = batch * self.batch_size
            yield permutation[offset:offset + self.batch_size].tolist()


def build_model(backbone, entry):
    import torch
    if backbone == 'B0':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.load_state_dict(torch.load(entry['path'], map_location='cpu', weights_only=True), strict=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
    else:
        import timm
        from safetensors.torch import load_file
        kwargs = {'img_size': 512} if backbone == 'SwinTiny' else {}
        model = timm.create_model(entry['model_id'].split('/')[1], pretrained=False, num_classes=1000, **kwargs)
        state = load_file(entry['path'])
        if backbone == 'SwinTiny':
            state = {k: v for k, v in state.items() if 'relative_position_index' not in k and 'attn_mask' not in k}
        model.load_state_dict(state, strict=True)
        model.reset_classifier(5)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def train(backbone):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import fcntl
    import numpy as np
    import torch
    from torch.nn import functional as F

    root = Path(os.environ['SPOD_RUN_DIR'])
    require_runtime(root)
    config_path = PACKAGE / 'configs/native_base.json'
    config = read(config_path)
    recipe = config['backbones'][backbone]
    seed, updates = config['seed'], config['updates']
    entry = initialization_for(backbone)
    if sha(entry['path']) != entry['sha256']:
        raise ValueError('Public initialization hash mismatch')
    identity = checkpoint_identity(root)
    config_sha = sha(config_path)
    rows = read(root / 'data/TRAIN.json')
    if len(rows) < config['batch_size']:
        raise ValueError('TRAIN has fewer than one full batch')
    output = root / 'checkpoints' / backbone
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        final = output / 'stage_15001.pt'
        if (output / 'DONE.json').exists():
            done = read(output / 'DONE.json')
            if done.get('global_step') != updates or done.get('sha256') != sha(final):
                raise ValueError('Completed base checkpoint differs from its run record')
            return
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        def seed_all():
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def rng_state():
            return {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(0)}

        def restore_rng(state):
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu'].cpu())
            torch.cuda.set_rng_state(state['torch_cuda'].cpu(), 0)

        def atomic_save(value, destination):
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            with temporary.open('wb') as stream:
                torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, destination)

        seed_all()
        model = build_model(backbone, entry).cuda()
        seed_all()
        settings = config['optimizer']
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                      weight_decay=settings['weight_decay'],
                                      betas=tuple(settings['betas']), eps=settings['eps'])
        position = 0
        latest = output / 'latest.pt'
        if latest.exists():
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            position = validate_resume(checkpoint, identity, backbone, config_sha,
                                       entry['sha256'], len(rows), seed, updates)
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            restore_rng(checkpoint['rng'])
            del checkpoint
        sampler = BatchSampler(len(rows), seed, position, updates, config['batch_size'])
        loader = torch.utils.data.DataLoader(TrainingDataset(rows), batch_sampler=sampler,
                    num_workers=recipe['workers'], pin_memory=True,
                    generator=torch.Generator().manual_seed(recipe['loader_seed']))
        model.train()

        def snapshot():
            return {**identity, 'backbone': backbone, 'global_step': position,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'rng': rng_state(), 'config_sha256': config_sha,
                    'initialization_sha256': entry['sha256'],
                    'sampler': {'size': len(rows), 'seed': seed, 'next_update': position}}

        curve = output / 'training.csv'
        with curve.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(['step', 'loss'])
            for batch in loader:
                x = batch['image'].cuda(non_blocking=True)
                y = batch['grade'].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if backbone == 'ConvNeXtTiny':
                    features = model.norm_pre(model.stages(model.stem(x)))
                    logits = model.forward_head(features)
                else:
                    logits = model(x)
                if recipe['loss'] == 'log_softmax_float64_nll':
                    logp = torch.log_softmax(logits.double(), 1)
                    loss = -logp[torch.arange(len(y), device=y.device), y].mean()
                else:
                    loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite base training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                position += 1
                if position % 100 == 0 or position in (1, updates):
                    writer.writerow([position, float(loss.detach())]); stream.flush()
                if position % config['checkpoint_every'] == 0 or position == updates:
                    atomic_save(snapshot(), latest)
        if position != updates:
            raise RuntimeError('Base training stopped before its final update')
        atomic_save({**identity, 'model': model.state_dict(), 'backbone': backbone,
                     'global_step': position, 'initialization_sha256': entry['sha256']}, final)
        write(output / 'DONE.json', {'status': 'PASS', 'global_step': position, 'sha256': sha(final)})
        print(str(final), flush=True)
    finally:
        lock.close()
