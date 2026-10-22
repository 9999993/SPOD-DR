"""Resolve local training resources and check file identities without model imports."""
import os
import re
from pathlib import Path, PurePosixPath
from data_protocol import read, sha
from native_parent import parent_path, check_parent_hash, describe_parent

PACKAGE = Path(__file__).resolve().parents[2]
VENDOR = PACKAGE / 'src/scripts/main/vendor'
BUNDLED_ASSETS = VENDOR / 'assets'
BACKBONES = ('B0', 'B3', 'ConvNeXtTiny', 'SwinTiny', 'ResNet50')
FAMILIES = ('main', 'ablation', 'sensitivity', 'native', 'native_base')


def _relative_file(base, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Resource index paths must use relative forward-slash paths')
    rel = PurePosixPath(value)
    if rel.is_absolute() or '..' in rel.parts or ':' in value:
        raise ValueError('Resource index path escapes its root')
    path = (Path(base) / Path(*rel.parts)).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError('Resource path escapes its root')
    return path


def resolve_initialization(package=PACKAGE, initialization_dir=None):
    package = Path(package).resolve()
    directory = (Path(initialization_dir).expanduser() if initialization_dir is not None
                 else package / 'checkpoints/initialization').resolve()
    registry = read(package / 'checkpoints/public_initialization.json')
    if not isinstance(registry, dict) or set(registry) != set(BACKBONES):
        raise ValueError('Initialization registry must declare the five supported backbones')
    resolved = {}
    for backbone, original in registry.items():
        if not isinstance(original, dict): raise ValueError(f'{backbone}: invalid initialization entry')
        entry = dict(original)
        if 'path' not in entry or not isinstance(entry.get('sha256'), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', entry['sha256']):
            raise ValueError(f'{backbone}: initialization path/SHA256 missing or invalid')
        for key in ('path', 'config_path'):
            if key not in entry: continue
            value = entry[key]
            if not isinstance(value, str) or not value.startswith('initialization/'):
                raise ValueError('Initialization paths must be relative to checkpoints/initialization')
            entry[key] = str(_relative_file(directory, value[len('initialization/'):]))
        resolved[backbone] = entry
    return resolved


def initialization_for(backbone):
    """Consumers use the exact mapping bound by the unified training entry."""
    run = os.environ.get('SPOD_RUN_DIR')
    if not run:
        raise RuntimeError('Use src/scripts/training.py to configure training resources')
    registry = read(Path(run) / 'INITIALIZATION.json')
    if backbone not in registry:
        raise ValueError(f'No initialization mapping for {backbone}')
    return registry[backbone]


def inspect_resources(package=PACKAGE, initialization_dir=None, backbone=None, family='main', parent_checkpoint=None):
    """Read/hash files only. Does not download, write, import torch, or load models."""
    if parent_checkpoint is not None and (family != 'native' or backbone is None):
        raise ValueError('--parent-checkpoint requires --family native and one --backbone')
    package = Path(package).resolve()
    if family not in FAMILIES or (backbone is not None and backbone not in BACKBONES):
        raise ValueError('Unsupported family/backbone')
    if family == 'sensitivity' and backbone == 'SwinTiny':
        raise ValueError('SwinTiny sensitivity is not supported')
    selected = [backbone] if backbone else [b for b in BACKBONES if family != 'sensitivity' or b != 'SwinTiny']
    vendor = package / 'src/scripts/main/vendor'
    issues, inputs = [], []
    checked_sources = 0
    try:
        records = read(vendor / 'SOURCE_FILES.json')['files']
        if not isinstance(records, list) or not records:
            raise ValueError('Bundled source index is empty or invalid')
        for record in records:
            path = _relative_file(vendor / 'assets', record['asset_relative_path'])
            if not path.is_file():
                issues.append(f'Bundled dependency missing: {path}')
            elif sha(path) != record['sha256']:
                issues.append(f'Bundled dependency SHA256 mismatch: {path}')
            else: checked_sources += 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f'Cannot validate bundled dependency index: {exc}')
    for name in ('common.py', 'assets.py', 'models.py', 'parent_models.py', 'training.py',
                 'parent_training.py', 'numerics.py', 'objectives.py', 'parent_objectives.py'):
        if not (vendor / 'code' / name).is_file():
            issues.append(f'Bundled training module missing: {vendor / "code" / name}')
    if family == 'native':
        for b in selected:
            try:
                parent = (describe_parent(parent_checkpoint, b) if parent_checkpoint is not None
                          else read(package / 'configs/native' / f'{b}.json')['parent'])
                path = parent_path(parent, vendor / 'assets', package)
                if not path.is_file():
                    issues.append(f'{b}: native parent missing: {path}; see configs/native/{b}.json')
                    status = 'missing'
                else:
                    check_parent_hash(parent, sha(path)); status = 'hash_matches'
                inputs.append({'backbone': b, 'kind': 'native_parent', 'path': str(path), 'status': status})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(f'{b}: cannot validate native parent: {exc}')
    else:
        try:
            registry = resolve_initialization(package, initialization_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f'Cannot resolve initialization registry: {exc}')
            registry = {}
        for b in selected:
            if b not in registry: continue
            entry = registry[b]; path = Path(entry['path']); status = 'hash_matches'
            if not path.is_file():
                status = 'missing'
                issues.append(f'{b}: initialization missing: {path}. Source: {entry.get("url", "checkpoints/public_initialization.json")}')
            elif sha(path) != entry['sha256'].lower():
                status = 'hash_mismatch'; issues.append(f'{b}: initialization SHA256 mismatch: {path}')
            inputs.append({'backbone': b, 'kind': 'public_initialization', 'path': str(path), 'status': status})
    return {'status': 'FILES_OK' if not issues else 'FILES_INCOMPLETE',
            'family': family, 'backbones': selected, 'bundled_sources_verified': checked_sources,
            'inputs': inputs, 'issues': issues,
            'scope': 'File presence and SHA256 only; no environment, model loading or training validation.'}


def require_resources(**kwargs):
    result = inspect_resources(**kwargs)
    if result['issues']:
        raise ValueError('Training resources are incomplete:\n' + '\n'.join(result['issues']))
    return result
