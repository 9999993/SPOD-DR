"""User dataset manifests and run identity; no model imports or image decoding."""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

PROTOCOL = 'SPOD_DATASET_V1'
INPUT_ADAPTER = 'USER_MANIFESTS_V1'
REQUIRED_ROLES = ('TRAIN', 'DEV', 'CAL')
BENCHMARK_ROLES = ('DDR', 'IDRiD', 'MESSIDOR-2')
SOURCE_ALIASES = {'ddr': 'DDR', 'ddrexternal': 'DDR',
                  'idrid': 'IDRiD', 'idridexternaltest': 'IDRiD',
                  'messidor2': 'MESSIDOR-2'}


def read(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique_keys)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def normalize(value):
    return re.sub('[^a-z0-9]', '', str(value).casefold())


def external_source(value):
    v = normalize(value)
    return v in SOURCE_ALIASES or v in ('messidor', 'retino')


def validate_roles(roles):
    roles = list(roles)
    if not set(REQUIRED_ROLES).issubset(roles):
        raise ValueError('TRAIN, DEV and CAL manifests are required')
    if len({r.casefold() for r in roles if isinstance(r, str)}) != len(roles):
        raise ValueError('Manifest role names must be unique strings, ignoring case')
    for role in roles:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', role):
            raise ValueError('Manifest role names must use letters, digits, underscore or hyphen')
        if role not in REQUIRED_ROLES and role.upper().startswith(REQUIRED_ROLES):
            raise ValueError('Evaluation names cannot start with TRAIN, DEV or CAL')
        if role.upper() in ('CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]):
            raise ValueError('Manifest role uses a reserved filename')


def validate_bundle(bundle, expected_counts=None):
    validate_roles(bundle)
    if expected_counts is not None:
        if not isinstance(expected_counts, dict) or not set(expected_counts).issubset(bundle):
            raise ValueError('expected_counts must map declared roles to positive integers')
        if any(type(n) is not int or n < 1 for n in expected_counts.values()):
            raise ValueError('Expected counts must be positive integers')
    owner = {}; counts = {}; patient_missing = {}
    for role, records in bundle.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f'{role}: empty/invalid manifest')
        counts[role] = len(records)
        if expected_counts is not None and role in expected_counts and counts[role] != expected_counts[role]:
            raise ValueError(f'{role}: expected {expected_counts[role]} rows, got {counts[role]}')
        patient_missing[role] = 0
        image_ids = set()
        for r in records:
            if not isinstance(r, dict):
                raise ValueError(f'{role}: each row must be an object')
            for key in ('image_id', 'source', 'cache_path', 'cache_sha256', 'component_hash'):
                if not isinstance(r.get(key), str) or not r[key].strip():
                    raise ValueError(f'{role}: {key} must be a nonempty string')
            if str(r.get('grade')) not in ('0', '1', '2', '3', '4'):
                raise ValueError(f'{role}: grade must be an integer 0..4')
            if 'label' in r and str(r['label']) != str(int(int(r['grade'])>=2)):
                raise ValueError(f'{role}: binary label differs from grade>=2')
            if not re.fullmatch('[0-9a-fA-F]{64}', r['cache_sha256']):
                raise ValueError(f'{role}: invalid cache_sha256')
            if role in ('TRAIN', 'DEV', 'CAL') and external_source(r['source']):
                raise ValueError(f'{role}: forbidden evaluation source {r["source"]}')
            if role in BENCHMARK_ROLES and normalize(r['source']) != normalize(role):
                raise ValueError(f'{role}: incorrect evaluation source identity')
            if r.get('role') and str(r['role']).upper() not in (role.upper(), 'SOURCE_' + role.upper()):
                raise ValueError(f'{role}: incompatible row role {r["role"]}')
            # DEV folds and prediction files use image_id alone as the row key.
            identity = str(r['image_id'])
            if identity in image_ids:
                raise ValueError(f'{role}: duplicate image identity')
            image_ids.add(identity)
            keys = [('image_id', (normalize(r['source']), identity)), ('cache_sha256', r['cache_sha256'].lower()),
                    ('component_hash', r['component_hash']),
                    ('cache_path', os.path.normcase(str(Path(r['cache_path']).resolve())))]
            if r.get('image_sha256'):
                if not isinstance(r['image_sha256'], str) or not re.fullmatch('[0-9a-fA-F]{64}', r['image_sha256']):
                    raise ValueError(f'{role}: invalid image_sha256')
                keys.append(('image_sha256', r['image_sha256'].lower()))
            # Never infer patients from filenames. Only supplied identifiers are compared.
            for key in ('patient_id', 'exam_id'):
                if r.get(key):
                    if not isinstance(r[key], str): raise ValueError(f'{role}: {key} must be a string')
                    keys.append((key, (normalize(r['source']), r[key])))
            if not r.get('patient_id'): patient_missing[role] += 1
            for key in keys:
                previous = owner.get(key)
                if previous is not None and previous != role:
                    raise ValueError(f'cross-role overlap {previous}/{role}: {key[0]}')
                owner[key] = role
        if role in ('DEV', 'CAL') and {int(r['grade']) >= 2 for r in records} != {False, True}:
            raise ValueError(f'{role}: both grade 0/1 and grade 2/3/4 samples are required')
    return {'protocol': PROTOCOL, 'counts': counts, 'training_role': 'TRAIN',
            'external_training_images': 0, 'selection_role': 'DEV', 'calibration_role': 'CAL',
            'evaluation_roles': [r for r in bundle if r not in REQUIRED_ROLES],
            'missing_patient_ids': patient_missing,
            'limits': 'Checks supplied identities/exact hashes, not unknown patients or near duplicates.'}


def align_folds(path, dev):
    """Join by image ID, or by explicit row_index plus matching component_hash."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    dev_by_id = {str(r['image_id']): r for r in dev}
    if len(dev_by_id) != len(dev) or len(rows) != len(dev):
        raise ValueError('DEV fold size or manifest image IDs are not unique')
    has_ids = [bool(str(r.get('image_id', '')).strip()) for r in rows]
    if any(has_ids) and not all(has_ids):
        raise ValueError('DEV folds cannot mix identified and unidentified rows')
    if not all(has_ids):
        # Never join by CSV line order: row identity and component must both agree.
        dev_by_index = {}
        for r in dev:
            index = str(r.get('row_index', ''))
            if not index or index in dev_by_index:
                raise ValueError('DEV row_index must be present and unique')
            dev_by_index[index] = r
        mapped = []
        seen = set()
        for r in rows:
            index = str(r.get('row_index', ''))
            if index not in dev_by_index or index in seen:
                raise ValueError('DEV fold row_index missing, duplicated or unknown')
            original = dev_by_index[index]
            if not r.get('component_hash') or r['component_hash'] != original['component_hash']:
                raise ValueError('DEV fold component_hash does not match its row_index')
            seen.add(index)
            mapped.append({**r, 'image_id': str(original['image_id'])})
        rows = mapped
    by_id = {str(r['image_id']): r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(dev_by_id):
        raise ValueError('DEV fold IDs do not exactly match DEV manifest')
    # When redundant identity columns are present, they must agree too.
    for image_id, r in by_id.items():
        original = dev_by_id[image_id]
        for key in ('row_index', 'component_hash'):
            if r.get(key) and str(r[key]) != str(original.get(key, '')):
                raise ValueError(f'DEV fold {key} contradicts image_id')
    out = [{'image_id': str(r['image_id']), 'fold': int(by_id[str(r['image_id'])]['fold'])} for r in dev]
    if set(r['fold'] for r in out) != set(range(5)):
        raise ValueError('DEV must contain folds 0..4')
    groups = {}
    for r, f in zip(dev, out):
        for field in ('component_hash', 'cache_sha256', 'image_sha256', 'cache_path', 'patient_id', 'exam_id'):
            if not r.get(field): continue
            scope = normalize(r['source']) if field in ('patient_id','exam_id') else 'global'
            value = r[field].lower() if field.endswith('sha256') else r[field]
            if field == 'cache_path': value = os.path.normcase(str(Path(value).resolve()))
            key = (scope, field, value)
            if key in groups and groups[key] != f['fold']:
                raise ValueError('DEV patient/component appears in different folds')
            groups[key] = f['fold']
    for fold in range(5):
        for held_out in (False, True):
            classes = {int(r['grade']) >= 2 for r, f in zip(dev, out) if (f['fold'] == fold) == held_out}
            if classes != {False, True}:
                raise ValueError('Each DEV fold and its complement must contain both referral classes')
    return out


def normalize_manifest_fields(bundle):
    """Normalize known evaluation dataset aliases, retaining original dataset fields."""
    out = {}
    for role, records in bundle.items():
        if not isinstance(records, list):
            raise ValueError(f'{role}: manifest must be a JSON array')
        normalized = []
        for original in records:
            if not isinstance(original, dict):
                raise ValueError(f'{role}: each manifest row must be an object')
            row = dict(original)
            if role in BENCHMARK_ROLES:
                identities = [row[k] for k in ('source', 'dataset') if str(row.get(k, '')).strip()]
                if not identities or any(SOURCE_ALIASES.get(normalize(value)) != role for value in identities):
                    raise ValueError(f'{role}: unknown or conflicting source/dataset identity')
                row['source'] = role
            normalized.append(row)
        out[role] = normalized
    return out


def dataset_inputs(input_dir=None, data_config=None):
    """Read user-owned manifests, with optional user-specified count/hash checks."""
    if (input_dir is None) == (data_config is None):
        raise ValueError('Choose exactly one of --inputs and --data-config')
    expected_counts, expected_sha256 = None, {}
    if data_config is not None:
        path = Path(data_config).resolve()
        config = read(path)
        keys = {'schema', 'manifests', 'dev_folds', 'expected_counts', 'expected_sha256'}
        if not isinstance(config, dict) or set(config) - keys or config.get('schema') != 'spod_dataset_v1':
            raise ValueError('Invalid dataset configuration schema or unknown fields')
        manifests = config.get('manifests')
        if not isinstance(manifests, dict): raise ValueError('manifests must be a role-to-path object')
        validate_roles(manifests)
        def locate(value):
            if not isinstance(value, str) or not value.strip(): raise ValueError('Manifest paths must be nonempty strings')
            p = Path(value)
            return (p if p.is_absolute() else path.parent / p).resolve()
        inputs = {role: locate(value) for role, value in manifests.items()}
        inputs['DEV_FOLDS'] = locate(config.get('dev_folds'))
        expected_counts = config.get('expected_counts')
        expected_sha256 = config.get('expected_sha256', {})
        config_sha256 = sha(path)
    else:
        directory = Path(input_dir).resolve()
        inputs = {role: directory / f'{role}.json' for role in REQUIRED_ROLES}
        inputs.update({role: directory / f'{role}.json' for role in BENCHMARK_ROLES if (directory / f'{role}.json').is_file()})
        inputs['DEV_FOLDS'] = directory / 'DEV_FOLDS.csv'
        config_sha256 = None
    if not isinstance(expected_sha256, dict) or not set(expected_sha256).issubset(inputs):
        raise ValueError('expected_sha256 keys must be declared roles or DEV_FOLDS')
    for role, digest in expected_sha256.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest):
            raise ValueError(f'{role}: expected SHA256 must have 64 hexadecimal characters')
        if sha(inputs[role]) != digest.lower(): raise ValueError(f'{role}: input SHA256 mismatch')
    return inputs, expected_counts, config_sha256


def prepare(input_dir, output, *, data_config=None):
    inputs, expected_counts, config_sha256 = dataset_inputs(input_dir, data_config)
    roles = [r for r in inputs if r != 'DEV_FOLDS']
    output = Path(output).resolve()
    bundle = normalize_manifest_fields({r: read(inputs[r]) for r in roles})
    # Cache paths are relative to the individual manifest, not to the working directory.
    for role, records in bundle.items():
        for r in records:
            if r.get('cache_path'):
                if not isinstance(r['cache_path'], str): raise ValueError(f'{role}: cache_path must be a string')
                p=Path(r['cache_path'])
                r['cache_path']=str((p if p.is_absolute() else inputs[role].parent/p).resolve())
    audit = validate_bundle(bundle, expected_counts)
    folds = align_folds(inputs['DEV_FOLDS'], bundle['DEV'])
    hashes = {r: sha(p) for r, p in inputs.items()}
    if output.exists() and any(output.iterdir()):
        if not (output / 'SPLIT_RECORD.json').exists():
            raise ValueError('Refusing existing unmarked output directory (possible legacy run)')
        record = validate_prepared(output)
        if (record['input_sha256'] != hashes or record.get('dataset_config_sha256') != config_sha256
                or record['expected_counts'] != expected_counts):
            raise ValueError('Inputs changed; use a new output directory')
        for role, records in bundle.items():
            frozen = read(output/'data'/f'{role}.json')
            if [r['cache_path'] for r in frozen] != [r['cache_path'] for r in records]:
                raise ValueError('Resolved cache paths changed; use a new output directory')
        return record
    output.mkdir(parents=True, exist_ok=True)
    for role, records in bundle.items():
        # Preserve membership and order; absolute cache paths were resolved above.
        normalized = []
        for i, original in enumerate(records):
            r = dict(original); r['row_index'] = str(i)
            r['label']=str(int(int(r['grade'])>=2))
            path = Path(r['cache_path'])
            if not path.is_absolute(): r['cache_path'] = str((inputs[role].parent/path).resolve())
            normalized.append(r)
        write(output / 'data' / f'{role}.json', normalized)
    with (output/'DEV_FOLDS.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id', 'fold'])
        writer.writeheader(); writer.writerows(folds)
    files = {f'data/{r}.json': sha(output/'data'/f'{r}.json') for r in roles}
    files['DEV_FOLDS.csv'] = sha(output/'DEV_FOLDS.csv')
    record = {**audit, 'input_adapter': INPUT_ADAPTER,
              'input_sha256': hashes, 'files': files,
              'dataset_config_sha256': config_sha256,
              'expected_counts': expected_counts}
    write(output/'SPLIT_RECORD.json', record)
    return record


def validate_prepared(root, production=False):
    root = Path(root)
    record = read(root/'SPLIT_RECORD.json')
    if (record.get('protocol') != PROTOCOL or record.get('external_training_images') != 0
            or record.get('input_adapter') != INPUT_ADAPTER):
        raise ValueError('Prepared dataset protocol/adapter mismatch; use a new output directory')
    roles = record.get('counts', {})
    if not isinstance(roles, dict): raise ValueError('Prepared dataset counts missing')
    validate_roles(roles)
    required = {f'data/{r}.json' for r in roles} | {'DEV_FOLDS.csv'}
    if set(record.get('files', {})) != required:
        raise ValueError('Prepared manifest file set mismatch')
    for rel, digest in record['files'].items():
        if sha(root/rel) != digest:
            raise ValueError(f'Prepared input changed: {rel}')
    bundle={r: read(root/'data'/f'{r}.json') for r in roles}
    checked = validate_bundle(bundle, record['expected_counts'])
    if checked['counts'] != record['counts'] or checked['evaluation_roles'] != record.get('evaluation_roles'):
        raise ValueError('Prepared role/count metadata mismatch')
    align_folds(root/'DEV_FOLDS.csv',bundle['DEV'])
    return record


def prepared_rows(root, role):
    record = validate_prepared(root)
    if role not in record['counts']:
        raise ValueError(f'Undeclared data role: {role}')
    return read(Path(root)/'data'/f'{role}.json')


def checkpoint_identity(root):
    validate_prepared(root)
    identity={'data_protocol': PROTOCOL, 'data_contract_sha256': sha(Path(root)/'SPLIT_RECORD.json')}
    execution=Path(root)/'EXECUTION_CONTRACT.json'
    if execution.exists():identity['execution_contract_sha256']=sha(execution)
    return identity


def validate_checkpoint(checkpoint, root):
    identity = checkpoint_identity(root)
    if any(checkpoint.get(k) != v for k, v in identity.items()):
        raise ValueError('Checkpoint is historical or belongs to another training dataset; cannot reuse')


def require_runtime(root):
    validate_prepared(root, production=True)
    execution=read(Path(root)/'EXECUTION_CONTRACT.json')
    if execution.get('protocol')!=PROTOCOL or execution.get('data_contract_sha256')!=sha(Path(root)/'SPLIT_RECORD.json'):
        raise ValueError('Execution/data contract mismatch')


def verify_cache_files(root):
    """Explicit future preflight, never invoked by packaging/tests on real data."""
    record = validate_prepared(root, production=True)
    for role in record['counts']:
        for row in read(Path(root)/'data'/f'{role}.json'):
            if sha(row['cache_path']) != row['cache_sha256'].lower():
                raise ValueError(f'{role}: actual cache content differs: {row["image_id"]}')


def evaluation_roles(root):
    return tuple(validate_prepared(root)['evaluation_roles'])


def training_size(root):
    return validate_prepared(root)['counts']['TRAIN']


def run_root(family, task):
    if family not in ('main','ablation','sensitivity','native','native_base'):
        raise ValueError('Invalid family')
    if task in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', task):
        raise ValueError('Invalid task identifier')
    base = Path(os.environ.get('SPOD_RUNS', 'runs')).resolve()
    # Data/execution contracts, not folder names, reject incompatible prior runs.
    result=(base/family/task).resolve()
    if not result.is_relative_to(base/family): raise ValueError('Output escapes task namespace')
    return result
