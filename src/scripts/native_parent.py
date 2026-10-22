"""Validate native parent metadata without changing checkpoint contents.

A parent initializes a new continuation. It is not an optimizer/RNG resume.
Legacy parents need not contain metadata introduced by the current interface.
"""
from collections.abc import Mapping
import os
import hashlib
from pathlib import Path
import re


def describe_parent(path, backbone):
    """Pin a user-supplied completed base checkpoint without loading it."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Native parent checkpoint not found: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'actual_sha256': digest,
            'backbone': backbone, 'global_step': 15001}


def configured_parent(config, execution, backbone):
    override = execution.get('parent_override')
    if override is None:
        return config
    if override.get('backbone') != backbone:
        raise ValueError('Native parent backbone mismatch')
    return {**config, 'parent': dict(override)}


def parent_path(parent, assets, package):
    value = parent.get('path')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Native parent.path must name a checkpoint file')
    value = os.path.expandvars(value)
    prefix = '/SPARROW_ASSETS'
    if value == prefix or value.startswith(prefix + '/'):
        path = Path(assets) / value[len(prefix):].lstrip('/')
    else:
        path = Path(value)
    return (path if path.is_absolute() else Path(package) / path).resolve()


def check_parent_hash(parent, actual_sha256):
    expected = parent.get('actual_sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError('Native parent.actual_sha256 must be a SHA256 digest')
    if actual_sha256.lower() != expected.lower():
        raise ValueError('Native parent hash mismatch')


def parent_data_binding(checkpoint, parent, backbone, current_identity):
    """Return recorded data status; do not manufacture metadata for legacy files."""
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get('model'), Mapping):
        raise ValueError('Native parent must contain a model state mapping')
    if not checkpoint['model']:
        raise ValueError('Native parent model state is empty')
    for identity in (parent, checkpoint):
        if 'backbone' in identity and identity['backbone'] != backbone:
            raise ValueError('Native parent backbone mismatch')
    if 'global_step' in parent and checkpoint.get('global_step') != parent['global_step']:
        raise ValueError('Native continuation requires a completed 15001-step base checkpoint')
    keys = ('data_protocol', 'data_contract_sha256')
    present = [key in checkpoint for key in keys]
    if not any(present):
        # A pinned file hash identifies the file, not its training membership.
        return 'not_recorded'
    if not all(present):
        raise ValueError('Native parent dataset metadata is incomplete')
    if any(checkpoint[key] != current_identity[key] for key in keys):
        raise ValueError('Native parent dataset metadata differs from this run')
    return 'matches_run_record'
