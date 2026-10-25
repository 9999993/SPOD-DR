"""Synthetic file-layout tests; do not load models or contact external services."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from training_resources import BACKBONES, inspect_resources, resolve_initialization


class TrainingResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        self.weights = self.root / 'pretrained'
        self.vendor = self.package / 'src/scripts/main/vendor'
        self.payload = b'synthetic file for hashing; not a checkpoint'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.registry = {b: {'path': f'initialization/{b}/weights.bin',
                             'sha256': self.digest, 'url': f'https://example.invalid/{b}'}
                         for b in BACKBONES}
        self.put_json(self.package / 'checkpoints/public_initialization.json', self.registry)
        self.put(self.vendor / 'assets/module.py', b'# bundled test source\n')
        self.put_json(self.vendor / 'SOURCE_FILES.json', {'files': [
            {'asset_relative_path': 'module.py', 'sha256': hashlib.sha256(b'# bundled test source\n').hexdigest()}]})
        for name in ('common.py', 'assets.py', 'models.py', 'parent_models.py', 'training.py',
                     'parent_training.py', 'numerics.py', 'objectives.py', 'parent_objectives.py'):
            self.put(self.vendor / 'code' / name, b'# synthetic module\n')
        for b in BACKBONES:
            self.put(self.package / f'checkpoints/native_parents/{b}/stage_15001.pt', self.payload)
            self.put_json(self.package / f'configs/native/{b}.json', {'parent': {
                'path': f'checkpoints/native_parents/{b}/stage_15001.pt', 'actual_sha256': self.digest}})

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_json(self, path, data):
        self.put(path, json.dumps(data).encode())

    def test_default_directory_is_relative_to_package(self):
        mapping = resolve_initialization(self.package)
        self.assertEqual(Path(mapping['B0']['path']), self.package / 'checkpoints/initialization/B0/weights.bin')

    def test_custom_directory_is_shared_by_all_backbones(self):
        mapping = resolve_initialization(self.package, self.weights)
        self.assertEqual({Path(v['path']).parent.parent for v in mapping.values()}, {self.weights})

    def test_native_does_not_require_public_initialization_files(self):
        result = inspect_resources(package=self.package, family='native')
        self.assertEqual(result['status'], 'FILES_OK')
        self.assertEqual(len(result['inputs']), 5)
        self.assertTrue(all(r['kind'] == 'native_parent' for r in result['inputs']))

    def test_check_aggregates_missing_weights_without_writing(self):
        before = {str(p.relative_to(self.root)) for p in self.root.rglob('*')}
        result = inspect_resources(package=self.package, initialization_dir=self.weights)
        self.assertEqual(result['status'], 'FILES_INCOMPLETE')
        self.assertEqual(len(result['issues']), 5)
        self.assertTrue(all('Source: https://example.invalid/' in x for x in result['issues']))
        self.assertEqual(before, {str(p.relative_to(self.root)) for p in self.root.rglob('*')})

    def test_one_backbone_does_not_require_other_initializations(self):
        self.put(self.weights / 'B0/weights.bin', self.payload)
        result = inspect_resources(package=self.package, initialization_dir=self.weights, backbone='B0')
        self.assertEqual(result['status'], 'FILES_OK')
        self.assertEqual(len(result['inputs']), 1)

    def test_base_family_requires_public_initialization_for_all_five(self):
        for backbone in BACKBONES:
            self.put(self.weights / backbone / 'weights.bin', self.payload)
        result = inspect_resources(package=self.package, initialization_dir=self.weights,
                                   family='native_base')
        self.assertEqual(result['status'], 'FILES_OK')
        self.assertEqual(len(result['inputs']), 5)
        self.assertTrue(all(r['kind'] == 'public_initialization' for r in result['inputs']))

    def test_custom_native_parent_does_not_require_released_parent(self):
        parent = self.root / 'custom-parent.pt'
        self.put(parent, self.payload)
        (self.package / 'checkpoints/native_parents/B0/stage_15001.pt').unlink()
        result = inspect_resources(package=self.package, family='native', backbone='B0',
                                   parent_checkpoint=parent)
        self.assertEqual(result['status'], 'FILES_OK')
        self.assertEqual(result['inputs'][0]['path'], str(parent.resolve()))

    def test_wrong_public_file_hash_is_rejected(self):
        self.put(self.weights / 'B0/weights.bin', b'wrong bytes')
        result = inspect_resources(package=self.package, initialization_dir=self.weights, backbone='B0')
        self.assertEqual(result['inputs'][0]['status'], 'hash_mismatch')
        self.assertEqual(result['status'], 'FILES_INCOMPLETE')

    def test_modified_bundled_source_is_detected(self):
        self.put(self.vendor / 'assets/module.py', b'# different bytes')
        result = inspect_resources(package=self.package, family='native')
        self.assertTrue(any('Bundled dependency SHA256 mismatch' in x for x in result['issues']))

    def test_legacy_environment_variables_do_not_override_sources(self):
        with patch.dict(os.environ, {'SPARROW_ASSETS': '/unused-old-root', 'PUBLIC_MODEL_CACHE': '/unused-cache',
                                     'C24_ASSETS_ROOT': '/unused-old-root', 'SPOD_VENDOR_CODE': '/unused-code'}):
            self.assertEqual(inspect_resources(package=self.package, family='native')['status'], 'FILES_OK')

    def test_resource_index_cannot_escape_package(self):
        self.put_json(self.vendor / 'SOURCE_FILES.json', {'files': [
            {'asset_relative_path': '../outside.py', 'sha256': self.digest}]})
        result = inspect_resources(package=self.package, family='native')
        self.assertEqual(result['status'], 'FILES_INCOMPLETE')
        self.assertTrue(any('escapes' in x for x in result['issues']))

    def test_initialization_index_rejects_absolute_and_traversal_paths(self):
        for path in ('/server/private/model.pt', 'initialization/../model.pt'):
            with self.subTest(path=path):
                self.registry['B0']['path'] = path
                self.put_json(self.package / 'checkpoints/public_initialization.json', self.registry)
                with self.assertRaises(ValueError): resolve_initialization(self.package, self.weights)

    def test_swin_sensitivity_resources_are_checked(self):
        result = inspect_resources(package=self.package, backbone='SwinTiny', family='sensitivity')
        self.assertEqual(result['backbones'], ['SwinTiny'])


if __name__ == '__main__':
    unittest.main()
