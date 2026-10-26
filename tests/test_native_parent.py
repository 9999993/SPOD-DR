"""Native parent format compatibility; synthetic metadata, no model loading."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from native_parent import check_parent_hash, validate_parent, parent_path


class NativeParentTests(unittest.TestCase):
    def setUp(self):
        self.parent = {'backbone': 'B0', 'actual_sha256': 'a' * 64, 'path': 'local-assets/parent.pt'}
        self.current = {'data_protocol': 'SPOD_DATASET_V1', 'data_contract_sha256': 'b' * 64,
                        'execution_contract_sha256': 'c' * 64}
        # Stand-in values: these tests check metadata, not tensor loading.
        self.legacy = {'model': {'layer.weight': [1.0]}, 'backbone': 'B0'}

    def test_legacy_parent_is_accepted_unchanged(self):
        before = copy.deepcopy(self.legacy)
        self.assertIsNone(validate_parent(self.legacy, self.parent, 'B0', self.current))
        self.assertEqual(self.legacy, before)

    def test_legacy_parent_without_backbone_field(self):
        self.legacy.pop('backbone')
        self.assertIsNone(validate_parent(self.legacy, self.parent, 'B0', self.current))

    def test_matching_data_metadata_does_not_require_same_execution(self):
        checkpoint = {**self.legacy, **self.current, 'execution_contract_sha256': 'd' * 64}
        self.assertIsNone(validate_parent(checkpoint, self.parent, 'B0', self.current))

    def test_conflicting_metadata_is_not_silently_treated_as_legacy(self):
        checkpoint = {**self.legacy, **self.current, 'data_contract_sha256': 'd' * 64}
        with self.assertRaisesRegex(ValueError, 'differs from this run'):
            validate_parent(checkpoint, self.parent, 'B0', self.current)

    def test_partial_data_metadata_is_rejected(self):
        for key in ('data_protocol', 'data_contract_sha256'):
            with self.subTest(key=key):
                checkpoint = {**self.legacy, key: self.current[key]}
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    validate_parent(checkpoint, self.parent, 'B0', self.current)

    def test_wrong_backbone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'backbone mismatch'):
            validate_parent({**self.legacy, 'backbone': 'B3'}, self.parent, 'B0', self.current)
        with self.assertRaisesRegex(ValueError, 'backbone mismatch'):
            validate_parent(self.legacy, {**self.parent, 'backbone': 'B3'}, 'B0', self.current)

    def test_missing_invalid_or_empty_state_is_rejected(self):
        for value in ({}, {'model': []}, {'model': {}}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_parent(value, self.parent, 'B0', self.current)

    def test_file_hash_check_is_retained(self):
        check_parent_hash(self.parent, 'A' * 64)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            check_parent_hash(self.parent, 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'must be a SHA256'):
            check_parent_hash({**self.parent, 'actual_sha256': ''}, 'a' * 64)

    def test_relative_and_placeholder_paths(self):
        package = Path(__file__).resolve().parents[1]
        assets = package / 'local-assets'
        self.assertEqual(parent_path(self.parent, assets, package), assets / 'parent.pt')
        mapped = {**self.parent, 'path': '/SPARROW_ASSETS/native/B0.pt'}
        self.assertEqual(parent_path(mapped, assets, package), assets / 'native/B0.pt')
        absolute = {**self.parent, 'path': str(assets / 'absolute.pt')}
        self.assertEqual(parent_path(absolute, assets, package), assets / 'absolute.pt')


if __name__ == '__main__':
    unittest.main()
