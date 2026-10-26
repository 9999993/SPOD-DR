"""Synthetic data-interface regression tests.

Run from the repository root: python -m unittest discover -s tests -v"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import data_protocol as protocol


class DatasetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.output = self.root / 'prepared'
        self.config = self.root / 'dataset.json'
        self.bundle = {
            'TRAIN': [self.row('train', i) for i in range(3)],
            'DEV': [self.row('dev', i) for i in range(10)],
            'CAL': [self.row('cal', i) for i in range(2)],
            'HOLDOUT': [self.row('test', i) for i in range(2)],
        }
        self.options = {
            'schema': 'spod_dataset_v1',
            'manifests': {r: f'inputs/{r}.json' for r in self.bundle},
            'dev_folds': 'inputs/DEV_FOLDS.csv',
        }
        self.save_inputs()
        with (self.inputs / 'DEV_FOLDS.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'fold'])
            writer.writeheader()
            writer.writerows({'image_id': r['image_id'], 'fold': i // 2}
                             for i, r in enumerate(self.bundle['DEV']))

    @staticmethod
    def row(role, i):
        key = f'{role}_{i}'
        return {'image_id': key, 'source': 'ExampleDataset', 'grade': 2 * (i % 2),
                'cache_path': f'cache/{key}.npy',
                'cache_sha256': hashlib.sha256(key.encode()).hexdigest(),
                'component_hash': key}

    def save_inputs(self):
        for role, rows in self.bundle.items():
            (self.inputs / f'{role}.json').write_text(json.dumps(rows), encoding='utf-8')
        self.config.write_text(json.dumps(self.options), encoding='utf-8')

    def prepare(self):
        return protocol.prepare(None, self.output, data_config=self.config)

    def test_custom_counts_without_author_csv_and_relative_paths(self):
        record = self.prepare()
        self.assertEqual(record['counts'], {'TRAIN': 3, 'DEV': 10, 'CAL': 2, 'HOLDOUT': 2})
        self.assertEqual(record['evaluation_roles'], ['HOLDOUT'])
        row = protocol.prepared_rows(self.output, 'TRAIN')[0]
        self.assertEqual(row['cache_path'], str((self.inputs / 'cache/train_0.npy').resolve()))
        self.assertEqual(self.prepare(), record)
        self.assertFalse((self.inputs / 'train_manifest.csv').exists())

    def test_evaluation_manifests_optional_for_training(self):
        self.options['manifests'].pop('HOLDOUT')
        self.save_inputs()
        self.prepare()
        self.assertEqual(protocol.evaluation_roles(self.output), ())

    def test_compatible_input_directory_has_four_required_files(self):
        record = protocol.prepare(self.inputs, self.output)
        self.assertEqual(record['counts']['TRAIN'], 3)
        self.assertEqual(record['evaluation_roles'], [])

    def test_optional_expected_count_mismatch(self):
        self.options['expected_counts'] = {'TRAIN': 4}
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'expected 4 rows'):
            self.prepare()

    def test_optional_hash_check(self):
        self.options['expected_sha256'] = {'TRAIN': protocol.sha(self.inputs / 'TRAIN.json')}
        self.save_inputs()
        self.prepare()
        self.options['expected_sha256']['TRAIN'] = '0' * 64
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'input SHA256 mismatch'):
            protocol.dataset_inputs(data_config=self.config)

    def test_known_evaluation_source_rejected_from_training(self):
        self.bundle['TRAIN'][0]['source'] = 'DDR'
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'forbidden evaluation source'):
            self.prepare()

    def test_deepdrid_is_not_mistaken_for_idrid(self):
        self.bundle['TRAIN'][0]['source'] = 'DeepDRiD_Set_A'
        self.save_inputs()
        self.prepare()

    def test_duplicate_image_id_rejected_even_with_different_sources(self):
        self.bundle['TRAIN'][1]['image_id'] = self.bundle['TRAIN'][0]['image_id']
        self.bundle['TRAIN'][1]['source'] = 'AnotherDataset'
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'duplicate image identity'):
            self.prepare()

    def test_cross_role_cache_overlap_case_insensitive_hash(self):
        self.bundle['CAL'][0]['cache_sha256'] = self.bundle['TRAIN'][0]['cache_sha256'].upper()
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'cross-role overlap'):
            self.prepare()

    def test_cross_role_patient_overlap(self):
        self.bundle['TRAIN'][0]['patient_id'] = 'same_patient'
        self.bundle['CAL'][0]['patient_id'] = 'same_patient'
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'patient_id'):
            self.prepare()

    def test_patient_must_not_cross_dev_folds(self):
        self.bundle['DEV'][0]['patient_id'] = 'same_patient'
        self.bundle['DEV'][2]['patient_id'] = 'same_patient'
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'different folds'):
            self.prepare()

    def test_single_class_dev_fold_rejected(self):
        self.bundle['DEV'][1]['grade'] = 0
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'Each DEV fold'):
            self.prepare()

    def test_changed_inputs_cannot_reuse_run_directory(self):
        self.prepare()
        self.bundle['TRAIN'][0]['grade'] = 1
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'Inputs changed'):
            self.prepare()

    def test_prepared_manifest_tampering_detected(self):
        self.prepare()
        path = self.output / 'data/TRAIN.json'
        path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
            protocol.validate_prepared(self.output)

    def test_old_protocol_cannot_be_resumed(self):
        self.prepare()
        path = self.output / 'SPLIT_RECORD.json'
        record = protocol.read(path)
        record['protocol'] = 'OLD_PROTOCOL'
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'protocol/adapter mismatch'):
            protocol.validate_prepared(self.output)

    def test_ambiguous_or_unsafe_config_keys_rejected(self):
        for role in ('DEV_EXTRA', '../OTHER', 'CON', 'holdout'):
            with self.subTest(role=role):
                options = dict(self.options)
                options['manifests'] = {**self.options['manifests'], role: 'inputs/HOLDOUT.json'}
                self.config.write_text(json.dumps(options), encoding='utf-8')
                with self.assertRaises(ValueError):
                    protocol.dataset_inputs(data_config=self.config)
        self.config.write_text('{"schema":"spod_dataset_v1","schema":"spod_dataset_v1"}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            protocol.dataset_inputs(data_config=self.config)


if __name__ == '__main__':
    unittest.main()
