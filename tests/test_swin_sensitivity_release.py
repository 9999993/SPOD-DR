"""Swin sensitivity selection, training routing, and released asset contracts."""
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import data_protocol
import release_predict as predict
import training_pipeline as pipeline
import training_resources as resources

CASES = [(None, k, f'SwinTiny__K_{k}') for k in (2, 4, 8, 10)]
CASES += [(r, None, f'SwinTiny__rho_{r:g}') for r in (.00625, .0125, .05, .1)]


class SwinSensitivityTests(unittest.TestCase):
    def test_each_setting_resolves_for_prediction_and_training(self):
        for rho, queries, task in CASES:
            with self.subTest(task=task):
                spec = predict.model_spec('sensitivity', 'SwinTiny', rho=rho, queries=queries)
                self.assertEqual(spec['artifact_id'], 'sensitivity/' + task)
                plan = pipeline.task_plan('sensitivity', 'SwinTiny', None, rho, queries)
                self.assertEqual(plan['module_dir'], 'sensitivity')
                self.assertEqual(plan['env']['SPOD_TASK'], task)
                with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SPOD_RUNS': tmp}):
                    self.assertEqual(data_protocol.run_root('sensitivity', task),
                                     Path(tmp) / 'sensitivity' / task)

    def test_default_and_mixed_settings_are_rejected(self):
        for kwargs in ({}, {'queries': 6}, {'rho': .025}, {'rho': .05, 'queries': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                predict.model_spec('sensitivity', 'SwinTiny', **kwargs)

    def test_resource_check_includes_swin(self):
        result = resources.inspect_resources(backbone='SwinTiny', family='sensitivity')
        self.assertEqual(result['backbones'], ['SwinTiny'])

    def test_policies_configs_and_three_cohort_results_match(self):
        with (ROOT / 'results/sensitivity.csv').open(encoding='utf-8-sig', newline='') as f:
            results = list(csv.DictReader(f))
        for rho, queries, task in CASES:
            with self.subTest(task=task):
                spec = predict.model_spec('sensitivity', 'SwinTiny', rho=rho, queries=queries)
                row = predict.index_row(spec)
                config = predict.read(ROOT / row['config_file'])
                policy = predict.read(ROOT / row['policy_file'])
                self.assertEqual(config['K'], spec['query_count'])
                self.assertEqual(config['sam_rho'], spec['rho'])
                self.assertEqual(policy['model_sha256'], row['weight_sha256'])
                self.assertEqual(predict.digest(ROOT / row['policy_file']), row['policy_sha256'])
                self.assertTrue(math.isfinite(policy['selection']['threshold']))
                matched = [r for r in results if r['release_artifact_id'] == spec['artifact_id']]
                self.assertEqual({r['cohort'] for r in matched}, {'DDR', 'IDRiD', 'MESSIDOR-2'})
                self.assertEqual(len(matched), 3)


if __name__ == '__main__':
    unittest.main()
