"""The ordinary Swin reference is distinct from the current FP10 ablations."""
import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import release_predict
import training_pipeline
import training_resources


def rows(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class SwinNativeReleaseTests(unittest.TestCase):
    def test_registered_inference_and_continuation(self):
        spec = release_predict.model_spec('native', 'SwinTiny')
        self.assertEqual(spec['artifact_id'], 'native/SwinTiny')
        plan = training_pipeline.task_plan('native', 'SwinTiny', None, None, None)
        self.assertEqual(plan['module_dir'], 'native')

    def test_original_selected_policy_identity(self):
        weight, policy, weight_sha, policy_sha, threshold = release_predict.selected_assets('native', 'SwinTiny')
        self.assertEqual(weight_sha, 'f10b81257b88a50e9619861d1b799221d8b50ed12f3c30f8696fb40ebe00ed14')
        self.assertEqual(policy_sha, 'd6b921347ad9e3a131d2f06151cf13f99a3a26adcca43022131715aa607cf767')
        self.assertEqual(json.loads(policy.read_text())['global_step'], 18001)
        self.assertEqual(threshold, -8.260903425261821)

    def test_five_native_references_and_matching_current_main_ci(self):
        native = {(r['backbone'], r['cohort']): r for r in rows('results/native.csv')}
        self.assertEqual(len(native), 15)
        self.assertEqual({b for b, _ in native}, {'B0', 'B3', 'ConvNeXtTiny', 'ResNet50', 'SwinTiny'})
        main = {(r['backbone'], r['cohort']): r for r in rows('results/main.csv')}
        pairs = [r for r in rows('results/ci/main_vs_native.csv') if r['backbone'] == 'SwinTiny']
        self.assertEqual(len(pairs), 12)
        for r in pairs:
            key = 'SwinTiny', r['cohort']
            self.assertEqual(r['candidate'], 'SPOD_MAIN__SwinTiny')
            self.assertEqual(r['reference'], 'NATIVE_LENGTH_MATCHED__SwinTiny')
            self.assertAlmostEqual(float(r['estimate']),
                float(main[key][r['metric']]) - float(native[key][r['metric']]), places=12)
        absolute = [r for r in rows('results/ci/native_absolute.csv') if r['backbone'] == 'SwinTiny']
        self.assertEqual(len(absolute), 12)

    def test_continuation_checks_missing_parent_explicitly(self):
        result = training_resources.inspect_resources(backbone='SwinTiny', family='native')
        self.assertEqual(result['backbones'], ['SwinTiny'])
        self.assertEqual(len(result['inputs']), 1)
        entry = result['inputs'][0]
        self.assertEqual(entry['kind'], 'native_parent')
        if not Path(entry['path']).is_file():
            self.assertEqual(result['status'], 'FILES_INCOMPLETE')
            self.assertEqual(entry['status'], 'missing')
        else:
            self.assertEqual(result['status'], 'FILES_OK')


if __name__ == '__main__':
    unittest.main()
