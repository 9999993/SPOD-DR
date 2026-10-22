"""Keep the paper baseline's full contiguous suffix and selected assets together."""
import ast
import csv
import json
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativeContiguousTests(unittest.TestCase):
    def test_actual_suffix_mapper(self):
        source = ROOT / 'src/scripts/native/engine.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'suffix')
        fallback = object()
        namespace = {'old': types.SimpleNamespace(cm=types.SimpleNamespace(
            suffix_modules=lambda *args: fallback))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        import torch
        from torch import nn
        b0 = types.SimpleNamespace(features=nn.Sequential(*[nn.Linear(1, 1) for _ in range(9)]))
        self.assertEqual(namespace['suffix']('B0', b0), list(b0.features.children())[4:])
        b3 = nn.Module()
        b3.blocks = nn.Sequential(*[nn.Linear(1, 1) for _ in range(7)])
        for name in ('conv_head', 'bn2', 'global_pool', 'classifier'):
            b3.add_module(name, nn.Linear(1, 1))
        self.assertEqual(namespace['suffix']('B3', b3),
                         list(b3.blocks.children())[3:] + [b3.conv_head, b3.bn2, b3.global_pool])
        self.assertIs(namespace['suffix']('ResNet50', b3), fallback)

    def test_config_scope(self):
        for backbone, prefix in [('B0', 'features.8.'), ('B3', 'blocks.6.')]:
            config = json.loads((ROOT / f'configs/native/{backbone}.json').read_text())
            self.assertTrue(any(p.startswith(prefix) for p in config['suffix_parameters']))
            self.assertEqual(config['version'], 'NATIVE_CONTIGUOUS_SUFFIX_CORRECTION')

    def test_selected_policy_identity(self):
        expected = {
            'B0': ('2dbce9ec29ef9c473baf025bec0ce4ab367b9da2198b877ee8a5dc7cebb9ff2e', 21001, -12.564110646251594),
            'B3': ('1dd208acaa6c434006a026aec830fae8d14f3496ea60c858f61d761d0365172a', 18001, -16.801475954487675)}
        for backbone, (sha, step, threshold) in expected.items():
            policy = json.loads((ROOT / f'checkpoints/native/{backbone}/POLICY.json').read_text())
            self.assertEqual((policy['model_sha256'], policy['global_step'], policy['selection']['threshold']),
                             (sha, step, threshold))

    def test_baseline_results_and_paired_ci_agree(self):
        def rows(path):
            with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))
        baseline = {(r['backbone'], r['cohort']): r for r in rows('results/native.csv')}
        main = {(r['backbone'], r['cohort']): r for r in rows('results/main.csv')}
        for backbone, qwk, fp in [('B0', .720725392847831, 1426), ('B3', .7556608106344846, 1544)]:
            row = baseline[backbone, 'DDR']
            self.assertEqual(float(row['qwk']), qwk)
            self.assertEqual(int(row['FR_numerator']), fp)
        for row in rows('results/ci/main_vs_native.csv'):
            if row['backbone'] in ('B0', 'B3'):
                key = row['backbone'], row['cohort']
                self.assertEqual(row['reference'], 'NATIVE_CONTIGUOUS_SUFFIX_CORRECTION__' + row['backbone'])
                self.assertAlmostEqual(float(row['estimate']),
                    float(main[key][row['metric']]) - float(baseline[key][row['metric']]), places=12)


if __name__ == '__main__':
    unittest.main()
