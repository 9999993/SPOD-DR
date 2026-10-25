"""Synthetic baseline-entry tests; no model imports, images, or training."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import data_protocol
import native_parent
import training_pipeline as pipeline
import training_resources as resources


class NativeBaseEntryTests(unittest.TestCase):
    def test_base_plan_for_five_backbones(self):
        for backbone in pipeline.BACKBONES:
            with self.subTest(backbone=backbone):
                plan = pipeline.task_plan('native_base', backbone, None, None, None)
                self.assertEqual(plan['module_dir'], 'native_base')

    def test_swin_paths_match_supported_families(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SPOD_RUNS': tmp}):
            for family, task in [('native', 'SwinTiny'), ('native_base', 'SwinTiny'),
                                 ('ablation', 'SwinTiny__NO_LOCAL'),
                                 ('sensitivity', 'SwinTiny__K_4')]:
                self.assertEqual(data_protocol.run_root(family, task), Path(tmp) / family / task)

    def test_base_is_training_only_before_resource_or_model_access(self):
        with patch.object(pipeline, 'require_resources') as check, contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                pipeline.main(['evaluate', '--family', 'native_base', '--backbone', 'B0',
                               '--data-config', 'unused.json', '--runs', 'unused'])
            self.assertIn('native_base produces a parent checkpoint', err.getvalue())
            check.assert_not_called()

    def test_parent_override_restricted_to_native_and_one_backbone(self):
        for family, backbone in [('main', 'B0'), ('native_base', 'B0'), ('native', None)]:
            with self.subTest(family=family, backbone=backbone), self.assertRaises(ValueError):
                resources.inspect_resources(family=family, backbone=backbone, parent_checkpoint='unused.pt')

    def test_custom_parent_is_pinned_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'parent.pt'; f.write_bytes(b'not a real checkpoint')
            override = native_parent.describe_parent(f, 'B0')
            self.assertEqual(override['actual_sha256'], hashlib.sha256(f.read_bytes()).hexdigest())
            self.assertEqual(override['path'], str(f.resolve()))
            original = {'parent': {'path': 'release.pt', 'actual_sha256': 'a' * 64}, 'backbone': 'B0'}
            changed = native_parent.configured_parent(original, {'parent_override': override}, 'B0')
            self.assertEqual(changed['parent'], override)
            self.assertEqual(original['parent']['path'], 'release.pt')
            with self.assertRaises(ValueError):
                native_parent.configured_parent(original, {'parent_override': override}, 'B3')

    def test_custom_parent_requires_completed_base_and_matching_data(self):
        identity = {'data_protocol': data_protocol.PROTOCOL, 'data_contract_sha256': 'b' * 64}
        checkpoint = {**identity, 'global_step': 15001, 'model': {'w': [1]}, 'backbone': 'B0'}
        parent = {'backbone': 'B0', 'global_step': 15001}
        self.assertEqual(native_parent.parent_data_binding(checkpoint, parent, 'B0', identity), 'matches_run_record')
        for change in [{'global_step': 15000}, {'global_step': 21001}, {'global_step': None}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                native_parent.parent_data_binding({**checkpoint, **change}, parent, 'B0', identity)
        with self.assertRaises(ValueError):
            native_parent.parent_data_binding({**checkpoint, 'data_contract_sha256': 'c' * 64}, parent, 'B0', identity)

    def test_changed_custom_parent_cannot_reuse_run_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'SPLIT_RECORD.json').write_text('{}')
            cfg = root / 'config.json'; cfg.write_text('{}')
            f = root / 'parent.pt'; f.write_bytes(b'first')
            plan = pipeline.task_plan('native', 'B0', None, None, None)
            plan['parent_override'] = native_parent.describe_parent(f, 'B0')
            pipeline.bind_task(root, plan, cfg)
            f.write_bytes(b'second')
            plan['parent_override'] = native_parent.describe_parent(f, 'B0')
            with self.assertRaisesRegex(ValueError, 'different task/config/data'):
                pipeline.bind_task(root, plan, cfg)

    def test_five_backbone_base_and_continuation_interfaces(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            inputs = root / 'inputs'; inputs.mkdir()
            bundles = {}
            for role, count in [('TRAIN', 16), ('DEV', 10), ('CAL', 2), ('HOLDOUT', 2)]:
                bundles[role] = [{'image_id': f'{role}_{i}', 'source': 'ExampleDataset',
                    'grade': 2 * (i % 2), 'cache_path': f'cache/{role}_{i}.npy',
                    'cache_sha256': hashlib.sha256(f'{role}_{i}'.encode()).hexdigest(),
                    'component_hash': f'{role}_{i}'} for i in range(count)]
                (inputs / f'{role}.json').write_text(json.dumps(bundles[role]))
            with (inputs / 'DEV_FOLDS.csv').open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['image_id', 'fold'])
                writer.writerows((row['image_id'], i // 2) for i, row in enumerate(bundles['DEV']))
            config = root / 'dataset.json'
            config.write_text(json.dumps({'schema': 'spod_dataset_v1',
                'manifests': {role: f'inputs/{role}.json' for role in bundles},
                'dev_folds': 'inputs/DEV_FOLDS.csv'}))
            # Only the dispatch boundary is mocked; parsing, preparation, paths,
            # execution binding and metadata checks run on synthetic manifests.
            with patch.object(pipeline, 'require_resources') as resources_check, \
                 patch.object(pipeline, 'execute') as dispatch, \
                 patch.object(pipeline.sys, 'platform', 'linux'):
                for backbone in pipeline.BACKBONES:
                    common = ['--data-config', str(config), '--runs', str(root / 'runs'),
                              '--backbone', backbone]
                    pipeline.main(['train', '--family', 'native_base', *common])
                    base = root / 'runs/native_base' / backbone
                    base_id = data_protocol.checkpoint_identity(base)
                    self.assertEqual(dispatch.call_args.args[0]['family'], 'native_base')
                    self.assertEqual(resources_check.call_args.kwargs['family'], 'native_base')
                    parent = base / 'checkpoints' / backbone / 'stage_15001.pt'
                    parent.parent.mkdir(parents=True)
                    parent.write_bytes(b'synthetic parent: no model state is loaded')
                    for action in ('train', 'evaluate'):
                        pipeline.main([action, '--family', 'native', *common,
                                       '--parent-checkpoint', str(parent)])
                        plan, actual_action = dispatch.call_args.args
                        self.assertEqual(actual_action, action)
                        continuation = root / 'runs/native' / backbone
                        current_id = data_protocol.checkpoint_identity(continuation)
                        checkpoint = {**base_id, 'backbone': backbone,
                                      'global_step': 15001, 'model': {'w': [1]}}
                        self.assertEqual(native_parent.parent_data_binding(checkpoint,
                            plan['parent_override'], backbone, current_id), 'matches_run_record')
                        self.assertEqual(plan['parent_override']['path'], str(parent.resolve()))
                    self.assertFalse((continuation / 'checkpoints').exists())
                self.assertEqual(dispatch.call_count, 15)
            self.assertNotIn('torch', sys.modules)
            self.assertNotIn('timm', sys.modules)

    def test_recipe_preserves_backbone_loss_precision(self):
        cfg = json.loads((ROOT / 'configs/native_base.json').read_text())
        self.assertEqual(cfg['updates'], 15001)
        self.assertEqual(cfg['batch_size'], 16)
        self.assertEqual(cfg['seed'], 2101)
        self.assertEqual(cfg['optimizer']['lr'], 3e-5)
        self.assertEqual(set(cfg['backbones']), set(pipeline.BACKBONES))
        self.assertEqual({b for b, v in cfg['backbones'].items() if v['loss'] == 'log_softmax_float64_nll'}, {'B0', 'ConvNeXtTiny'})


class BaseResumeTests(unittest.TestCase):
    def test_resume_rejects_incompatible_state_before_loading_model(self):
        from native_base.worker import validate_resume
        identity = {'data_protocol': data_protocol.PROTOCOL, 'data_contract_sha256': 'a' * 64,
                    'execution_contract_sha256': 'b' * 64}
        checkpoint = {**identity, 'backbone': 'B0', 'global_step': 500,
                      'config_sha256': 'c' * 64, 'initialization_sha256': 'd' * 64,
                      'model': {'w': [1]}, 'optimizer': {}, 'rng': {},
                      'sampler': {'size': 31, 'seed': 2101, 'next_update': 500}}
        self.assertEqual(validate_resume(checkpoint, identity, 'B0', 'c' * 64, 'd' * 64, 31, 2101, 15001), 500)
        for change in [{'global_step': -1}, {'global_step': 15002}, {'backbone': 'B3'},
                       {'initialization_sha256': 'e' * 64}, {'config_sha256': 'e' * 64},
                       {'execution_contract_sha256': 'e' * 64},
                       {'sampler': {'size': 31, 'seed': 2101, 'next_update': 499}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_resume({**checkpoint, **change}, identity, 'B0', 'c' * 64, 'd' * 64, 31, 2101, 15001)

    def test_parent_worker_import_does_not_import_model_libraries(self):
        import subprocess
        command = "import sys;sys.path.insert(0,'src/scripts');import native_base.worker;assert 'torch' not in sys.modules;assert 'timm' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
