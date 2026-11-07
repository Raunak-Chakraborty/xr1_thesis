"""Serious selector and real small torch CPU transactions; no XR1 weights/GPU.

High-step fixtures relocate genuine AdamW/scheduler state to synthetic counters.
The explicit continuation test performs real CPU optimizer steps on both paths.
Framework entrypoint tests mock model/Trainer construction and immutable assets.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from mibot.utils import claix_checkpoint as ck
from mibot.utils import claix_last2_serious as serious
from mibot.utils.claix_contract import BASELINE, FROZEN_TRAIN_SHA256, resume_fingerprint
from mibot.utils.claix_experiment import bind_experiment, load_experimental_resume
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
from mibot.utils.cosine_warmup import get_cosine_schedule_with_warmup
from mibot.utils.doosan_replay import RankOffsetSampler, make_contract
from test_claix_checkpoint import checkpoint_fixture
import test_claix_experimental as experimental_tests
from test_claix_experimental import Config
from test_claix_launch import ROOT, launch
import claix_b24_last2_serious_audit as audit
import claix_b24_storage_preflight as storage


def identity(root):
    return dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                run_id='1' * 32, run_root=str(root), zip_cache_root=str(root / 'cache'),
                zip_manifest_sha256='b' * 64, serious_variant=serious.VARIANT,
                checkpoint_interval=500, experiment_name=serious.NAME,
                source_commit='c' * 40, source_tree_sha256='d' * 64,
                scientific_config_sha256='e' * 64, inputs_sha256='f' * 64,
                pretrained_sha256=serious.PRETRAINED_SHA256, language_sha256=serious.LANGUAGE_SHA256,
                dataset_sha256=FROZEN_TRAIN_SHA256)


def toy():
    keys = sorted([f'{prefix}{index}.weight' for prefix, indices in serious.STACKS.items() for index in indices]
                  + [f'model.{name}.weight' for name in serious.INTERFACES])
    weights = {key: torch.nn.Parameter(torch.ones(2, 3)) for key in keys}
    optimizer = torch.optim.AdamW(weights.values(), lr=1., betas=(.9, .95), eps=1e-8, weight_decay=.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,
        num_training_steps=48284, max_lr=2e-5, min_lr=5e-6, warmup_lr_start=5e-7)
    return weights, optimizer, scheduler


def optimize(weights, optimizer, scheduler):
    optimizer.zero_grad()
    sum((value.square() * torch.rand_like(value)).sum() for value in weights.values()).backward()
    optimizer.step()
    scheduler.step()


def payload(root, step=500, *, real_steps=None):
    value = checkpoint_fixture(step, 48284)
    weights, optimizer, scheduler = toy()
    for _ in range(real_steps if real_steps is not None else 1):
        optimize(weights, optimizer, scheduler)
    if real_steps is not None and real_steps != step:
        raise ValueError('Real step fixture mismatch')
    value['state_dict'] = {key: p.detach().clone() for key, p in weights.items()}
    value['optimizer_states'] = [copy.deepcopy(optimizer.state_dict())]
    value['lr_schedulers'] = [scheduler.state_dict()]
    if real_steps is None:
        for state in value['optimizer_states'][0]['state'].values():
            state['step'].fill_(step)
        lr = scheduler.lr_lambdas[0](step)
        value['optimizer_states'][0]['param_groups'][0]['lr'] = lr
        value['lr_schedulers'][0].update(last_epoch=step, _step_count=step + 1, _last_lr=[lr])
    value['xr1_compact_fp32'].update(last_n=2, pretrained_sha256=serious.PRETRAINED_SHA256,
        language_sha256=serious.LANGUAGE_SHA256, source_commit=BASELINE,
        selected_tensor_count=len(weights), trainable_parameter_count=sum(p.numel() for p in weights.values()),
        selected_key_sha256=hashlib.sha256('\n'.join(sorted(weights)).encode()).hexdigest())
    value['xr1_production'].update(experiment=identity(root), rng_by_rank=[ck.rng_state()],
                                  optimizer_parameter_keys=[list(weights)])
    return value


def telemetry(target, step):
    with (target / 'telemetry.jsonl').open('w') as stream:
        stream.write(json.dumps(dict(event='train_start', step=0)) + '\n')
        for number in range(1, step + 1):
            stream.write(json.dumps(dict(event='step', step=number)) + '\n')


class SeriousConfigTest(unittest.TestCase):
    def test_exact_science_and_original_initialization(self):
        new, old = serious.recipe(), serious.recipe('claix_b24_checkpoint500')
        self.assertEqual(new['data'], old['data'])
        model = copy.deepcopy(old['model']); model['params']['last_n'] = 2
        self.assertEqual(new['model'], model)
        for key in old['trainer']:
            if key != 'exp_name':
                self.assertEqual(new['trainer'][key], old['trainer'][key], key)
        self.assertEqual(new['trainer']['max_steps'], 48284)
        self.assertEqual(new['data']['params']['train_datasets']['batch_size'], 24)
        self.assertEqual(new['trainer']['scheduler']['params']['num_training_steps'], 48284)
        self.assertEqual(new['trainer']['scheduler']['params']['num_warmup_steps'], 500)
        self.assertEqual(divmod(48284 * 24, 289703), (4, 4))
        self.assertIsNone(new['trainer']['ckpt_path'])
        self.assertIsNone(new['trainer']['default_root_dir'])
        self.assertEqual(new['model']['params']['pretrained_sha256'], serious.PRETRAINED_SHA256)
        serious.validate_science(new, new['production'])

    def test_changed_science_rejected_even_on_fresh_start(self):
        for chain, value in [('model.params.last_n', 1), ('trainer.max_steps', 30),
            ('trainer.save_interval', 250), ('trainer.optimizer.params.lr', .2),
            ('trainer.scheduler.params.num_training_steps', 500), ('trainer.accumulate_grad_batches', 2),
            ('data.params.train_datasets.batch_size', 12), ('data.params.train_datasets.use_wrench', True),
            ('model.params.model.attn_implementation', 'eager'), ('model.params.pretrained_sha256', '0' * 64)]:
            config = serious.recipe(); cursor = config
            keys = chain.split('.')
            for key in keys[:-1]: cursor = cursor[key]
            cursor[keys[-1]] = value
            with self.subTest(chain=chain), self.assertRaisesRegex(RuntimeError, 'scientific'):
                serious.validate_science(config, config['production'])

    def test_selector_separates_all_legacy_modes(self):
        profile = launch.experiment_profile(500, serious_last2=True)
        self.assertEqual(profile['name'], serious.NAME)
        self.assertEqual(profile['config'], serious.CONFIG)
        self.assertNotEqual(profile, launch.experiment_profile(500))
        self.assertNotEqual(profile, launch.experiment_profile(250))
        with self.assertRaises(RuntimeError): launch.experiment_profile(250, serious_last2=True)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            for extra in ([], ['--mode', 'experimental', '--experimental-unqualified'],
                          ['--mode', 'h100-qualification', '--qualification-last2-vram-probe']):
                with self.assertRaises(SystemExit):
                    launch.main(['--serious-last2', '--run-root', folder, *extra])

    def test_clean_source_gate_and_dirty_rejection(self):
        with patch.object(serious.subprocess, 'check_output', side_effect=['c' * 40, ' M changed\n']):
            with self.assertRaisesRegex(RuntimeError, 'clean committed'): serious.source_identity()
        with patch.object(serious.subprocess, 'check_output', side_effect=['c' * 40, '']):
            self.assertEqual(serious.source_identity()['source_commit'], 'c' * 40)

    def test_fresh_manifest_binds_source_uuid_science_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root / 'cache'; cache.mkdir()
            (cache / 'media_manifest.jsonl').write_text('synthetic\n')
            cfg = serious.recipe(); cfg['trainer']['default_root_dir'] = folder
            cfg['production']['experimental_run_id'] = '1' * 32
            science = serious.validate_science(cfg, cfg['production'])
            source = dict(source_commit='c' * 40, source_tree_sha256=launch.tree_digest())
            inputs = dict(scientific_config_sha256=science, train_sha256=FROZEN_TRAIN_SHA256,
                          zip_manifest_sha256=ck.file_sha256(cache / 'media_manifest.jsonl'), zip_cache_root=str(cache))
            manifest = dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='1' * 32, git_commit=source['source_commit'], source_tree_sha256=source['source_tree_sha256'],
                plan=dict(run_root=folder, checkpoint_interval=500, experiment_name=serious.NAME,
                          serious_variant=serious.VARIANT, serious_inputs=inputs))
            ck.atomic_json(root / 'experiment.json', manifest)
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_SERIOUS_LAST2='1',
                            XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache)), patch.object(serious, 'source_identity', return_value=source):
                bound = bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                self.assertEqual(bound['inputs_sha256'], ck.config_digest(inputs))
                for key, value in [('experimental_run_id', '2' * 32), ('serious_last2', False)]:
                    bad = copy.deepcopy(cfg); bad['production'][key] = value
                    with self.assertRaises(RuntimeError): bind_experiment(bad, bad['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                with patch.dict(os.environ, XR1_SERIOUS_LAST2='0'), self.assertRaises(RuntimeError):
                    bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))


class SeriousTransactionTest(unittest.TestCase):
    def test_actual_torch_publication_readback_receipt_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'step-000500.ckpt'; value = payload(root)
            with patch.object(torch, 'load', wraps=torch.load) as reads:
                receipt = ck.publish_checkpoint(value, path)
                self.assertEqual(reads.call_count, 1)
            loaded = ck.load_verified(path, expected_config='a' * 64, expected_experiment=identity(root))
            self.assertEqual(receipt['sha256'], ck.file_sha256(path))
            self.assertEqual(receipt['bytes'], path.stat().st_size)
            self.assertEqual(receipt, json.loads((root / 'latest.json').read_text()))
            self.assertEqual(loaded['xr1_compact_fp32']['last_n'], 2)
            self.assertEqual(loaded['xr1_replay']['rank_sample_offset'], 12000)
            self.assertEqual(set(loaded['state_dict']), set(value['state_dict']))
            with self.assertRaises(FileExistsError): ck.publish_checkpoint(value, path)

    def test_all_milestones_final_safe_stop_and_two_recovery_bounded_real_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            steps = sorted({500, 1000, 1500, *ck.SERIOUS_LAST2_MILESTONES,
                            44501, 45500, 46000, 46500, 47000, 47500, 48000, 48284})
            for step in steps:
                path = root / f'step-{step:06d}.ckpt'
                receipt = ck.publish_checkpoint(payload(root, step), path)
                if step == 44501: ck.atomic_json(root / 'safe-stop.json', receipt)
                with patch.object(torch, 'load', wraps=torch.load) as reads:
                    result = ck.rotate_checkpoints(root, expected_config='a' * 64,
                                                   retention_milestones=ck.SERIOUS_LAST2_MILESTONES)
                self.assertEqual(reads.call_count, result['verified'])
                self.assertLessEqual(reads.call_count, ck.ROTATION_VERIFY_BUDGET)
            retained = {int(p.stem[5:]) for p in root.glob('*.ckpt')}
            self.assertEqual(retained, set(ck.SERIOUS_LAST2_MILESTONES) | {44501, 48000, 48284})
            for step in retained:
                ck.load_verified(root / f'step-{step:06d}.ckpt', resumable=False)
            with self.assertRaises(RuntimeError): ck.load_verified(root / 'step-048284.ckpt')

    def test_missing_milestone_payload_receipt_and_foreign_receipt_fail_before_deletion(self):
        for damage in ('payload', 'receipt', 'foreign', 'size'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for step in (5000, 10000, 10500, 11000):
                    ck.publish_checkpoint(payload(root, step), root / f'step-{step:06d}.ckpt')
                path = root / 'step-005000.ckpt'; receipt = path.with_suffix('.verified.json')
                if damage == 'payload': path.unlink()
                elif damage == 'receipt': receipt.unlink()
                elif damage == 'size': path.write_bytes(b'bad')
                else:
                    row = json.loads(receipt.read_text()); row['experiment']['run_id'] = '2' * 32
                    receipt.write_text(json.dumps(row))
                before = {p.name for p in root.iterdir()}
                with self.assertRaisesRegex(RuntimeError, 'milestone'):
                    ck.rotate_checkpoints(root, retention_milestones=ck.SERIOUS_LAST2_MILESTONES)
                self.assertEqual(before, {p.name for p in root.iterdir()})

    def test_checkpoint_rejects_arbitrary_last_two_and_changed_metadata_state(self):
        value = payload(Path('/tmp/synthetic'))
        for mutation in ('last_n', 'source', 'base', 'language', 'variant', 'keys', 'optimizer', 'scheduler',
                         'lr', 'weight_decay', 'mapping', 'rng'):
            bad = copy.deepcopy(value)
            if mutation == 'last_n': bad['xr1_compact_fp32']['last_n'] = 1
            elif mutation == 'source': bad['xr1_compact_fp32']['source_commit'] = 'wrong'
            elif mutation == 'base': bad['xr1_compact_fp32']['pretrained_sha256'] = '0' * 64
            elif mutation == 'language': bad['xr1_compact_fp32']['language_sha256'] = '0' * 64
            elif mutation == 'variant': bad['xr1_production']['experiment']['serious_variant'] = 'arbitrary'
            elif mutation == 'keys': bad['state_dict'].pop(next(iter(bad['state_dict'])))
            elif mutation == 'optimizer': bad['optimizer_states'][0]['state'][0]['exp_avg'] = torch.zeros(2, 3, dtype=torch.bfloat16)
            elif mutation == 'scheduler': bad['lr_schedulers'][0].pop('base_lrs')
            elif mutation == 'lr': bad['optimizer_states'][0]['param_groups'][0]['lr'] = 1.
            elif mutation == 'weight_decay': bad['optimizer_states'][0]['param_groups'][0]['weight_decay'] = 0.
            elif mutation == 'mapping': bad['xr1_production'].pop('optimizer_parameter_keys')
            else: bad['xr1_production']['rng_by_rank'] = []
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                ck.validate_production_checkpoint(bad, value['xr1_replay_contract'])

    def test_foreign_uuid_source_fingerprint_and_corrupted_payload_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'step-000500.ckpt'
            ck.publish_checkpoint(payload(root), path)
            for key, value in [('run_id', '2' * 32), ('source_tree_sha256', '0' * 64),
                               ('source_commit', '0' * 40), ('pretrained_sha256', '0' * 64)]:
                expected = dict(identity(root), **{key: value})
                with self.assertRaises(RuntimeError): ck.load_verified(path, expected_experiment=expected)
            with self.assertRaises(RuntimeError): ck.load_verified(path, expected_config='0' * 64)
            with path.open('r+b') as stream: stream.write(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'checksum'): ck.load_verified(path)

    def test_e1_and_sync500_last_one_cannot_resume_serious(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for last_n in (1, 2):
                value = checkpoint_fixture(30, 48284)
                if last_n == 2:
                    value['xr1_compact_fp32']['last_n'] = 2
                    value['xr1_production']['qualification_last2_vram_probe'] = True
                path = root / str(last_n) / 'step-000030.ckpt'
                ck.publish_checkpoint(value, path)
                with self.assertRaisesRegex(RuntimeError, 'experimental run'):
                    ck.load_verified(path, expected_experiment=identity(root))

    def test_failed_serialization_keeps_previous_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = root / 'step-000500.ckpt'
            ck.publish_checkpoint(payload(root), old)
            before = (root / 'latest.json').read_bytes()
            with patch.object(torch, 'save', side_effect=OSError('disk full')), self.assertRaises(OSError):
                ck.publish_checkpoint(payload(root, 1000), root / 'step-001000.ckpt')
            self.assertEqual(before, (root / 'latest.json').read_bytes())
            ck.load_verified(old)

    def test_actual_optimizer_rng_resume_and_sampler_cursor_equal_continuous(self):
        saved_rng = ck.rng_state()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); target = root / 'project_xr1_doosan' / serious.NAME
                value = payload(root, 3, real_steps=3); path = target / 'step-000003.ckpt'
                ck.publish_checkpoint(value, path); telemetry(target, 3)
                source = {key: identity(root)[key] for key in ('source_commit', 'source_tree_sha256')}
                with patch.object(serious, 'source_identity', return_value=source):
                    loaded = load_experimental_resume(path, value['xr1_replay_contract'], 'a' * 64, identity(root))
                outputs = []
                for state in (value, loaded):
                    weights, optimizer, scheduler = toy()
                    for name, p in weights.items(): p.data.copy_(state['state_dict'][name])
                    optimizer.load_state_dict(copy.deepcopy(state['optimizer_states'][0]))
                    scheduler.load_state_dict(state['lr_schedulers'][0])
                    ck.restore_rng(state['xr1_production']['rng_by_rank'][0])
                    noise = (random.random(), np.random.random())
                    optimize(weights, optimizer, scheduler)
                    outputs.append((weights, optimizer, scheduler, noise))
                for name in outputs[0][0]:
                    self.assertTrue(torch.equal(outputs[0][0][name], outputs[1][0][name]))
                self.assertEqual(outputs[0][2].state_dict(), outputs[1][2].state_dict())
                self.assertEqual(outputs[0][3], outputs[1][3])
                self.assertEqual(outputs[1][2].last_epoch, 4)
                dataset = range(48284 * 24)
                def sampler(offset):
                    base = torch.utils.data.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
                    return RankOffsetSampler(base, offset, batch_size=24)
                continuous = iter(sampler(0))
                for _ in range(72): next(continuous)
                resumed = iter(sampler(loaded['xr1_replay']['rank_sample_offset']))
                self.assertEqual([next(continuous) for _ in range(48)], [next(resumed) for _ in range(48)])
        finally:
            ck.restore_rng(saved_rng)

    def test_no_silent_rollback_or_wrong_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / 'project_xr1_doosan' / serious.NAME
            paths = [target / f'step-{step:06d}.ckpt' for step in (2, 3)]
            for step, path in zip((2, 3), paths): ck.publish_checkpoint(payload(root, step), path)
            telemetry(target, 3)
            with self.assertRaises(RuntimeError): launch.check_resume_directory(target, paths[0])
            launch.check_resume_directory(target, paths[1])
            telemetry(target, 4)
            with self.assertRaisesRegex(RuntimeError, 'Telemetry'): launch.check_resume_directory(target, paths[1])
            telemetry(target, 3)
            receipt = json.loads((target / 'latest.json').read_text())
            (target / 'step-000004.ckpt').write_bytes(b'orphan')
            with self.assertRaisesRegex(RuntimeError, 'ahead'): serious.check_resume_evidence(target, receipt)

    def test_launcher_same_run_manifest_source_and_recomputed_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / 'project_xr1_doosan' / serious.NAME
            cfg = serious.recipe(); cfg['trainer']['default_root_dir'] = folder
            experiment = identity(root)
            experiment.update(source_tree_sha256=launch.tree_digest(),
                              scientific_config_sha256=serious.validate_science(cfg, cfg['production']))
            inputs = {key: experiment[key] for key in ('source_commit', 'source_tree_sha256',
                'scientific_config_sha256', 'pretrained_sha256', 'language_sha256', 'zip_manifest_sha256', 'zip_cache_root')}
            inputs['train_sha256'] = FROZEN_TRAIN_SHA256
            experiment['inputs_sha256'] = ck.config_digest(inputs)
            plan = dict(run_root=folder, run_id=experiment['run_id'], checkpoint_interval=500,
                        experiment_name=serious.NAME, serious_variant=serious.VARIANT, serious_inputs=inputs, overrides=[])
            manifest = dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id=experiment['run_id'], plan=plan, git_commit=experiment['source_commit'],
                source_tree_sha256=experiment['source_tree_sha256'])
            ck.atomic_json(root / 'experiment.json', manifest)
            fingerprint = resume_fingerprint(cfg, dict(sha256=FROZEN_TRAIN_SHA256), 12288, experiment=experiment)
            value = payload(root, 3)
            value['xr1_production'].update(experiment=experiment, config_sha256=fingerprint)
            path = target / 'step-000003.ckpt'; ck.publish_checkpoint(value, path); telemetry(target, 3)
            with patch.object(launch, 'checkout_identity', return_value=dict(git_commit=experiment['source_commit'])), \
                 patch.object(audit, 'compose', return_value=cfg):
                accepted = launch.check_experimental_resume(root, path, plan)
                self.assertEqual(accepted['run_id'], experiment['run_id'])
                with self.assertRaises(RuntimeError): launch.check_experimental_resume(root, path, dict(plan, run_id='0' * 32))
                cfg['trainer']['scheduler']['params']['num_training_steps'] = 300
                with self.assertRaisesRegex(RuntimeError, 'configuration'):
                    launch.check_experimental_resume(root, path, plan)
            with patch.object(launch, 'checkout_identity', return_value=dict(git_commit='0' * 40)):
                with self.assertRaisesRegex(RuntimeError, 'source/configuration'):
                    launch.check_experimental_resume(root, path, plan)


class SeriousRuntimeTest(unittest.TestCase):
    def boundary(self, root, step, *, stop=False):
        value = payload(root, step)
        plugin = VerifiedCheckpointIO(expected_config='a' * 64, expected_experiment=identity(root), instrument=True)
        module = SimpleNamespace(_compact_cfg=dict(config_sha256='a' * 64, experiment=identity(root)),
            _replay_contract=value['xr1_replay_contract'], parameters=lambda: [torch.nn.Parameter(torch.ones(1)),
                torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)])
        def save(path, weights_only):
            value['xr1_production']['rng_by_rank'] = module._checkpoint_rng
            plugin.save_checkpoint(value, path)
        trainer = SimpleNamespace(global_step=step, world_size=1, is_global_zero=True, max_steps=48284,
            optimizers=[SimpleNamespace(param_groups=[dict(lr=2e-5)])], save_checkpoint=save,
            strategy=SimpleNamespace(barrier=lambda *_: None, checkpoint_io=plugin), should_stop=False)
        runtime = ProductionRuntime(root, deadline=time.time() + 3600, interval=500,
                                    retention_milestones=ck.SERIOUS_LAST2_MILESTONES)
        runtime.started = runtime.fit_started = runtime.previous_end = time.perf_counter() - .1
        runtime.previous_step = step - 1; runtime.initial_step = 0
        runtime.retained, runtime.views, runtime.tokens = 24, 48, 100
        runtime.data_pipeline_seconds = .01
        if stop: runtime.stop.handle(signal.SIGUSR1, None)
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
        return runtime, trainer, module

    def test_sync500_only_and_immediate_safe_stop(self):
        for step in (250, 500, 750, 1000):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                runtime, _, _ = self.boundary(root, step)
                self.assertEqual(runtime.last_saved, step if step % 500 == 0 else -1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); runtime, trainer, module = self.boundary(root, 751, stop=True)
            self.assertTrue(trainer.should_stop)
            self.assertEqual(json.loads((root / 'safe-stop.json').read_text())['global_step'], 751)
            with contextlib.redirect_stdout(io.StringIO()): runtime.on_fit_end(trainer, module)

    def test_milestone_and_final_runtime_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in (*ck.SERIOUS_LAST2_MILESTONES, 48284):
                runtime, trainer, module = self.boundary(root, step)
                self.assertEqual(runtime.last_saved, step)
            with contextlib.redirect_stdout(io.StringIO()): runtime.on_fit_end(trainer, module)
            self.assertTrue((root / 'step-048284.ckpt').is_file())
            self.assertEqual(runtime.retention_milestones, ck.SERIOUS_LAST2_MILESTONES)

    def test_missing_pin_and_space_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'milestone'): self.boundary(root, 10000)
            self.assertFalse(list(root.glob('*.ckpt')))
            stat = SimpleNamespace(f_bavail=1, f_frsize=1024, f_favail=20000)
            with patch.object(serious.os, 'statvfs', return_value=stat), self.assertRaisesRegex(RuntimeError, 'headroom'):
                self.boundary(root, 500)
            self.assertFalse(list(root.glob('*.ckpt')))


class SeriousEntrypointTest(unittest.TestCase):
    def test_actual_entrypoint_fresh_and_same_run_resume_retains_recipe_and_runtime(self):
        # Execute the real function body. Only framework/model construction,
        # GPU properties and unavailable immutable file hashes are substituted.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root / 'cache'; cache.mkdir()
            (cache / 'media_manifest.jsonl').write_text('synthetic\n')
            cfg = serious.recipe(); cfg['trainer']['default_root_dir'] = folder
            cfg['production']['experimental_run_id'] = '1' * 32
            source = dict(source_commit='c' * 40, source_tree_sha256=launch.tree_digest())
            inputs = dict(scientific_config_sha256=serious.validate_science(cfg, cfg['production']),
                          train_sha256=FROZEN_TRAIN_SHA256,
                          zip_manifest_sha256=ck.file_sha256(cache / 'media_manifest.jsonl'), zip_cache_root=str(cache))
            ck.atomic_json(root / 'experiment.json', dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='1' * 32, git_commit=source['source_commit'], source_tree_sha256=source['source_tree_sha256'],
                plan=dict(run_root=folder, checkpoint_interval=500, experiment_name=serious.NAME,
                          serious_variant=serious.VARIANT, serious_inputs=inputs)))
            inventory = dict(episodes=255, frames=289703, sha256=FROZEN_TRAIN_SHA256)
            model_builder = Mock(return_value=SimpleNamespace())
            trainer = Mock(); trainer.train_dataloader = None
            trainer_type = Mock(return_value=trainer)
            tree = ast.parse((ROOT / 'xr1/tools/train_doosan_compact.py').read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            main.decorator_list = []; main.args.args[0].annotation = None
            main.body = [node for node in main.body if not (isinstance(node, ast.FunctionDef) and node.name == '_file_digest')]
            namespace = dict(Config=Config, OmegaConf=SimpleNamespace(to_container=lambda value, **_: copy.deepcopy(value)),
                torch=torch, os=os, Path=Path, json=json, BASELINE=BASELINE, resume_fingerprint=resume_fingerprint,
                split_inventory=Mock(return_value=inventory), seed_everything=Mock(),
                DDPStrategy=lambda **_: 'synthetic-ddp', make_contract=make_contract,
                ModelCheckpoint=Mock(), ModelSummary=Mock(), CSVLogger=Mock(), ProductionRuntime=ProductionRuntime,
                VerifiedCheckpointIO=VerifiedCheckpointIO, DATASETS=SimpleNamespace(build=Mock()),
                MIMODEL=SimpleNamespace(build=model_builder), Trainer=trainer_type,
                _file_digest=lambda path: serious.PRETRAINED_SHA256 if path == cfg['model']['params']['pretrained'] else serious.LANGUAGE_SHA256)
            exec(compile(ast.Module(body=[main], type_ignores=[]), 'train_doosan_compact.py', 'exec'), namespace)
            shutdown = SimpleNamespace(shutdown_loader=Mock())
            # Initialize torch's lazy optimizer imports before temporarily
            # replacing framework modules; patch.dict restores sys.modules.
            value = payload(root, 3, real_steps=3)
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_SERIOUS_LAST2='1',
                            XR1_SAFE_STOP_EPOCH=str(time.time() + 1000), XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache),
                            WORLD_SIZE='1', MAX_LENGTH='12288'), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.cuda, 'get_device_name', return_value='SYNTHETIC H100'), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=96 * 1024**3)), \
                 patch.object(serious, 'source_identity', return_value=source), \
                 patch.dict(sys.modules, {'mibot.data.datamodule.doosan_datamodule': shutdown}), \
                 contextlib.redirect_stdout(io.StringIO()):
                namespace['main'](cfg)
                params = model_builder.call_args.args[0]['params']
                self.assertEqual(params['compact_fp32']['last_n'], 2)
                self.assertEqual(params['compact_fp32']['resume_step'], 0)
                self.assertEqual(params['pretrained'], cfg['model']['params']['pretrained'])
                self.assertEqual(params['optimizer'], cfg['trainer']['optimizer'])
                self.assertEqual(params['scheduler'], cfg['trainer']['scheduler'])
                self.assertIsNone(trainer.fit.call_args.kwargs['ckpt_path'])
                runtime = trainer_type.call_args.kwargs['callbacks'][1]
                self.assertEqual(runtime.interval, 500)
                self.assertEqual(runtime.retention_milestones, ck.SERIOUS_LAST2_MILESTONES)
                self.assertEqual(trainer_type.call_args.kwargs['max_steps'], 48284)
                target = root / 'project_xr1_doosan' / serious.NAME
                start = json.loads(next(target.glob('experimental-start-*.json')).read_text())
                self.assertEqual(start['initial_checkpoint_mode'], 'original_pretrained_step_zero')
                value['xr1_production'].update(config_sha256=start['config_sha256'], experiment=start['experiment'])
                path = target / 'step-000003.ckpt'
                ck.publish_checkpoint(value, path); telemetry(target, 3)
                before = {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()}
                resumed = copy.deepcopy(cfg); resumed['trainer']['ckpt_path'] = str(path)
                namespace['main'](resumed)
                self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], str(path))
                self.assertEqual(model_builder.call_args.args[0]['params']['compact_fp32']['resume_step'], 3)
                self.assertEqual(model_builder.call_args.args[0]['params']['scheduler'], cfg['trainer']['scheduler'])
                for name, content in before.items(): self.assertEqual((target / name).read_bytes(), content)
                model_builder.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'step-zero'): namespace['main'](cfg)
                model_builder.assert_not_called()


class SeriousWrapperAuditTest(unittest.TestCase):
    def test_audit_binds_verified_inputs_and_rejects_changed_base_cache_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root / 'cache'; cache.mkdir()
            (cache / 'media_manifest.jsonl').write_text('synthetic\n')
            for name in ('train', 'val', 'test'): (root / name).mkdir()
            for name in ('weights', 'language'): (root / name).write_bytes(b'synthetic')
            cfg = serious.recipe()
            cfg['data']['params']['train_datasets']['paths'] = [str(root / 'train')]
            cfg['model']['params'].update(pretrained=str(root / 'weights'), language_policy_path=str(root / 'language'))
            overrides = ['data.params.train_datasets.paths=' + json.dumps([str(root / 'train')]),
                         'model.params.pretrained=' + json.dumps(str(root / 'weights')),
                         'model.params.language_policy_path=' + json.dumps(str(root / 'language')),
                         'trainer.default_root_dir=null']
            (root / 'paths.json').write_text(json.dumps(overrides))
            splits = dict(train=FROZEN_TRAIN_SHA256, validation='1' * 64, test='2' * 64)
            (root / 'splits.json').write_text(json.dumps(splits))
            source = dict(source_commit='c' * 40, source_tree_sha256=launch.tree_digest())
            request = dict(run_root=str(root / 'new'), expected_commit=source['source_commit'], python=sys.executable,
                overrides=str(root / 'paths.json'), validation_root=str(root / 'val'), test_root=str(root / 'test'),
                split_identities=str(root / 'splits.json'), cpu_acceptance=str(root / 'cpu.json'),
                zip_cache_root=str(cache), zip_manifest_sha256=ck.file_sha256(cache / 'media_manifest.jsonl'),
                filesystem_source='synthetic', project_id=1, resume=None)
            hashes = {str(root / 'weights'): serious.PRETRAINED_SHA256, str(root / 'language'): serious.LANGUAGE_SHA256}
            real_hash = ck.file_sha256
            with patch.object(audit, 'source_identity', return_value=source), \
                 patch.object(audit, 'compose', return_value=cfg), \
                 patch.object(audit, 'audit_splits', return_value=[dict(sha256=value) for value in splits.values()]), \
                 patch.object(audit, 'file_sha256', side_effect=lambda path: hashes[str(path)] if str(path) in hashes else real_hash(path)), \
                 patch.object(audit, 'inspect_directory', return_value=dict(filesystem=dict(fstype='lustre'))), \
                 patch.object(audit, 'check_cpu_acceptance', return_value='3' * 64), \
                 patch.object(audit, 'check_active_runs'), \
                 patch.object(audit, 'serious_evaluate', return_value=dict(status='PASS')), \
                 patch.dict(os.environ, XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache)):
                inputs, storage_result = audit.collect_inputs(request)
                self.assertEqual(inputs['pretrained_sha256'], serious.PRETRAINED_SHA256)
                report = dict(schema='claix-last2-serious-audit-v1', status='PASS', request=request,
                              inputs=inputs, storage=storage_result)
                report_path = root / 'audit.json'; report_path.write_text(json.dumps(report))
                plan = dict(run_root=request['run_root'], filesystem_source='synthetic', project_id=1, overrides=overrides)
                self.assertEqual(audit.validate_report(report_path, plan, sys.executable, None)[0], inputs)
                with self.assertRaises(RuntimeError): audit.validate_report(report_path, dict(plan, project_id=2), sys.executable, None)
                hashes[str(root / 'weights')] = '0' * 64
                with self.assertRaisesRegex(RuntimeError, 'Original pretrained'): audit.collect_inputs(request)
                hashes[str(root / 'weights')] = serious.PRETRAINED_SHA256
                with self.assertRaisesRegex(RuntimeError, 'ZIP manifest'):
                    audit.collect_inputs(dict(request, zip_manifest_sha256='0' * 64))
                (root / 'new').mkdir()
                with self.assertRaisesRegex(RuntimeError, 'never-used'): audit.collect_inputs(request)

    def test_dedicated_wrapper_resources_and_byte_attestation(self):
        path = ROOT / 'scripts' / serious.BATCH_SCRIPT
        subprocess.run(['bash', '-n', str(path)], check=True)
        batch = path.read_text()
        for directive in ('account=thes2452', 'partition=c23g', 'nodes=1', 'ntasks=1',
                          'gres=gpu:hopper:1', 'cpus-per-task=12', 'mem=96G', 'time=23:45:00', 'signal=B:USR1@1800'):
            self.assertIn('#SBATCH --' + directive + '\n', batch)
        self.assertIn('--serious-last2', batch)
        fields = experimental_tests.AllocationTest.fields(self)
        fields['JobName'] = 'xr1_last2_serious'
        def validate(memory, supplied=batch):
            def output(cmd, **kwargs):
                if cmd[0] == 'date': return '10000'
                if cmd[0] == 'nvidia-smi': return memory if '--query-gpu=memory.total' in cmd else 'NVIDIA H100\n'
                if cmd[:3] == ['scontrol', 'write', 'batch_script']: return supplied.encode()
                raise AssertionError(cmd)
            with patch.dict(os.environ, SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='0'), \
                 patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(stdout=' '.join(k + '=' + v for k, v in fields.items()))), \
                 patch.object(launch.subprocess, 'check_output', side_effect=output), \
                 patch.object(launch.time, 'time', return_value=100), \
                 patch('claix_b24_filesystem.inspect_directory'), contextlib.redirect_stdout(io.StringIO()):
                return launch.allocation_deadline('experimental', checkpoint_interval=500, serious_last2=True)
        self.assertEqual(validate('95826\n'), 8200)
        for memory in ('81920', '81559', 'N/A', '95826\n95826'):
            with self.assertRaises(RuntimeError): validate(memory)
        with self.assertRaisesRegex(RuntimeError, 'wrapper'): validate('95826', batch + '\n')
        with self.assertRaisesRegex(RuntimeError, 'wrapper'): validate('95826', batch.replace('\n', '\r\n'))
        fields['JobName'] = 'renamed'
        with self.assertRaises(RuntimeError): validate('95826')

    def test_storage_budget_accounts_for_nine_pins_final_recovery_temporary_and_backlog(self):
        budget = serious.storage_budget()
        self.assertEqual(budget['retained_milestones'], 9)
        self.assertEqual(budget['payload_slots'], 22)
        self.assertGreater(budget['required_free_bytes'], 70 * 1024**3)
        self.assertGreater(serious.storage_budget(4 * 1024**3)['required_free_bytes'], budget['required_free_bytes'])
        self.assertEqual(storage.MIN_FREE_BYTES, 30 * 1024**3)

    def test_storage_requires_full_budget_from_user_project_and_filesystem(self):
        original = dict(status='PASS', checks=dict(project_accounting=True, uid_quota=True, pid_quota=True),
            quota_measurements=[dict(identity='uid', remaining_bytes=100 * 1024**3, remaining_inodes=20000),
                                dict(identity='pid', remaining_bytes=100 * 1024**3, remaining_inodes=20000)],
            available_bytes=100 * 1024**3, available_inodes=20000)
        with tempfile.TemporaryDirectory() as folder:
            for key in ('success', 'filesystem', 'user', 'project', 'inodes'):
                row = copy.deepcopy(original)
                if key == 'filesystem': row['available_bytes'] = 40 * 1024**3
                if key == 'inodes': row['available_inodes'] = 9999
                if key in ('user', 'project'): row['quota_measurements'][key == 'project']['remaining_bytes'] = 40 * 1024**3
                with patch.object(storage, 'experimental_evaluate', return_value=row):
                    result = storage.serious_evaluate(Path(folder) / 'new', 'synthetic', 1)
                self.assertEqual(result['status'], 'PASS' if key == 'success' else 'FAIL')

    def test_continuation_budget_uses_real_receipts_and_existing_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / 'project_xr1_doosan' / serious.NAME
            sizes = []
            for step in (5000, 10000, 10500):
                receipt = ck.publish_checkpoint(payload(root, step), target / f'step-{step:06d}.ckpt')
                sizes.append(receipt['bytes'])
            row = dict(checks=dict(project_accounting=True), quota_measurements=[],
                       available_bytes=100 * 1024**3, available_inodes=20000)
            with patch.object(storage, 'experimental_evaluate', return_value=row):
                result = storage.serious_evaluate(root, 'synthetic', 1)
            budget = result['budget']
            self.assertEqual(budget['measured_receipt_max_bytes'], max(sizes))
            self.assertEqual(budget['existing_pinned_bytes'], sum(sizes[:2]))
            self.assertEqual(budget['required_free_bytes'], serious.storage_budget(max(sizes))['required_free_bytes'] - sum(sizes[:2]))

    def test_duplicate_active_run_and_unknown_queue_fail_closed(self):
        with patch.object(audit.subprocess, 'check_output', return_value='123|xr1_last2_serious\n'):
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'): audit.check_active_runs()
            audit.check_active_runs('123')
        with patch.object(audit.subprocess, 'check_output', return_value='bad queue output'):
            with self.assertRaises(RuntimeError): audit.check_active_runs()

    def test_cpu_acceptance_never_accepts_stubbed_or_foreign_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'acceptance.json'
            path.write_text(json.dumps(dict(schema='claix-last2-serious-cpu-acceptance-v1', status='PASS', framework_stubs=True)))
            with self.assertRaisesRegex(RuntimeError, 'real-framework'): audit.check_cpu_acceptance(path, {})

    def test_historical_recipes_and_completed_wrapper_unchanged(self):
        for name in ('scripts/claix_b24_sync500_last2.sbatch', 'scripts/claix_b24_combined_qualification.sbatch',
                     'xr1/configs/claix_b24_checkpoint500.yaml', 'xr1/configs/claix_b24_production.yaml'):
            self.assertEqual((ROOT / name).read_bytes(), subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT))
        self.assertEqual(ck.HISTORICAL_STEPS, frozenset((2000, 5000, 10000, 20000, 30000, 40000)))


if __name__ == '__main__':
    unittest.main()
