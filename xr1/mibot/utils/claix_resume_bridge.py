"""Explicit, narrowly scoped bridge for the original verified XR-1 step-21500 checkpoint.

Does not edit the original checkpoint. New checkpoint publications use the new
source-derived fingerprint, while only the initial restore accepts the
already verified original fingerprint. Not a general bypass mechanism.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from mibot.utils.claix_contract import engineering_source_digest

BASE_COMMIT = '527be264ce6a5b462468c0a406f1a876c8bbdfa4'
OLD_SOURCE_SHA256 = '852945709dac80152459c26ab5dab5e70e158b631ea27612d28ee2d172ef3d0b'
OLD_CONFIG_SHA256 = '87ec83b0200ec9f8ab9e3cafa75079dd0d7881dcea0913c0d03ca0a1571687ea'
CKPT_SHA256 = '48fae2e1881e85ef8d760b30780ea21aa146053b1702b1084c56fb6644d64813'
TRAIN_SHA256 = '1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548'

# Only reviewed loader instrumentation and the explicit resume bridge itself.
APPROVED_PROTECTED_FILES = frozenset({
    'xr1/mibot/data/datamodule/doosan_datamodule.py',
    'xr1/mibot/data/datasets/doosan_json_dataset.py',
    'xr1/mibot/data/datasets/doosan_zip_images.py',
    'xr1/mibot/utils/loader_diagnostics.py',
    'xr1/mibot/utils/loader_trace.py',
    'xr1/mibot/utils/loader_watchdog.py',
    'xr1/mibot/utils/claix_resume_bridge.py',
    'xr1/mibot/models/runner/doosan_compact_runner.py',
    'xr1/tools/train_doosan_compact.py',
    'scripts/claix_b24_launch.py',
    'scripts/claix_b24_production.sbatch',
})


def authorize_original_resume(checkpoint, new_fingerprint, inventory, contract):
    choice = os.environ.get('XR1_OPTION_B_RESUME_BRIDGE')
    if choice is None:
        return None
    if choice != 'EXPLICIT_STEP_21500':
        raise RuntimeError('Unrecognized Option B bridge approval')
    if not isinstance(new_fingerprint, str) or len(new_fingerprint) != 64 or new_fingerprint == OLD_CONFIG_SHA256:
        raise RuntimeError('Unexpected candidate fingerprint')
    if inventory.get('sha256') != TRAIN_SHA256 or inventory.get('episodes') != 255 or inventory.get('frames') != 289703:
        raise RuntimeError('Frozen train inventory changed')
    if (contract.get('schema') != 'xr1-doosan-fixed-horizon-replay-v2'
            or contract.get('batch_size') != 24 or contract.get('data_horizon') != 48284
            or contract.get('world_size') != 1 or contract.get('num_workers') != 4
            or contract.get('sampler_seed') != 42 or contract.get('accumulate_grad_batches') != 1):
        raise RuntimeError('Frozen replay contract changed')
    ck = Path(checkpoint)
    if ck.name != 'step-021500.ckpt' or ck.is_symlink() or not ck.is_file():
        raise RuntimeError('Bridge accepts only the original step-21500 checkpoint')
    receipt_path = ck.with_suffix('.verified.json')
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_size > 16384:
        raise RuntimeError('Missing original verified receipt')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'claix-b24-verified-v1'
            or receipt.get('path') != ck.name or receipt.get('global_step') != 21500
            or receipt.get('sha256') != CKPT_SHA256
            or receipt.get('config_sha256') != OLD_CONFIG_SHA256
            or receipt.get('bytes') != ck.stat().st_size):
        raise RuntimeError('Original checkpoint receipt identity drift')
    # load_verified subsequently hashes all checkpoint bytes and validates model,
    # optimizer, RNG and scheduler state against this receipt and old config.
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=8).strip()
    if git('status', '--porcelain'):
        raise RuntimeError('Option B bridge requires an unmodified source checkout')
    commit = git('rev-parse', 'HEAD')
    if not commit or commit == BASE_COMMIT:
        raise RuntimeError('Option B bridge requires a new reviewed source commit')
    subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', BASE_COMMIT, commit],
                   check=True, timeout=8)
    modified = set(git('diff', '--name-only', BASE_COMMIT, commit).splitlines())
    protected = {name for name in modified if name.startswith('xr1/') or name.startswith('scripts/claix_b24')}
    if protected != APPROVED_PROTECTED_FILES:
        raise RuntimeError('Unexpected protected source delta: ' + repr(sorted(protected ^ APPROVED_PROTECTED_FILES)))
    source_sha256 = engineering_source_digest()
    if source_sha256 == OLD_SOURCE_SHA256:
        raise RuntimeError('No source change to bridge')
    return dict(
        schema='xr1-option-b-step21500-cross-source-restore-v1',
        status='USER_SELECTED_OPTION_B_SOURCE_BRIDGE',
        source_checkpoint_path=str(ck.resolve()),
        source_checkpoint_sha256=CKPT_SHA256,
        original_source_commit=BASE_COMMIT,
        original_source_sha256=OLD_SOURCE_SHA256,
        candidate_commit=commit,
        candidate_source_sha256=source_sha256,
        original_config_sha256=OLD_CONFIG_SHA256,
        candidate_config_sha256=new_fingerprint,
        frozen_training_inventory_sha256=TRAIN_SHA256,
        original_global_step=21500,
        sample_offset=516000,
        scientific_input_equivalence_cpu_job=5422806,
        original_checkpoint_modified=False,
        runtime_resume_expected_config=OLD_CONFIG_SHA256,
        new_checkpoints_expected_config=new_fingerprint,
    )
