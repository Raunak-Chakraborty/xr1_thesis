"""Closed serious-run contract; no framework, allocation or training side effects."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess

NAME = 'xr1_doosan_b24_last2_sync500_serious_v1'
VARIANT = 'last2_sync500_serious_v1'
CONFIG = 'claix_b24_last2_sync500_serious'
BATCH_SCRIPT = 'claix_b24_last2_sync500_serious.sbatch'
HORIZON = 48284
PARAMETERS = 271570944
MIN_GPU_BYTES = 90 * 1024**3
PRETRAINED_SHA256 = '94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7'
LANGUAGE_SHA256 = '21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d'
ROOT = Path(__file__).resolve().parents[3]
STACKS = {'model.vlm.model.visual.blocks.': {22, 23},
          'model.vlm.model.language_model.layers.': {34, 35},
          'model.dit.layers.': {34, 35}}
INTERFACES = ('state_projector', 'action_projector', 'action_output_layer',
              't_embedder', 't_projector', 'sink')


def recipe(name=CONFIG):
    """Read the repository's global-only YAML inheritance without importing Hydra.

    Actual Hydra composition is separately required by CLAIX CPU acceptance.
    This deliberately rejects interpolation and unsupported default forms.
    """
    import yaml
    path = ROOT / 'xr1/configs' / (name + '.yaml')
    raw = path.read_text()
    if '${' in raw or '# @package _global_' not in raw:
        raise RuntimeError('Unsupported serious recipe composition')
    value = yaml.safe_load(raw)
    result = {}
    def merge(dst, src):
        for key, val in src.items():
            if isinstance(val, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], val)
            else:
                dst[key] = copy.deepcopy(val)
    for item in value.pop('defaults', []):
        if item == '_self_':
            continue
        if isinstance(item, dict) and len(item) == 1:
            key, val = next(iter(item.items()))
            item = key + '/' + val
        if not isinstance(item, str) or '..' in item or item.startswith('/'):
            raise RuntimeError('Unsupported serious recipe default')
        merge(result, recipe(item))
    merge(result, value)
    return result


def validate_science(conf, production):
    """Only paths, run UUID, checkpoint cursor and safe-stop request may vary."""
    from mibot.utils.claix_checkpoint import config_digest
    actual = copy.deepcopy(dict(conf))
    actual['production'] = copy.deepcopy(dict(production))
    expected = recipe()
    for cfg in (actual, expected):
        cfg['data']['params']['train_datasets']['paths'] = []
        for key in ('pretrained', 'language_policy_path'):
            cfg['model']['params'][key] = None
        for key in ('default_root_dir', 'ckpt_path'):
            cfg['trainer'][key] = None
        cfg['production']['experimental_run_id'] = None
        cfg['production']['stop_after_step'] = None
    if actual != expected:
        raise RuntimeError('Serious scientific configuration differs from the closed recipe')
    return config_digest(actual)


def source_identity():
    from mibot.utils.claix_contract import engineering_source_digest
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                    cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError('Serious run requires a clean committed source checkout')
    return dict(source_commit=commit, source_tree_sha256=engineering_source_digest())


def validate_identity(identity):
    from mibot.utils.claix_contract import FROZEN_TRAIN_SHA256
    required = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                    serious_variant=VARIANT, checkpoint_interval=500, experiment_name=NAME,
                    pretrained_sha256=PRETRAINED_SHA256, language_sha256=LANGUAGE_SHA256,
                    dataset_sha256=FROZEN_TRAIN_SHA256)
    if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in required.items()):
        raise RuntimeError('Invalid serious last-two experiment identity')
    for field, size in [('run_id', 32), ('source_commit', 40), ('source_tree_sha256', 64),
                        ('scientific_config_sha256', 64), ('inputs_sha256', 64),
                        ('zip_manifest_sha256', 64)]:
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != size or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Invalid serious identity: ' + field)
    for key in ('run_root', 'zip_cache_root'):
        if not isinstance(identity.get(key), str) or not Path(identity[key]).is_absolute():
            raise RuntimeError('Invalid serious identity path: ' + key)


def bind_serious(conf, production, inventory, manifest, identity):
    from mibot.utils.claix_checkpoint import config_digest
    science = validate_science(conf, production)
    if (os.environ.get('XR1_SERIOUS_LAST2') != '1'
            or os.environ.get('XR1_LAST2_VRAM_PROBE')
            or production.get('last2_vram_probe')):
        raise RuntimeError('Serious last-two training requires isolated launcher authorization')
    source = source_identity()
    plan = manifest['plan']
    inputs = plan.get('serious_inputs', {})
    if (manifest.get('git_commit') != source['source_commit']
            or manifest.get('source_tree_sha256') != source['source_tree_sha256']
            or plan.get('serious_variant') != VARIANT
            or inputs.get('scientific_config_sha256') != science
            or inputs.get('train_sha256') != inventory['sha256']
            or inputs.get('zip_manifest_sha256') != identity['zip_manifest_sha256']
            or inputs.get('zip_cache_root') != identity['zip_cache_root']):
        raise RuntimeError('Serious source/configuration/input binding mismatch')
    identity.update(serious_variant=VARIANT, **source, scientific_config_sha256=science,
                    inputs_sha256=config_digest(inputs), pretrained_sha256=PRETRAINED_SHA256,
                    language_sha256=LANGUAGE_SHA256, dataset_sha256=inventory['sha256'])
    validate_identity(identity)
    target = Path(identity['run_root']) / 'project_xr1_doosan' / NAME
    if not conf['trainer'].get('ckpt_path') and target.exists() and any(target.iterdir()):
        raise RuntimeError('Serious step-zero invocation cannot reuse existing experiment evidence')
    return identity


def validate_selected_keys(keys):
    seen = {prefix: set() for prefix in STACKS}
    interfaces = set()
    for key in keys:
        for prefix, allowed in STACKS.items():
            if key.startswith(prefix):
                part = key[len(prefix):].split('.', 1)[0]
                if not part.isdigit() or int(part) not in allowed:
                    raise RuntimeError('Serious checkpoint contains frozen transformer keys')
                seen[prefix].add(int(part))
                break
        else:
            module = key.split('.')[1] if key.startswith('model.') else ''
            if module not in INTERFACES:
                raise RuntimeError('Serious checkpoint contains unapproved interface keys')
            interfaces.add(module)
    if seen != STACKS or interfaces != set(INTERFACES):
        raise RuntimeError('Serious checkpoint is missing selected stacks/interfaces')


def validate_checkpoint(ck, identity):
    from mibot.utils.claix_contract import BASELINE
    from mibot.utils.doosan_replay import make_contract
    validate_identity(identity)
    meta = ck['xr1_compact_fp32']
    if (meta.get('last_n') != 2 or meta.get('pretrained_sha256') != PRETRAINED_SHA256
            or meta.get('language_sha256') != LANGUAGE_SHA256 or meta.get('source_commit') != BASELINE
            or ck['xr1_replay_contract'] != make_contract(data_horizon=HORIZON, batch_size=24,
                world_size=1, sampler_seed=42, num_workers=4)
            or ck['xr1_production'].get('qualification_last2_vram_probe') is not None):
        raise RuntimeError('Serious compact provenance/replay mismatch')
    validate_selected_keys(ck['state_dict'])
    # LambdaLR must retain all state needed by Lightning; the closure is rebuilt
    # from the identical source/configuration, never from a shortened horizon.
    scheduler = ck['lr_schedulers'][0]
    step = ck['global_step']
    progress = float(step - 500) / float(HORIZON - 500)
    expected_lr = (min(2e-5, 5e-7 + (2e-5 - 5e-7) * step / 500) if step < 500
                   else (2e-5 - 5e-6) * .5 * (1. + math.cos(math.pi * progress)) + 5e-6)
    required = {'last_epoch', 'base_lrs', '_last_lr', '_step_count', 'lr_lambdas'}
    if (not required.issubset(scheduler) or scheduler['_step_count'] != ck['global_step'] + 1
            or len(scheduler['base_lrs']) != len(ck['optimizer_states'][0]['param_groups'])
            or len(scheduler['_last_lr']) != len(scheduler['base_lrs'])
            or scheduler['base_lrs'] != [1.] * len(scheduler['base_lrs'])
            or scheduler['_last_lr'] != [expected_lr] * len(scheduler['base_lrs'])
            or scheduler['lr_lambdas'] != [None] * len(scheduler['base_lrs'])):
        raise RuntimeError('Incomplete serious cosine scheduler state')
    for group in ck['optimizer_states'][0]['param_groups']:
        if (not {'params', 'lr', 'initial_lr', 'betas', 'eps', 'weight_decay', 'amsgrad',
                 'maximize', 'capturable', 'differentiable'}.issubset(group)
                or tuple(group['betas']) != (0.9, 0.95) or group['eps'] != 1e-8
                or group['weight_decay'] not in (0.0, 0.1) or group['amsgrad'] is not False
                or group['initial_lr'] != 1.0 or group['lr'] != expected_lr
                or any(group[key] is not False for key in ('maximize', 'capturable', 'differentiable'))):
            raise RuntimeError('Incomplete or changed serious AdamW state')
    groups = ck['optimizer_states'][0]['param_groups']
    keys = ck['xr1_production'].get('optimizer_parameter_keys')
    if (not isinstance(keys, list) or len(keys) != len(groups)
            or any(not isinstance(row, list) or len(row) != len(group['params'])
                   for row, group in zip(keys, groups))):
        raise RuntimeError('Missing serious AdamW parameter key mapping')
    flat = [key for row in keys for key in row]
    if len(flat) != len(set(flat)) or set(flat) != set(ck['state_dict']):
        raise RuntimeError('Serious AdamW selected parameter key mismatch')
    for names, group in zip(keys, groups):
        for name, index in zip(names, group['params']):
            no_decay = any(word in name.lower() for word in ('bias', 'norm', 'ln', 'rotary_emb', 'adaln'))
            if group['weight_decay'] != (0.0 if no_decay else 0.1):
                raise RuntimeError('Serious AdamW weight decay/key mapping mismatch')
            state = ck['optimizer_states'][0]['state'][index]
            if state['exp_avg'].shape != ck['state_dict'][name].shape:
                raise RuntimeError('Serious AdamW/master parameter shape mismatch')


def check_milestones(directory, step, *, expected_config, expected_experiment):
    """Bounded receipt/stat checks only; no historical payload deserialization."""
    from mibot.utils.claix_checkpoint import SERIOUS_LAST2_MILESTONES, _read_receipt
    directory = Path(directory)
    for milestone in (*SERIOUS_LAST2_MILESTONES, HORIZON):
        if milestone > step:
            continue
        path = directory / f'step-{milestone:06d}.ckpt'
        try:
            row = _read_receipt(path.with_suffix('.verified.json'))
            if (path.is_symlink() or not path.is_file()
                    or row.get('schema') != 'claix-b24-verified-v1'
                    or row.get('path') != path.name or row.get('global_step') != milestone
                    or row.get('config_sha256') != expected_config
                    or row.get('experiment') != expected_experiment
                    or type(row.get('bytes')) is not int or row['bytes'] != path.stat().st_size
                    or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in row['sha256'])):
                raise ValueError('Invalid milestone receipt/payload')
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise RuntimeError(f'Missing/invalid serious milestone {milestone}') from error


def storage_budget(receipt_bytes=0):
    from mibot.utils.claix_checkpoint import ROTATION_VERIFY_BUDGET
    # Count separately even when final/recovery/safe-stop overlap. Reserve for a
    # full bounded rotation backlog plus serialization; never delete to fit.
    ceiling = 12 * PARAMETERS + 64 * 1024**2
    size = max(ceiling, receipt_bytes)
    slots = 9 + 1 + 2 + 1 + 1 + ROTATION_VERIFY_BUDGET
    return dict(checkpoint_bytes=size, retained_milestones=9, payload_slots=slots,
                required_free_bytes=slots * size + 4 * 1024**3, required_free_inodes=10000)


def before_publication(directory, step, fingerprint, identity):
    """Fail before serialization on missing pins or insufficient write headroom."""
    from mibot.utils.claix_checkpoint import _read_receipt
    check_milestones(directory, step - 1, expected_config=fingerprint, expected_experiment=identity)
    latest = Path(directory) / 'latest.json'
    measured = _read_receipt(latest)['bytes'] if latest.exists() else 0
    budget = storage_budget(measured)
    stat = os.statvfs(directory)
    if stat.f_bavail * stat.f_frsize < 2 * budget['checkpoint_bytes'] + 4 * 1024**3 or stat.f_favail < 10000:
        raise RuntimeError('Serious checkpoint serialization headroom exhausted; preserve evidence')


def check_resume_evidence(target, receipt):
    from mibot.utils.claix_checkpoint import _read_receipt, load_verified
    target = Path(target)
    step = receipt['global_step']
    for path in target.glob('step-*.verified.json'):
        row = _read_receipt(path)
        if type(row.get('global_step')) is not int or row['global_step'] > step:
            raise RuntimeError('Serious receipt ahead of latest verified; no rollback')
    for path in target.glob('step-*.ckpt'):
        if not path.stem[5:].isdigit() or int(path.stem[5:]) > step:
            raise RuntimeError('Serious payload ahead of latest verified; no rollback')
    with (target / 'telemetry.jsonl').open() as stream:
        first = json.loads(stream.readline(65537))
    if first.get('event') != 'train_start' or first.get('step') != 0:
        raise RuntimeError('Serious telemetry must originate at pretrained step zero')
    stop = target / 'safe-stop.json'
    if os.path.lexists(stop):
        row = _read_receipt(stop)
        name = f"step-{row['global_step']:06d}.ckpt"
        if row.get('path') != name or row != _read_receipt(target / name.replace('.ckpt', '.verified.json')):
            raise RuntimeError('Invalid serious safe-stop evidence')
        ck = load_verified(target / name, expected_config=receipt['config_sha256'],
                           expected_experiment=receipt['experiment'], resumable=False)
        del ck
