"""Local-filesystem compact checkpoint transactions. Only load trusted pickle files."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from mibot.utils.doosan_replay import validate_full_compact_resume


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('CUDA RNG topology mismatch')
        torch.cuda.set_rng_state_all(state['cuda'])


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_production_checkpoint(ck, contract, *, expected_config=None, resumable=True):
    if ck.get('xr1_replay_contract') != contract:
        raise RuntimeError('Missing or inconsistent checkpoint replay contract')
    validate_full_compact_resume(ck, contract, resumable=resumable)
    meta = ck['xr1_compact_fp32']
    for field in ('pretrained_sha256', 'language_sha256'):
        value = meta.get(field, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Missing immutable compact provenance: ' + field)
    if (not meta.get('source_commit') or meta.get('last_n') != 1
            or meta.get('trainable_storage_dtype') != 'torch.float32'
            or meta.get('frozen_weights_source') != 'external_original_pretrained_checkpoint'
            or meta.get('optimizer') != 'torch.optim.AdamW with fp32 exp_avg and exp_avg_sq'):
        raise RuntimeError('Incomplete approved compact metadata')
    try:
        loop = ck['loops']['fit_loop']
        completed = loop['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed']
        batches = loop['epoch_loop.batch_progress']['total']
        current = loop['epoch_loop.batch_progress']['current']
        if (completed != ck['global_step'] or batches['processed'] != completed
                or batches['ready'] != completed or batches['started'] != completed
                or batches['completed'] != completed - 1 or current != batches
                or ck.get('epoch') != 0):
            raise RuntimeError('Checkpoint is not at an on_train_batch_end optimizer boundary')
    except (KeyError, TypeError) as error:
        raise RuntimeError('Incomplete Lightning optimizer/batch loop progress') from error
    runtime = ck.get('xr1_production', {})
    if runtime.get('schema') != 'claix-b24-runtime-v1':
        raise RuntimeError('Missing production checkpoint metadata')
    fingerprint = runtime.get('config_sha256', '')
    if (len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint)
            or (expected_config is not None and fingerprint != expected_config)):
        raise RuntimeError('Incompatible resolved resume configuration')
    states = runtime.get('rng_by_rank')
    if not isinstance(states, list) or len(states) != contract['world_size']:
        raise RuntimeError('Missing per-rank RNG state')
    for state in states:
        if set(state) != {'python', 'numpy', 'torch', 'cuda'} or not isinstance(state['torch'], torch.Tensor):
            raise RuntimeError('Incomplete rank RNG state')
        if (state['torch'].dtype != torch.uint8 or state['torch'].ndim != 1
                or not isinstance(state['cuda'], list)):
            raise RuntimeError('Invalid rank RNG state')
        try:
            random.Random().setstate(state['python'])
            np.random.RandomState().set_state(state['numpy'])
            torch.Generator().set_state(state['torch'].cpu())
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError('Unreadable rank RNG state') from error
    # Explicit upper bound excludes accidental serialization of the 5B backbone.
    # FP32 masters + two FP32 moments = 12 bytes/element, plus metadata/headroom.
    return 12 * int(ck['xr1_compact_fp32']['trainable_parameter_count']) + 64 * 1024**2


def publish_checkpoint(ck, path):
    """Read back and verify before publishing immutable payload, receipt, then pointer.

    A crash between payload and receipt leaves an orphan, never a valid resume.
    Existing committed checkpoints are never overwritten. Rotation happens later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.verified.json').exists():
        raise FileExistsError(f'Checkpoint destination already exists: {path}')
    ceiling = validate_production_checkpoint(ck, ck['xr1_replay_contract'], resumable=False)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(ck, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if Path(tmp).stat().st_size > ceiling:
            raise RuntimeError('Compact checkpoint exceeds expected size ceiling')
        reread = torch.load(tmp, map_location='cpu', weights_only=False)
        validate_production_checkpoint(reread, ck['xr1_replay_contract'], resumable=False)
        receipt = dict(schema='claix-b24-verified-v1', path=path.name,
                       global_step=int(ck['global_step']), bytes=Path(tmp).stat().st_size,
                       sha256=file_sha256(tmp), config_sha256=ck['xr1_production']['config_sha256'])
        # Atomic no-clobber publication, even if another writer bypassed flock.
        os.link(tmp, path)
        os.unlink(tmp)
        fsync_directory(path.parent)
        atomic_json(path.with_suffix('.verified.json'), receipt)
        atomic_json(path.parent / 'latest.json', receipt)
        return receipt
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_verified(path, contract, *, expected_config=None, resumable=True):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.verified.json').read_text())
    if (receipt.get('schema') != 'claix-b24-verified-v1' or receipt.get('path') != path.name
            or receipt.get('bytes') != path.stat().st_size or receipt.get('sha256') != file_sha256(path)):
        raise RuntimeError('Checkpoint receipt or checksum mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    validate_production_checkpoint(ck, contract, expected_config=expected_config, resumable=resumable)
    if receipt['global_step'] != ck['global_step']:
        raise RuntimeError('Receipt step mismatch')
    if receipt.get('config_sha256') != ck['xr1_production']['config_sha256']:
        raise RuntimeError('Receipt configuration mismatch')
    return ck


HISTORICAL_STEPS = frozenset((2000, 5000, 10000, 20000, 30000, 40000))


def rotate_checkpoints(directory, keep=2):
    """Prune only verified owned payloads; keep all receipts as audit evidence.

    Publication and rotation require the launcher's exclusive run-root lock.
    Invalid/orphan/unrelated files are never deletion candidates.
    """
    if keep < 2:
        raise ValueError('Keep at least two recovery checkpoints')
    directory = Path(directory)
    latest = json.loads((directory / 'latest.json').read_text())
    rows = []
    for path in directory.glob('step-*.verified.json'):
        try:
            if path.is_symlink():
                continue
            row = json.loads(path.read_text())
            step = row['global_step']
            name = f'step-{step:06d}.ckpt'
            payload = directory / name
            if (type(step) is not int or step <= 0 or row.get('path') != name
                    or path.name != f'step-{step:06d}.verified.json'
                    or row.get('schema') != 'claix-b24-verified-v1'
                    or row.get('config_sha256') != latest.get('config_sha256')
                    or payload.is_symlink() or not payload.is_file()
                    or row.get('bytes') != payload.stat().st_size):
                continue
            rows.append((step, payload, row))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    rows.sort(key=lambda item: item[0], reverse=True)
    # Fail closed before any deletion if publication's pointer is inconsistent.
    if not rows or latest != rows[0][2] or latest['sha256'] != file_sha256(rows[0][1]):
        raise RuntimeError('latest.json does not identify the newest verified checkpoint')
    protected = set(HISTORICAL_STEPS)
    stop_path = directory / 'safe-stop.json'
    if stop_path.exists():
        stop = json.loads(stop_path.read_text())
        matches = [step for step, _, row in rows if row == stop]
        if not matches:
            raise RuntimeError('safe-stop.json does not identify a retained checkpoint')
        protected.update(matches)
    # Validate recovery candidates before counting them; a damaged payload must
    # never displace the previous two usable checkpoints.
    recovery = 0
    for step, payload, row in rows:
        if step in protected and recovery >= keep:
            continue
        try:
            ck = torch.load(payload, map_location='cpu', weights_only=False, mmap=True)
            load_verified(payload, ck['xr1_replay_contract'],
                          expected_config=latest['config_sha256'], resumable=False)
            del ck
        except Exception:
            # Corrupt/unverified payloads remain available for investigation.
            continue
        recovery += 1
        if recovery > keep and step not in protected:
            payload.unlink()
    fsync_directory(directory)
