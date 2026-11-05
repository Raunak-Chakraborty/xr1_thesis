"""Local-filesystem compact checkpoint transactions. Only load trusted pickle files."""
from __future__ import annotations

import hashlib
import heapq
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


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError('Refusing symlink JSON destination')
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(tmp, path)
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_production_checkpoint(ck, contract, *, expected_config=None, resumable=True, expected_experiment=None):
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
    experiment = runtime.get('experiment')
    if expected_experiment is not None and experiment != expected_experiment:
        raise RuntimeError('Checkpoint does not belong to this experimental run')
    if experiment is not None:
        if (not isinstance(experiment, dict)
                or experiment.get('schema') != 'claix-b24-experimental-checkpoint-v1'
                or experiment.get('experimental_unqualified') is not True
                or len(experiment.get('run_id', '')) != 32
                or len(experiment.get('zip_manifest_sha256', '')) != 64
                or not Path(experiment.get('run_root', '')).is_absolute()):
            raise RuntimeError('Incomplete experimental checkpoint identity')
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
    if os.path.lexists(path) or os.path.lexists(path.with_suffix('.verified.json')):
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
        if ck['xr1_production'].get('experiment') is not None:
            receipt['experiment'] = dict(ck['xr1_production']['experiment'])
        # Atomic no-clobber publication, even if another writer bypassed flock.
        os.link(tmp, path)
        os.unlink(tmp)
        fsync_directory(path.parent)
        atomic_json(path.with_suffix('.verified.json'), receipt, exclusive=True)
        atomic_json(path.parent / 'latest.json', receipt)
        return receipt
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_receipt(path):
    # Receipts/pointers are tiny metadata. Unexpected large/special files must
    # not consume arbitrary memory or block on a FIFO during retention scans.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Receipt must be a regular non-symlink file')
    with path.open('rb') as stream:
        raw = stream.read(16*1024 + 1)
    if len(raw) > 16*1024:
        raise ValueError('Oversized checkpoint receipt')
    return json.loads(raw)


def load_verified(path, contract=None, *, expected_config=None, resumable=True, expected_experiment=None):
    path = Path(path)
    if path.is_symlink() or path.with_suffix('.verified.json').is_symlink():
        raise RuntimeError('Symlink checkpoint/receipt refused')
    receipt = _read_receipt(path.with_suffix('.verified.json'))
    if expected_experiment is not None and receipt.get('experiment') != expected_experiment:
        raise RuntimeError('Checkpoint receipt does not belong to this experimental run')
    if (receipt.get('schema') != 'claix-b24-verified-v1' or receipt.get('path') != path.name
            or receipt.get('bytes') != path.stat().st_size or receipt.get('sha256') != file_sha256(path)):
        raise RuntimeError('Checkpoint receipt or checksum mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    validate_production_checkpoint(ck, contract if contract is not None else ck['xr1_replay_contract'],
                                   expected_config=expected_config, resumable=resumable,
                                   expected_experiment=expected_experiment)
    if receipt['global_step'] != ck['global_step']:
        raise RuntimeError('Receipt step mismatch')
    if receipt.get('config_sha256') != ck['xr1_production']['config_sha256']:
        raise RuntimeError('Receipt configuration mismatch')
    if receipt.get('experiment') != ck['xr1_production'].get('experiment'):
        raise RuntimeError('Receipt experimental identity mismatch')
    return ck


HISTORICAL_STEPS = frozenset((2000, 5000, 10000, 20000, 30000, 40000))


ROTATION_VERIFY_BUDGET = 8


def rotate_checkpoints(directory, keep=2, *, expected_config=None):
    """Receipt-selected retention under the launcher's exclusive run-root lock.

    Scan small receipts only, keeping at most eight ordinary rows and seven
    pins. Verify at most eight distinct payloads, one at a time, including newest, recovery, safe-stop and deletion candidates. Old
    milestones require no payload I/O unless needed for recovery. Backlogs are
    reclaimed incrementally; corrupt/unverified files are never deleted. Stat
    size is only a filter, never integrity evidence. No persistent trust cache.
    """
    if keep < 2 or keep >= ROTATION_VERIFY_BUDGET:
        raise ValueError('Keep between two and seven recovery checkpoints')
    directory = Path(directory)
    def pointer(name):
        path = directory / name
        if path.is_symlink():
            raise RuntimeError('Symlink retention pointer refused')
        return _read_receipt(path)
    latest = pointer('latest.json')
    if not isinstance(latest, dict) or (expected_config is not None
            and latest.get('config_sha256') != expected_config):
        raise RuntimeError('Incompatible latest checkpoint pointer')
    has_stop = os.path.lexists(directory / 'safe-stop.json')
    stop = pointer('safe-stop.json') if has_stop else None
    if has_stop and not isinstance(stop, dict):
        raise RuntimeError('Malformed safe-stop pointer')
    recent = []
    pinned = {}
    committed_count = 0
    newest_receipt = None
    for path in directory.glob('step-*.verified.json'):
        try:
            if path.is_symlink():
                continue
            row = _read_receipt(path)
            if not isinstance(row, dict):
                continue
            step = row['global_step']
            if type(step) is not int or step <= 0:
                continue
            name = f'step-{step:06d}.ckpt'
            payload = directory / name
            if (row.get('path') != name or path.name != f'step-{step:06d}.verified.json'
                    or row.get('schema') != 'claix-b24-verified-v1'
                    or row.get('config_sha256') != latest.get('config_sha256')
                    or type(row.get('bytes')) is not int
                    or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64):
                continue
            committed_count += 1
            if newest_receipt is None or step > newest_receipt['global_step']:
                newest_receipt = row
            if (payload.is_symlink() or not payload.is_file()
                    or row['bytes'] != payload.stat().st_size):
                continue
            item = (step, payload, row)
            if step in HISTORICAL_STEPS or row == stop:
                pinned[step] = item
            else:
                heapq.heappush(recent, item)
                if len(recent) > ROTATION_VERIFY_BUDGET:
                    heapq.heappop(recent)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    rows = sorted([*recent, *pinned.values()], key=lambda item: item[0], reverse=True)
    if not rows or latest != rows[0][2] or latest != newest_receipt:
        raise RuntimeError('latest.json does not identify the newest verified checkpoint')
    checked = {}
    def verify(item):
        step, payload, row = item
        if step not in checked:
            if len(checked) >= ROTATION_VERIFY_BUDGET:
                raise RuntimeError('Rotation verification budget exhausted; operator recovery review required')
            try:
                # SHA256 followed by one full deserialization and validation.
                ck = load_verified(payload, expected_config=latest['config_sha256'], resumable=False)
                del ck
                checked[step] = True
            except Exception:
                checked[step] = False
        return checked[step]
    if not verify(rows[0]):
        raise RuntimeError('Newest checkpoint failed verification; no rotation allowed')
    protected = set(HISTORICAL_STEPS)
    if stop is not None:
        match = next((item for item in rows if item[2] == stop), None)
        if match is None or not verify(match):
            raise RuntimeError('safe-stop.json does not identify a verified retained checkpoint')
        protected.add(match[0])
    recovery = []
    for item in rows:
        if verify(item):
            recovery.append(item[0])
            if len(recovery) == keep:
                break
    # The first ever publication may have fewer than two checkpoints. Never
    # delete anything in that case; once two are available require both usable.
    if len(recovery) < keep:
        if committed_count > len(recovery):
            raise RuntimeError('Cannot establish required usable recovery checkpoints')
        return dict(verified=len(checked), deleted=0)
    protected.update(recovery)
    deletions = []
    for item in rows:
        if item[0] in protected:
            continue
        if item[0] not in checked and len(checked) >= ROTATION_VERIFY_BUDGET:
            break
        if verify(item):
            deletions.append(item[1])
    # All required recovery/pointer checks finish before the first unlink.
    for payload in deletions:
        payload.unlink()
    fsync_directory(directory)
    return dict(verified=len(checked), deleted=len(deletions))
